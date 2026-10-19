"""fal.ai backend, using the official ``fal-client`` package.

Needs ``FAL_KEY`` (or ``FAL_API_KEY``) in the environment (or ``key=``). Models are fal endpoint ids,
e.g. ``blackforestlabs/flux-3/text-to-image`` or ``fal-ai/nano-banana``.

fal endpoints do not share one input format: sizes come as ``image_size``
(an object or a ``"1024x1024"`` enum), as ``aspect_ratio``, or as width/height,
and not every endpoint takes a seed. So each endpoint's published OpenAPI input
schema (public, no key needed) decides which request fields are sent; fields an
endpoint cannot take are listed under ``unsupported`` in the stored request,
and a dropped seed sets ``seed_supported = False``.
"""

import asyncio
import base64
import math
import os
import time
from typing import Any, Self

import fal_client
import httpx

from .base import (
    GenerationError,
    GenerationRequest,
    GenerationResult,
    ImageBackend,
    ModelInfo,
    register_backend,
    utc_now,
)

MODELS_API = "https://api.fal.ai/v1/models"


def input_properties(openapi: dict[str, Any]) -> dict[str, Any]:
    """The input properties of an endpoint, from its OpenAPI document."""
    for path, operations in openapi.get("paths", {}).items():
        if "post" in operations and "/requests/" not in path:
            schema = operations["post"]["requestBody"]["content"]["application/json"]["schema"]
            name = schema["$ref"].rsplit("/", 1)[-1]
            return openapi["components"]["schemas"][name].get("properties", {})
    return {}


def _enum(spec: dict[str, Any]) -> list[Any]:
    values = list(spec.get("enum", []))
    for option in spec.get("anyOf", []):
        values += option.get("enum", [])
    return values


def _accepts_object(spec: dict[str, Any]) -> bool:
    return spec.get("type") == "object" or "$ref" in spec or any(
        o.get("type") == "object" or "$ref" in o for o in spec.get("anyOf", [])
    )


def map_arguments(request: GenerationRequest, properties: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Request fields this endpoint accepts, and the names of those it cannot take."""
    arguments: dict[str, Any] = {"prompt": request.prompt}
    unsupported: list[str] = []

    def optional(name: str, value: Any, field_name: str) -> None:
        if value is None:
            return
        if field_name in properties:
            arguments[field_name] = value
        else:
            unsupported.append(name)

    optional("seed", request.seed, "seed")
    optional("negative_prompt", request.negative_prompt, "negative_prompt")
    optional("steps", request.steps, "num_inference_steps")
    optional("guidance", request.guidance, "guidance_scale")

    w, h = request.width, request.height
    if "image_size" in properties:
        spec = properties["image_size"]
        if _accepts_object(spec):
            arguments["image_size"] = {"width": w, "height": h}
        elif f"{w}x{h}" in _enum(spec):
            arguments["image_size"] = f"{w}x{h}"
        elif w == h == 1024 and "square_hd" in _enum(spec):
            arguments["image_size"] = "square_hd"
        else:
            unsupported.append("size")
    elif "aspect_ratio" in properties:
        g = math.gcd(w, h)
        ratio = f"{w // g}:{h // g}"
        if ratio in _enum(properties["aspect_ratio"]):
            arguments["aspect_ratio"] = ratio
        else:
            unsupported.append("size")
    elif "width" in properties and "height" in properties:
        arguments |= {"width": w, "height": h}
    else:
        unsupported.append("size")

    if "num_images" in properties:
        arguments["num_images"] = 1
    if "png" in _enum(properties.get("output_format", {})):
        arguments["output_format"] = "png"
    return arguments | request.extra, unsupported


@register_backend
class FalBackend(ImageBackend):
    name = "fal"

    def __init__(self, key: str | None = None):
        self._key = key or os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY")
        self._client: fal_client.AsyncClient | None = None
        self._http: httpx.AsyncClient | None = None
        self._schemas: dict[str, dict[str, Any] | None] = {}

    async def __aenter__(self) -> Self:
        self._client = fal_client.AsyncClient(key=self._key)
        self._http = httpx.AsyncClient(timeout=60)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._http is not None:
            await self._http.aclose()
        self._client = self._http = None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {self._key}"} if self._key else {}

    async def _properties(self, endpoint: str) -> dict[str, Any] | None:
        """Cached input properties of an endpoint; None if its schema is unavailable."""
        if endpoint not in self._schemas:
            # Some keys get "Failed to fetch OpenAPI schema" instead of the schema, which is
            # public, so it is asked for again without the key.
            for headers in [self._headers(), {}] if self._key else [{}]:
                openapi = await self._openapi(endpoint, headers)
                if openapi is not None and "error" not in openapi:
                    try:
                        self._schemas[endpoint] = input_properties(openapi)
                    except (KeyError, ValueError, IndexError):
                        self._schemas[endpoint] = None
                    break
            else:
                return None
        return self._schemas[endpoint]

    async def _openapi(self, endpoint: str, headers: dict[str, str]) -> dict[str, Any] | None:
        """An endpoint's OpenAPI document as the catalogue returns it; None if it could not be fetched."""
        assert self._http is not None
        params = {"endpoint_id": endpoint, "expand": "openapi-3.0"}
        # Catalogue lookups (not generations) are rate limited, especially without a
        # key, so they are retried with backoff; a failure is not cached.
        for attempt in range(5):
            try:
                response = await self._http.get(MODELS_API, params=params, headers=headers)
                if response.status_code == 429:
                    await asyncio.sleep(2**attempt)
                    continue
                response.raise_for_status()
                models = response.json().get("models", [])
                return models[0].get("openapi") if models else None
            except (httpx.HTTPError, ValueError):
                return None
        return None

    async def _download(self, url: str) -> bytes:
        if url.startswith("data:"):
            return base64.b64decode(url.split(",", 1)[1])
        assert self._http is not None
        response = await self._http.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.content

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._client is None:
            raise RuntimeError("use FalBackend as an async context manager")
        result = GenerationResult(request, self.name, utc_now(), 0.0)
        start = time.monotonic()
        properties = await self._properties(request.model)
        if properties is None:
            result.error = GenerationError("unknownModel", f"no input schema found for fal endpoint {request.model!r}")
            result.latency_s = time.monotonic() - start
            return result
        arguments, unsupported = map_arguments(request, properties)
        result.provider_request = {"endpoint": request.model, "arguments": arguments, "unsupported": unsupported}
        result.seed_supported = "seed" not in unsupported
        try:
            response = await self._client.subscribe(request.model, arguments=arguments)
            images = response.get("images") or ([response["image"]] if response.get("image") else [])
            if images:
                result.image = await self._download(images[0]["url"])
                content_type = images[0].get("content_type") or "image/png"
                result.image_format = content_type.split("/")[-1].replace("jpeg", "jpg")
            else:
                result.error = GenerationError("noImage", "response contained no image")
            result.provider_response = {k: v for k, v in response.items() if k not in ("images", "image")}
        except fal_client.FalClientHTTPError as exc:
            result.error = GenerationError(
                exc.error_type or f"http_{exc.status_code}", exc.message, exc.status_code in (429, 500, 502, 503)
            )
        except Exception as exc:  # recorded as a failed sample rather than aborting the run
            result.error = GenerationError("unknown", f"{type(exc).__name__}: {exc}")
        result.latency_s = time.monotonic() - start
        return result

    async def search_models(self, query: str, *, text_to_image_only: bool = True) -> list[ModelInfo]:
        """fal endpoints matching ``query`` (public catalogue; no key needed)."""
        assert self._http is not None
        params: dict[str, Any] = {"q": query, "status": "active", "limit": 50}
        if text_to_image_only:
            params["category"] = "text-to-image"
        found: list[ModelInfo] = []
        while len(found) < 300:
            response = await self._http.get(MODELS_API, params=params, headers=self._headers())
            response.raise_for_status()
            page = response.json()
            for m in page.get("models", []):
                meta = m.get("metadata", {})
                found.append(
                    ModelInfo(
                        id=m["endpoint_id"],
                        name=meta.get("display_name", m["endpoint_id"]),
                        architecture=meta.get("category"),
                        source=meta.get("license_type"),
                        details=meta,
                    )
                )
            if not page.get("has_more") or not page.get("next_cursor"):
                return found
            params["cursor"] = page["next_cursor"]
        return found
