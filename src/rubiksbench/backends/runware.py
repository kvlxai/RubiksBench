"""Runware.ai backend, using the official ``runware-sdk`` package.

Needs ``RUNWARE_API_KEY`` in the environment (or ``api_key=``). Models are
addressed by their AIR identifier, e.g. ``runware:400@1``.
"""

import base64
import math
import re
import time
from typing import Any, Self

import httpx
from runware import Runware, RunwareError

from .base import (
    GenerationError,
    GenerationRequest,
    GenerationResult,
    ImageBackend,
    ModelInfo,
    register_backend,
    utc_now,
)


def supported_sizes(message: str) -> list[tuple[int, int]]:
    """Sizes listed in a Runware "unsupported dimensions" message, e.g. '2048x2048'."""
    return [(int(w), int(h)) for w, h in re.findall(r"'(\d+)x(\d+)'", message)]


def pixel_range(message: str) -> tuple[int, int] | None:
    """The allowed pixel count in a Runware "Total pixels ... must be between A and B" message."""
    match = re.search(r"pixels.*?between (\d+) and (\d+)", message, re.IGNORECASE)
    return (int(match[1]), int(match[2])) if match else None


def scaled_size(width: int, height: int, low: int, high: int) -> tuple[int, int]:
    """The size of the same shape, in steps of 64 pixels, whose pixel count lies strictly between low and high."""
    factor = math.sqrt((low if width * height <= low else high) / (width * height))
    w, h = (max(64, round(width * factor / 64) * 64), max(64, round(height * factor / 64) * 64))
    while w * h <= low:
        w, h = w + 64, round((w + 64) * height / width / 64) * 64
    while w * h >= high:
        w, h = w - 64, round((w - 64) * height / width / 64) * 64
    return w, h


def closest_size(width: int, height: int, supported: list[tuple[int, int]]) -> tuple[int, int]:
    """The supported size nearest in shape (aspect ratio), then in area."""
    aspect = width / height
    return min(
        supported, key=lambda s: (abs(math.log(s[0] / s[1] / aspect)), abs(s[0] * s[1] - width * height))
    )


@register_backend
class RunwareBackend(ImageBackend):
    name = "runware"

    def __init__(self, api_key: str | None = None, transport: str = "rest", output_format: str = "PNG"):
        self._api_key = api_key
        self._transport = transport
        self._output_format = output_format
        self._client: Runware | None = None
        # Models that rejected the seed parameter (e.g. OpenAI, some BFL models).
        self._seedless: set[str] = set()
        # Models that only accept certain sizes (e.g. Ideogram 4.0), with the size to use instead.
        self._sizes: dict[str, tuple[int, int]] = {}

    async def __aenter__(self) -> Self:
        # max_retries=0: a failed request is recorded, never silently re-sent.
        self._client = Runware(api_key=self._api_key, transport=self._transport, max_retries=0)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.__aexit__(*exc_info)
            self._client = None

    def build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "taskType": "imageInference",
            "model": request.model,
            "positivePrompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "numberResults": 1,
            "outputType": "base64Data",
            "outputFormat": self._output_format,
            "includeCost": True,
        }
        optional = {
            "seed": request.seed,
            "negativePrompt": request.negative_prompt,
            "steps": request.steps,
            "CFGScale": request.guidance,
        }
        payload |= {k: v for k, v in optional.items() if v is not None}
        return payload | request.extra

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._client is None:
            raise RuntimeError("use RunwareBackend as an async context manager")
        payload = self.build_payload(request)
        result = GenerationResult(request, self.name, utc_now(), 0.0, provider_request=payload)
        if request.model in self._seedless:
            self._drop_seed(payload, result)
        if request.model in self._sizes:
            self._resize(payload, result, self._sizes[request.model])
        start = time.monotonic()
        try:
            # A rejected seed or size is a validation error before any image exists, so
            # resending with the field adapted is not a regeneration; it is recorded.
            for attempt in range(4):
                try:
                    items = await self._client.run(payload)
                    break
                except RunwareError as exc:
                    if attempt == 3 or not self._adapt(exc, request.model, payload, result):
                        raise
            item = dict(items[0])
            if data := item.pop("imageBase64Data", None):
                result.image = base64.b64decode(data)
            elif url := item.get("imageURL"):
                async with httpx.AsyncClient() as http:
                    response = await http.get(url, follow_redirects=True)
                    response.raise_for_status()
                    result.image = response.content
            else:
                result.error = GenerationError("noImage", "response contained no image")
            result.image_format = self._output_format.lower()
            result.cost = item.get("cost")
            result.provider_response = item
        except RunwareError as exc:
            result.error = GenerationError(exc.code, exc.message, exc.retryable)
        except Exception as exc:  # recorded as a failed sample rather than aborting the run
            result.error = GenerationError("unknown", f"{type(exc).__name__}: {exc}")
        result.latency_s = time.monotonic() - start
        return result

    def _adapt(self, exc: RunwareError, model: str, payload: dict[str, Any], result: GenerationResult) -> bool:
        """Adapt the payload to a validation error the model will always raise; False if not adaptable."""
        if exc.code != "validation":
            return False
        if exc.parameter == "seed" and "seed" in payload:
            self._seedless.add(model)
            self._drop_seed(payload, result)
            return True
        # Size errors name "width/height", "width" or "height" depending on the model, so they
        # are recognised by their message: a list of sizes, or a range of pixel counts.
        current = (payload["width"], payload["height"])
        if supported := supported_sizes(exc.message):
            size = closest_size(*current, supported)
        elif limits := pixel_range(exc.message):
            size = scaled_size(*current, *limits)
        else:
            return False
        if size == current:
            return False
        self._sizes[model] = size
        self._resize(payload, result, size)
        return True

    @staticmethod
    def _drop_seed(payload: dict[str, Any], result: GenerationResult) -> None:
        payload.pop("seed", None)
        result.seed_supported = False

    @staticmethod
    def _resize(payload: dict[str, Any], result: GenerationResult, size: tuple[int, int]) -> None:
        requested = [result.request.width, result.request.height]
        payload["width"], payload["height"] = size
        result.adjustments["size"] = {"requested": requested, "used": list(size)}

    async def search_models(self, query: str, *, text_to_image_only: bool = True) -> list[ModelInfo]:
        if self._client is None:
            raise RuntimeError("use RunwareBackend as an async context manager")
        params: dict[str, Any] = {"search": query, "limit": 100}
        if text_to_image_only:
            params |= {"category": "checkpoint", "capabilities": ["io:text-to-image"]}
        found: list[ModelInfo] = []
        while True:
            page = (await self._client.model_search(params | {"offset": len(found)}))[0]
            results = page.get("results", [])
            found += [
                ModelInfo(
                    id=m["air"],
                    name=m.get("name", ""),
                    version=m.get("version"),
                    architecture=m.get("architecture"),
                    source=m.get("source"),
                    details=m,
                )
                for m in results
            ]
            if not results or len(found) >= page.get("totalResults", 0):
                return found
