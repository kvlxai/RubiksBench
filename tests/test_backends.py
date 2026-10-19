import asyncio
import base64
import json

import pytest
from runware import create_runware_error

from rubiksbench.backends import GenerationRequest, get_backend
from rubiksbench.backends.fal import map_arguments
from rubiksbench.backends.runware import closest_size, supported_sizes
from rubiksbench.storage.results import RunStore


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.payloads = response, error, []

    async def run(self, payload):
        self.payloads.append(payload)
        if self.error:
            raise self.error
        return self.response


def generate(client, request):
    backend = get_backend("runware", api_key="test")
    backend._client = client
    return asyncio.run(backend.generate(request))


def test_runware_success_records_payload_image_and_cost():
    client = FakeClient([{"imageBase64Data": base64.b64encode(b"PNGDATA").decode(), "cost": 0.004, "seed": 7}])
    request = GenerationRequest(model="runware:400@1", prompt="a cube", seed=7, steps=30)
    result = generate(client, request)
    assert result.ok and result.image == b"PNGDATA" and result.cost == 0.004
    sent = client.payloads[0]
    assert sent["seed"] == 7 and sent["steps"] == 30 and sent["positivePrompt"] == "a cube"
    assert "CFGScale" not in sent and "negativePrompt" not in sent
    assert "imageBase64Data" not in result.provider_response


class SeedRejectingClient(FakeClient):
    async def run(self, payload):
        self.payloads.append(dict(payload))
        if "seed" in payload:
            raise create_runware_error("invalidParameter", "seed not supported", parameter="seed")
        return self.response


def test_runware_drops_seed_for_models_that_reject_it():
    client = SeedRejectingClient([{"imageBase64Data": base64.b64encode(b"X").decode()}])
    backend = get_backend("runware", api_key="test")
    backend._client = client
    request = GenerationRequest(model="openai:gpt-image@2", prompt="p", seed=3)
    first = asyncio.run(backend.generate(request))
    second = asyncio.run(backend.generate(request))
    assert first.ok and not first.seed_supported and "seed" not in first.provider_request
    assert second.ok and not second.seed_supported
    # One rejected attempt, one resend, then the seed is left out up front.
    assert ["seed" in p for p in client.payloads] == [True, False, False]


def test_runware_failure_is_recorded_not_raised(tmp_path):
    client = FakeClient(error=create_runware_error("invalidParameter", "bad width", parameter="width"))
    result = generate(client, GenerationRequest(model="m", prompt="p", seed=1))
    assert not result.ok and result.error.message == "bad width"

    store = RunStore(tmp_path)
    store.add_generation("m__0000", result)
    record = json.loads((tmp_path / "generations.jsonl").read_text())
    assert record["error"]["message"] == "bad width" and "image_path" not in record


class FakeHfClient:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    async def text_to_image(self, prompt, **kwargs):
        from PIL import Image

        self.calls.append((prompt, kwargs))
        if self.error:
            raise self.error
        return Image.new("RGB", (8, 8), (255, 0, 0))


def test_huggingface_generates_png_and_omits_unset_parameters():
    backend = get_backend("huggingface", token="test")
    backend._client = client = FakeHfClient()
    request = GenerationRequest(model="black-forest-labs/FLUX.1-dev", prompt="a cube", seed=5)
    result = asyncio.run(backend.generate(request))
    assert result.ok and result.image.startswith(b"\x89PNG") and result.image_format == "png"
    prompt, kwargs = client.calls[0]
    assert prompt == "a cube" and kwargs["seed"] == 5 and kwargs["model"] == "black-forest-labs/FLUX.1-dev"
    assert "guidance_scale" not in kwargs and "negative_prompt" not in kwargs


def test_huggingface_failure_is_recorded_not_raised():
    backend = get_backend("huggingface", token="test")
    backend._client = FakeHfClient(error=RuntimeError("provider down"))
    result = asyncio.run(backend.generate(GenerationRequest(model="m", prompt="p", seed=1)))
    assert not result.ok and result.error.code == "unknown" and "provider down" in result.error.message


FLUX3_LIKE = {  # aspect ratio, no seed
    "prompt": {"type": "string"},
    "aspect_ratio": {"type": "string", "enum": ["16:9", "1:1"]},
    "output_format": {"type": "string", "enum": ["jpeg", "png"]},
}
GPT_LIKE = {"prompt": {"type": "string"}, "image_size": {"type": "string", "enum": ["auto", "1024x1024"]}}
FLUX_LIKE = {
    "prompt": {"type": "string"},
    "seed": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    "image_size": {"anyOf": [{"$ref": "#/components/schemas/ImageSize"}, {"enum": ["square_hd"]}]},
    "num_inference_steps": {"type": "integer"},
    "num_images": {"type": "integer"},
}


@pytest.mark.parametrize(
    "properties, sent, unsupported",
    [
        (FLUX3_LIKE, {"aspect_ratio": "1:1", "output_format": "png"}, ["seed", "steps"]),
        (GPT_LIKE, {"image_size": "1024x1024"}, ["seed", "steps"]),
        (FLUX_LIKE, {"seed": 7, "image_size": {"width": 1024, "height": 1024}, "num_inference_steps": 20,
                     "num_images": 1}, []),
    ],
)
def test_fal_sends_only_what_the_endpoint_accepts(properties, sent, unsupported):
    request = GenerationRequest(model="m", prompt="p", seed=7, steps=20)
    arguments, dropped = map_arguments(request, properties)
    assert arguments == {"prompt": "p"} | sent and dropped == unsupported


class FakeFalClient:
    def __init__(self):
        self.calls = []

    async def subscribe(self, application, arguments):
        self.calls.append((application, arguments))
        data = base64.b64encode(b"PNGDATA").decode()
        return {"images": [{"url": f"data:image/png;base64,{data}", "content_type": "image/png"}], "seed": 7}


def test_fal_generation_records_unsupported_seed():
    backend = get_backend("fal", key="test")
    backend._client = client = FakeFalClient()
    backend._schemas["blackforestlabs/flux-3/text-to-image"] = FLUX3_LIKE
    request = GenerationRequest(model="blackforestlabs/flux-3/text-to-image", prompt="a cube", seed=7)
    result = asyncio.run(backend.generate(request))
    assert result.ok and result.image == b"PNGDATA" and result.image_format == "png"
    assert not result.seed_supported and result.provider_request["unsupported"] == ["seed"]
    assert "seed" not in client.calls[0][1]


def test_fal_unknown_endpoint_is_a_recorded_failure():
    backend = get_backend("fal", key="test")
    backend._client = FakeFalClient()
    backend._schemas["nope"] = None
    result = asyncio.run(backend.generate(GenerationRequest(model="nope", prompt="p", seed=1)))
    assert not result.ok and result.error.code == "unknownModel"


def test_fal_schema_is_fetched_without_the_key_when_the_key_gets_an_error():
    import httpx

    openapi = {"paths": {"/fal-ai/x": {"post": {"requestBody": {"content": {"application/json": {
        "schema": {"$ref": "#/components/schemas/Input"}}}}}}},
        "components": {"schemas": {"Input": {"properties": {"prompt": {}, "seed": {}}}}}}
    seen = []

    def respond(request):
        keyed = "authorization" in request.headers
        seen.append(keyed)
        error = {"error": {"code": "expansion_failed", "message": "Failed to fetch OpenAPI schema"}}
        return httpx.Response(200, json={"models": [{"openapi": error if keyed else openapi}]})

    async def properties():
        backend = get_backend("fal", key="test")
        backend._http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        return await backend._properties("fal-ai/x")

    assert set(asyncio.run(properties())) == {"prompt", "seed"} and seen == [True, False]


IDEOGRAM_SIZES = (
    "Unsupported use of width/height parameters. Supported values are: '2048x2048', '1440x2880', "
    "'2880x1440', '1664x2496', '2496x1664'."
)


class SizeRejectingClient(FakeClient):
    async def run(self, payload):
        self.payloads.append(dict(payload))
        if (payload["width"], payload["height"]) != (2048, 2048):
            raise create_runware_error("invalidParameter", IDEOGRAM_SIZES, parameter="width/height")
        return self.response


def test_closest_supported_size_keeps_the_shape():
    sizes = supported_sizes(IDEOGRAM_SIZES)
    assert (2048, 2048) in sizes and len(sizes) == 5
    assert closest_size(1024, 1024, sizes) == (2048, 2048)
    assert closest_size(1024, 1536, sizes) == (1664, 2496)


def test_runware_resizes_for_models_with_fixed_sizes():
    client = SizeRejectingClient([{"imageBase64Data": base64.b64encode(b"X").decode()}])
    backend = get_backend("runware", api_key="test")
    backend._client = client
    request = GenerationRequest(model="ideogram:4@0", prompt="p", seed=3)
    first = asyncio.run(backend.generate(request))
    second = asyncio.run(backend.generate(request))
    for result in (first, second):
        assert result.ok and result.adjustments["size"] == {"requested": [1024, 1024], "used": [2048, 2048]}
    # One rejected attempt and a resend, then the supported size is used up front.
    assert [(p["width"], p["height"]) for p in client.payloads] == [(1024, 1024), (2048, 2048), (2048, 2048)]
