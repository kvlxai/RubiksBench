"""A library of generated images, reused by every run that asks for the same settings.

Images are grouped by everything that shapes a generation except the seed: the
provider, the model id, the prompt and every request setting. A run takes its
images from the matching group and only generates the ones still missing, so a
model is never paid for twice for the same benchmark settings.

    library/<backend>/<model id>/<settings hash>/
        settings.json         the backend and the request, without its seed
        generations.jsonl     one record per image, as in a run's generation log
        images/<n>.<ext>
"""

import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..backends.base import GenerationRequest, GenerationResult


def settings(backend: str, request: GenerationRequest) -> dict[str, Any]:
    """What makes images comparable: the provider and the request, without its seed."""
    return {"backend": backend} | asdict(replace(request, seed=None))


def settings_key(backend: str, request: GenerationRequest) -> str:
    canonical = json.dumps(settings(backend, request), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9@.]+", "-", text.lower()).strip("-")


class ImageLibrary:
    def __init__(self, root: Path):
        self.root = root

    def directory(self, backend: str, request: GenerationRequest) -> Path:
        return self.root / _slug(backend) / _slug(request.model) / settings_key(backend, request)

    def images(self, backend: str, request: GenerationRequest) -> list[dict[str, Any]]:
        """Every stored image for these settings, oldest first; ``image_path`` is relative to the library."""
        path = self.directory(backend, request) / "generations.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def add(self, result: GenerationResult) -> dict[str, Any]:
        """Store a generated image and return its library record."""
        assert result.image is not None
        return self._add(result.backend, result.request, result.metadata(), result.image,
                         result.image_format or "bin")

    def add_record(self, record: dict[str, Any], image: Path) -> tuple[dict[str, Any], bool]:
        """Store an image from a run's generation record: its library record, and whether it is new."""
        request = GenerationRequest(**record["request"])
        digest = record.get("image_sha256") or hashlib.sha256(image.read_bytes()).hexdigest()
        for existing in self.images(record["backend"], request):
            if existing["image_sha256"] == digest:
                return existing, False
        metadata = {k: v for k, v in record.items()
                    if k not in ("sample_id", "model_name", "iteration", "image_path", "image_sha256",
                                 "reused_from", "backfill_of")}
        return self._add(record["backend"], request, metadata, image.read_bytes(), image.suffix.lstrip(".")), True

    def _add(self, backend: str, request: GenerationRequest, metadata: dict[str, Any], image: bytes,
             suffix: str) -> dict[str, Any]:
        directory = self.directory(backend, request)
        (directory / "images").mkdir(parents=True, exist_ok=True)
        if not (directory / "settings.json").exists():
            (directory / "settings.json").write_text(json.dumps(settings(backend, request), indent=2))
        path = directory / "images" / f"{len(self.images(backend, request)):04d}.{suffix}"
        path.write_bytes(image)
        record = metadata | {
            "image_path": str(path.relative_to(self.root)),
            "image_sha256": hashlib.sha256(image).hexdigest(),
        }
        with (directory / "generations.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def entries(self) -> list[tuple[dict[str, Any], int]]:
        """Every settings group with its number of images."""
        found = []
        for path in sorted(self.root.glob("*/*/*/settings.json")):
            log = path.parent / "generations.jsonl"
            found.append((json.loads(path.read_text()), len(log.read_text().splitlines()) if log.exists() else 0))
        return found
