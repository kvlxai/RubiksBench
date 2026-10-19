"""On-disk run layout: one directory per run, images plus JSONL records.

    runs/<run_id>/
        run.json              run configuration
        generations.jsonl     one record per sample (including failures and backfills); images
                              taken from the image library carry ``reused_from``
        evaluations.jsonl     one record per generated image, appended as each is scored
        references.jsonl      real reference photos scored with the run (copied to references/)
        results-<UTC>.json/.html      scores and report, written on every (re-)scoring
        images/<id>.<ext>
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ..backends.base import GenerationResult


class RunStore:
    def __init__(self, root: Path):
        self.root = root
        self.images = root / "images"
        self.images.mkdir(parents=True, exist_ok=True)

    def write_config(self, config: dict[str, Any]) -> None:
        (self.root / "run.json").write_text(json.dumps(config, indent=2))

    def read_config(self) -> dict[str, Any]:
        return json.loads((self.root / "run.json").read_text())

    def add_generation(self, sample_id: str, result: GenerationResult, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {"sample_id": sample_id, **fields, **result.metadata()}
        if result.image is not None:
            path = self.images / f"{sample_id}.{result.image_format or 'bin'}"
            path.write_bytes(result.image)
            record["image_path"] = str(path.relative_to(self.root))
            record["image_sha256"] = hashlib.sha256(result.image).hexdigest()
        with (self.root / "generations.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def add_image(self, sample_id: str, record: dict[str, Any], image: Path, **fields: Any) -> dict[str, Any]:
        """Add a copy of an image already stored elsewhere (the image library)."""
        path = self.images / f"{sample_id}{image.suffix}"
        shutil.copyfile(image, path)
        record = {"sample_id": sample_id, **fields, **record, "image_path": str(path.relative_to(self.root))}
        with (self.root / "generations.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def generation_attempts(self) -> list[dict[str, Any]]:
        """Every generation record ever written, including ones later backfilled."""
        path = self.root / "generations.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def generations(self) -> list[dict[str, Any]]:
        """The current record per sample, in run order (a backfill replaces a failed attempt)."""
        latest: dict[str, dict[str, Any]] = {}
        for record in self.generation_attempts():
            latest[record["sample_id"]] = record
        return list(latest.values())

    def set_references(self, records: list[dict[str, Any]]) -> None:
        (self.root / "references.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    def references(self) -> list[dict[str, Any]]:
        path = self.root / "references.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def add_evaluation(self, sample_id: str, evaluation: dict[str, Any]) -> None:
        with (self.root / "evaluations.jsonl").open("a") as f:
            f.write(json.dumps({"sample_id": sample_id} | evaluation) + "\n")

    def evaluations(self) -> dict[str, dict[str, Any]]:
        """Latest evaluation per sample."""
        path = self.root / "evaluations.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return {r.pop("sample_id"): r for r in records}

    def clear_evaluations(self) -> None:
        (self.root / "evaluations.jsonl").unlink(missing_ok=True)
