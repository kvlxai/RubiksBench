"""Evaluations of images already scored, so re-scoring an unchanged image is instant.

An image is identified by its content (SHA-256), and its evaluation is only reused
by the same evaluator: the same version, the same scoring code (a fingerprint of
the files that decide an image's result, so an edit without a version bump is
never served stale results) and the same extractor setup (e.g. with or without
CLIP recognition). Extractors without a ``cache_variant`` (hand annotations) are
never cached.

    <library>/evaluations/<evaluator version>/<code fingerprint>-<extractor variant>/
        <sha256[:2]>/<sha256>.json              the evaluation
        <sha256[:2]>/<sha256>.extraction.jpg    its overlay, when the extractor drew one
"""

import functools
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .. import EVALUATOR_VERSION

PACKAGE = Path(__file__).parent.parent
# Files that only aggregate or present results; editing them never changes an image's evaluation.
_PRESENTATION = {"evaluation/report.py", "evaluation/scoring.py"}


@functools.cache
def code_fingerprint() -> str:
    """A hash of the source files that decide one image's evaluation."""
    digest = hashlib.sha256()
    for path in sorted(p for d in ("vision", "cube", "evaluation") for p in (PACKAGE / d).glob("*.py")):
        name = path.relative_to(PACKAGE).as_posix()
        if name not in _PRESENTATION:
            digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:12]


class EvaluationCache:
    def __init__(self, root: Path):
        self.root = root

    def _base(self, image: Path, extractor: Any) -> Path | None:
        variant = getattr(extractor, "cache_variant", None)
        if variant is None:
            return None
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        return self.root / EVALUATOR_VERSION / f"{code_fingerprint()}-{variant}" / digest[:2] / digest

    def get(self, image: Path, extractor: Any) -> dict[str, Any] | None:
        """The stored evaluation of this image, its overlay copied next to it; None if not stored."""
        base = self._base(image, extractor)
        if base is None or not (path := base.with_suffix(".json")).exists():
            return None
        evaluation = json.loads(path.read_text())
        details = evaluation.get("extraction_details", {})
        if details.get("overlay"):
            overlay = image.with_suffix(".extraction.jpg")
            shutil.copyfile(base.with_suffix(".extraction.jpg"), overlay)
            details["overlay"] = overlay.name
        return evaluation

    def put(self, image: Path, extractor: Any, evaluation: dict[str, Any]) -> None:
        base = self._base(image, extractor)
        if base is None:
            return
        base.parent.mkdir(parents=True, exist_ok=True)
        if overlay := evaluation.get("extraction_details", {}).get("overlay"):
            shutil.copyfile(image.parent / overlay, base.with_suffix(".extraction.jpg"))
        base.with_suffix(".json").write_text(json.dumps(evaluation))
