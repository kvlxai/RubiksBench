"""Run a benchmark: take images from the library or generate them serially, evaluate each, score per model."""

import random
import re
import shutil
import sys
import time
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import EVALUATOR_VERSION, __version__
from .backends import ImageBackend, get_backend
from .backends.base import utc_now
from .config import BenchmarkConfig
from .evaluation.evaluator import evaluate_extraction
from .evaluation.report import write_results
from .evaluation.scoring import score_run
from .storage.library import ImageLibrary
from .storage.results import RunStore
from .vision.extractor import Extractor


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _seeds(first: int | None, count: int) -> list[int]:
    if first is not None:
        return [first + i for i in range(count)]
    rng = random.SystemRandom()
    return [rng.randrange(1, 2**31) for _ in range(count)]


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class _Backends:
    """Backends opened on first use, so a run that only reuses library images needs no API keys."""

    def __init__(self, stack: AsyncExitStack):
        self._stack = stack
        self._open: dict[str, ImageBackend] = {}

    async def get(self, name: str) -> ImageBackend:
        if name not in self._open:
            self._open[name] = await self._stack.enter_async_context(get_backend(name))
        return self._open[name]


def _reusable(pool: list[dict[str, Any]], seed: int, fixed_seed: bool) -> dict[str, Any] | None:
    """Take a library image for this sample: the same seed when seeds are fixed, else the oldest unused."""
    for i, record in enumerate(pool):
        if not fixed_seed or record["request"]["seed"] == seed:
            return pool.pop(i)
    return None


async def generate_run(config: BenchmarkConfig, store: RunStore, extractor: Extractor, library: ImageLibrary) -> None:
    """Fill every model's samples from the image library, generating (serially) only what is missing.

    Every new image goes into the library too, for later runs with the same settings.
    """
    store.write_config(
        {
            "benchmark_version": __version__,
            "evaluator_version": EVALUATOR_VERSION,
            "started_at": datetime.now(UTC).isoformat(),
            "config": config.as_dict(),
        }
    )
    async with AsyncExitStack() as stack:
        backends = _Backends(stack)
        for name, model in config.models.items():
            first = config.first_seed(model)
            pool = library.images(model.backend, config.request(model))
            for index, seed in enumerate(_seeds(first, config.iterations)):
                sample_id = f"{_slug(name)}__{index:04d}"
                fields = {"model_name": name, "iteration": index}
                if (found := _reusable(pool, seed, first is not None)) is not None:
                    record = store.add_image(sample_id, found, library.root / found["image_path"], **fields,
                                             reused_from=found["image_path"])
                    status = format_status(evaluate_sample(store, record, extractor))
                    _log(f"[{name}] {index + 1}/{config.iterations} seed={found['request']['seed']} "
                         f"from the library {status}")
                    continue
                backend = await backends.get(model.backend)
                result = await backend.generate(config.request(model, seed))
                if result.ok:
                    record = library.add(result)
                    record = store.add_image(sample_id, record, library.root / record["image_path"], **fields)
                    status = format_status(evaluate_sample(store, record, extractor))
                else:
                    store.add_generation(sample_id, result, **fields)
                    status = f"GENERATION FAILED {result.error.code}: {result.error.message}"
                _log(f"[{name}] {index + 1}/{config.iterations} seed={seed} {result.latency_s:.1f}s {status}")


async def backfill(config: BenchmarkConfig, library: ImageLibrary) -> tuple[int, int]:
    """Generate what the library lacks for a benchmark: ``iterations`` images per model.

    With fixed seeds the missing seeds are generated, else any missing count with
    random seeds. Each missing image is requested once; a failure is logged and
    left for the next backfill. Returns (generated, failed).
    """
    generated = failed = 0
    async with AsyncExitStack() as stack:
        backends = _Backends(stack)
        for name, model in config.models.items():
            first = config.first_seed(model)
            stored = library.images(model.backend, config.request(model))
            if first is not None:
                have = {r["request"]["seed"] for r in stored}
                missing = [s for s in _seeds(first, config.iterations) if s not in have]
            else:
                missing = _seeds(None, max(0, config.iterations - len(stored)))
            if not missing:
                _log(f"[{name}] {len(stored)} images in the library, none missing")
                continue
            backend = await backends.get(model.backend)
            for done, seed in enumerate(missing, 1):
                result = await backend.generate(config.request(model, seed))
                if result.ok:
                    library.add(result)
                    generated += 1
                    status = "stored"
                else:
                    failed += 1
                    status = f"GENERATION FAILED {result.error.code}: {result.error.message}"
                _log(f"[{name}] {done}/{len(missing)} missing, seed={seed} {result.latency_s:.1f}s {status}")
    return generated, failed


def import_run(store: RunStore, library: ImageLibrary) -> int:
    """Copy a run's generated images into the library (once each), so later runs can reuse them."""
    added = 0
    for record in store.generations():
        if record.get("image_path") and not record.get("error"):
            added += library.add_record(record, store.root / record["image_path"])[1]
    return added


def format_status(evaluation: dict[str, Any]) -> str:
    """One image's result for a progress line, e.g. "INVALID (duplicate_centers) · validity 32% · 9/17"."""
    flags = ", ".join(evaluation.get("flags", []))
    status = evaluation["verdict"] + (f" ({flags})" if flags else "")
    if (validity := evaluation.get("validity")) is not None:
        status += f" · validity {validity['score']:.0%}"
    if (rubric := evaluation.get("rubric")) is not None:
        status += f" · {rubric['points']:g}/{rubric['max_points']:g}"
    return status


def _remaining(seconds: float) -> str:
    if seconds < 90:
        return f"~{seconds:.0f}s left"
    return f"~{seconds / 60:.0f} min left"


def evaluate_sample(store: RunStore, record: dict[str, Any], extractor: Extractor) -> dict[str, Any]:
    evaluation = evaluate_extraction(extractor.extract(store.root / record["image_path"]))
    store.add_evaluation(record["sample_id"], evaluation)
    return evaluation


REFERENCE_MODEL = "Reference photos (real cubes)"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def import_references(store: RunStore, directory: Path, extractor: Extractor, evaluate: bool = True) -> int:
    """Copy real cube photos into the run and score them like generated images.

    They are kept apart from the generation log and shown as a calibration row:
    every real cube should score full points.
    """
    target = store.root / "references"
    target.mkdir(exist_ok=True)
    photos = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in _IMAGE_SUFFIXES and ".extraction." not in p.name
    )
    records = []
    for index, photo in enumerate(photos):
        shutil.copyfile(photo, target / photo.name)
        if (annotation := photo.with_suffix(".json")).exists():  # for the annotation extractor
            shutil.copyfile(annotation, target / annotation.name)
        records.append({
            "sample_id": f"reference__{photo.stem}",
            "model_name": REFERENCE_MODEL,
            "iteration": index,
            "reference": True,
            "request": {"model": "real photo", "prompt": None, "seed": None},
            "seed_supported": False,
            "image_path": f"references/{photo.name}",
            "error": None,
            "cost": None,
            "latency_s": 0.0,
            "started_at": utc_now(),
        })
    store.set_references(records)
    if evaluate:
        for record in records:
            evaluate_sample(store, record, extractor)
    return len(records)


def report_run(store: RunStore) -> tuple[dict[str, Any], dict[str, Path]]:
    """Score the evaluations already stored for a run; write timestamped result files."""
    samples, evaluations = store.references() + store.generations(), store.evaluations()
    scores = score_run(samples, evaluations)
    return scores, write_results(store.root, store.read_config(), samples, evaluations, scores)


def evaluate_run(
    store: RunStore, extractor: Extractor, references: Path | None = None
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Re-evaluate every image of a run (e.g. with a newer extractor), then report.

    ``references`` (re)imports a directory of real cube photos; otherwise the
    run's existing reference photos are re-evaluated.
    """
    store.clear_evaluations()
    if references is not None:
        import_references(store, references, extractor, evaluate=False)
    records = [r for r in store.references() + store.generations() if r.get("image_path")]
    _log(f"scoring {len(records)} images ({getattr(extractor, 'name', 'extractor')} extractor; "
         "the first one also loads the models, so it takes longer)")
    durations: list[float] = []
    for done, record in enumerate(records, 1):
        begun = time.monotonic()
        status = format_status(evaluate_sample(store, record, extractor))
        durations.append(time.monotonic() - begun)
        # The first image includes loading the models, so it is left out of the estimate.
        typical = sum(durations[1:]) / (len(durations) - 1) if len(durations) > 1 else None
        left = f" · {_remaining(typical * (len(records) - done))}" if typical is not None and done < len(records) else ""
        _log(f"[{record['model_name']}] #{record['iteration']}  {done}/{len(records)}  {durations[-1]:.1f}s  {status}{left}")
    return report_run(store)


def new_run_dir(out: Path, config_path: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return out / f"{stamp}-{_slug(config_path.stem)}"
