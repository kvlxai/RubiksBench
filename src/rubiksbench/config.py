"""Benchmark configuration: a benchmark TOML that picks models from a models TOML.

The benchmark file holds what is the same for every model::

    prompt = "A valid scrambled Rubik's Cube"
    iterations = 10                # images per model
    models_file = "models.toml"    # relative to this file (the default)
    library = "library"            # generated images, reused by every run (the default)

    [models]                       # provider -> model names from the models file
    runware = ["FLUX 3 Image"]
    fal = ["MOSS-Image"]

The models file lists, per provider, the models it serves and their generation
settings. A plain string is just the model id; a table can add ``steps``,
``cfg``, ``width``, ``height``, ``seed`` and provider-specific ``extra``
parameters. Anything left out takes the benchmark's value, or the provider's
default::

    [runware]
    "FLUX 3 Image" = "bfl:flux@3-image"

    [fal."MOSS-Image"]
    id = "fal-ai/moss-image"
    steps = 30
    cfg = 4.5

Without ``seed`` every image gets a fresh random seed; with ``seed = N`` image i
uses N + i. Seeds are always recorded.
"""

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .backends.base import GenerationRequest

DEFAULT_PROMPT = "A valid scrambled Rubik's Cube"
MODEL_SETTINGS = {"id", "steps", "cfg", "width", "height", "seed", "extra"}


@dataclass(frozen=True)
class ModelConfig:
    """One model on one provider, with the generation settings that differ from the defaults."""

    backend: str
    id: str
    steps: int | None = None
    cfg: float | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    # Provider-specific parameters sent with every request.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkConfig:
    # Display name -> model and its settings.
    models: dict[str, ModelConfig]
    prompt: str = DEFAULT_PROMPT
    iterations: int = 20
    seed: int | None = None
    # How stickers are read from each image: "vision", "sam" or "annotation".
    extractor: str = "vision"
    # Directory of real cube photos scored alongside every run as a calibration row.
    references: str | None = None
    width: int = 1024
    height: int = 1024
    negative_prompt: str | None = None
    models_file: str = "models.toml"
    # Directory of generated images that runs reuse (see storage/library.py).
    library: str = "library"
    # Models generated at the same time; each model's own requests stay one after another.
    concurrency: int = 8

    @classmethod
    def load(cls, path: Path) -> "BenchmarkConfig":
        with path.open("rb") as f:
            data = tomllib.load(f)
        known = {f.name for f in fields(cls)}
        if unknown := sorted(set(data) - known):
            raise ValueError(f"{path}: unknown setting(s) {unknown}; known: {sorted(known)}")
        selection = data.get("models")
        if not selection or not isinstance(selection, dict):
            raise ValueError(f'{path}: needs a non-empty [models] table of provider = ["model name", ...]')
        from .vision.extractor import EXTRACTORS

        if data.get("extractor", "vision") not in EXTRACTORS:
            raise ValueError(f"{path}: extractor must be one of {sorted(EXTRACTORS)}")
        if data.get("iterations", 1) < 1:
            raise ValueError(f"{path}: iterations must be at least 1")
        if data.get("concurrency", 1) < 1:
            raise ValueError(f"{path}: concurrency must be at least 1")
        catalogue = load_models(path.parent / data.get("models_file", cls.models_file))
        data["models"] = select_models(path, selection, catalogue)
        return cls(**data)

    def request(self, model: ModelConfig, seed: int | None = None) -> GenerationRequest:
        """The request for one image of ``model``: its own settings over the benchmark's."""
        return GenerationRequest(
            model=model.id,
            prompt=self.prompt,
            seed=seed,
            width=model.width or self.width,
            height=model.height or self.height,
            negative_prompt=self.negative_prompt,
            steps=model.steps,
            guidance=model.cfg,
            extra=model.extra,
        )

    def first_seed(self, model: ModelConfig) -> int | None:
        return model.seed if model.seed is not None else self.seed

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_models(path: Path) -> dict[str, dict[str, ModelConfig]]:
    """A models file: provider -> model name -> settings."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    catalogue: dict[str, dict[str, ModelConfig]] = {}
    for backend, models in data.items():
        if not isinstance(models, dict):
            raise ValueError(f"{path}: [{backend}] must be a table of model name = id or settings")
        catalogue[backend] = {}
        for name, entry in models.items():
            entry: Any
            if isinstance(entry, str):
                entry = {"id": entry}
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                raise ValueError(f'{path}: {backend} "{name}" needs an id')
            if unknown := sorted(set(entry) - MODEL_SETTINGS):
                raise ValueError(f'{path}: {backend} "{name}": unknown setting(s) {unknown}; '
                                 f"known: {sorted(MODEL_SETTINGS)} (put provider parameters under extra)")
            catalogue[backend][name] = ModelConfig(backend=backend, **entry)
    return catalogue


def select_models(
    path: Path, selection: dict[str, Any], catalogue: dict[str, dict[str, ModelConfig]]
) -> dict[str, ModelConfig]:
    """The benchmark's models by display name; a name on several providers gets the provider added."""
    picked: list[tuple[str, ModelConfig]] = []
    for backend, names in selection.items():
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError(f'{path}: [models] {backend} must be a list of model names')
        available = catalogue.get(backend, {})
        if missing := [n for n in names if n not in available]:
            raise ValueError(f"{path}: not in the models file under [{backend}]: {missing}")
        picked += [(name, available[name]) for name in names]
    counts: dict[str, int] = {}
    for name, _ in picked:
        counts[name] = counts.get(name, 0) + 1
    return {(name if counts[name] == 1 else f"{name} ({model.backend})"): model for name, model in picked}
