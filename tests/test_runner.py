import asyncio
import json
from pathlib import Path

import pytest

from rubiksbench.backends.base import (
    GenerationError,
    GenerationRequest,
    GenerationResult,
    ImageBackend,
    register_backend,
    utc_now,
)
from rubiksbench.backends import available_backends
from rubiksbench.config import DEFAULT_PROMPT, BenchmarkConfig
from rubiksbench.runner import REFERENCE_MODEL, backfill, evaluate_run, generate_run, import_run, report_run
from rubiksbench.storage.library import ImageLibrary
from rubiksbench.storage.results import RunStore
from rubiksbench.vision.extractor import AnnotationExtractor

from .test_validator import REAL_CUBE

REPO = Path(__file__).parent.parent


@register_backend
class FakeBackend(ImageBackend):
    name = "fake"
    requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        result = GenerationResult(request, self.name, utc_now(), 0.1, cost=0.01)
        if request.model == "broken":
            result.error = GenerationError("provider", "model unavailable")
        else:
            result.image, result.image_format = f"img {request.seed}".encode(), "png"
        return result


MODELS = '[fake]\n"Good" = "good"\n"Broken" = "broken"\n'


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def bench(tmp_path: Path, settings: str, models: str = '["Good"]', catalogue: str = MODELS) -> BenchmarkConfig:
    """A benchmark picking ``models`` from the fake provider of a models file next to it."""
    write(tmp_path / "models.toml", catalogue)
    return BenchmarkConfig.load(write(tmp_path / "b.toml", f"{settings}\n[models]\nfake = {models}\n"))


def test_shipped_benchmark_loads():
    config = BenchmarkConfig.load(REPO / "configs" / "benchmark.toml")
    assert config.iterations >= 1 and config.models
    assert all(m.backend in available_backends() for m in config.models.values())


def test_defaults(tmp_path):
    config = bench(tmp_path, "")
    assert config.iterations == 20 and config.prompt == DEFAULT_PROMPT and config.seed is None
    assert config.extractor == "vision"


def test_extractor_setting(tmp_path):
    assert bench(tmp_path, 'extractor = "sam"').extractor == "sam"
    with pytest.raises(ValueError, match="extractor"):
        bench(tmp_path, 'extractor = "yolo"')


def test_config_rejects_unknown_settings(tmp_path):
    with pytest.raises(ValueError, match="iteration"):
        bench(tmp_path, "iteration = 3")
    with pytest.raises(ValueError, match="models"):
        BenchmarkConfig.load(write(tmp_path / "b.toml", "iterations = 3\n"))
    with pytest.raises(ValueError, match="Missing"):
        bench(tmp_path, "", models='["Missing"]')
    with pytest.raises(ValueError, match="cgf"):
        bench(tmp_path, "", catalogue='[fake.Good]\nid = "good"\ncgf = 3\n')


def test_model_settings_override_the_benchmark(tmp_path):
    catalogue = MODELS + '[fake.Tuned]\nid = "tuned"\nsteps = 30\ncfg = 4.5\nwidth = 512\nseed = 7\nextra = { sampler = "euler" }\n'
    config = bench(tmp_path, 'prompt = "a cube"\nheight = 768', models='["Good", "Tuned"]', catalogue=catalogue)
    good, tuned = config.models["Good"], config.models["Tuned"]
    assert config.request(good, 1) == GenerationRequest("good", "a cube", 1, width=1024, height=768)
    assert config.request(tuned, 1) == GenerationRequest(
        "tuned", "a cube", 1, width=512, height=768, steps=30, guidance=4.5, extra={"sampler": "euler"})
    assert config.first_seed(good) is None and config.first_seed(tuned) == 7


def test_a_model_on_two_providers_is_named_after_each(tmp_path):
    write(tmp_path / "models.toml", '[fake]\n"Good" = "good"\n[other]\n"Good" = "other-good"\n')
    config = BenchmarkConfig.load(write(tmp_path / "b.toml", '[models]\nfake = ["Good"]\nother = ["Good"]\n'))
    assert {n: m.backend for n, m in config.models.items()} == {"Good (fake)": "fake", "Good (other)": "other"}


def test_run_generates_serially_and_scores(tmp_path):
    config = bench(tmp_path, "iterations = 2", models='["Good", "Broken"]')
    store = RunStore(tmp_path / "run")
    FakeBackend.requests.clear()
    # The first image is annotated before the run starts, so it is scored as soon as it is generated.
    write(store.images / "good__0000.json", json.dumps(REAL_CUBE))
    asyncio.run(generate_run(config, store, AnnotationExtractor(), ImageLibrary(tmp_path / "library")))
    evaluations = store.evaluations()
    assert evaluations["good__0000"]["verdict"] == "VALID_STANDARD"
    assert evaluations["good__0001"]["verdict"] == "INCONCLUSIVE"
    assert set(evaluations) == {"good__0000", "good__0001"}

    assert [r.model for r in FakeBackend.requests] == ["good", "good", "broken", "broken"]
    seeds = [r.seed for r in FakeBackend.requests]
    assert len(set(seeds)) == 4 and all(seeds)

    scores, paths = report_run(store)
    assert evaluate_run(store, AnnotationExtractor())[0] == scores  # re-scoring gives the same result
    good, broken = scores["Good"], scores["Broken"]
    assert good["verdicts"]["VALID_STANDARD"] == 1 and good["verdicts"]["INCONCLUSIVE"] == 1
    assert good["standard_valid_rate"] == 0.5 and good["total_cost"] == pytest.approx(0.02)
    assert broken["generation_failures"] == 2 and broken["images"] == 0
    assert broken["standard_valid_rate"] is None

    generations = store.generations()
    assert all("seed" in g["request"] for g in generations)
    assert json.loads(paths["json"].read_text())["scores"] == scores
    # One valid image (17 points) and one unannotated, so inconclusive at the first check (0.5).
    assert good["rubric_mean"] == pytest.approx(8.75)
    # Validity: the valid image is 100%, the unreadable one 0%.
    assert good["validity_mean"] == pytest.approx(0.5)
    assert (good["passed"], good["failed"], good["generation_failures"]) == (1, 1, 0)
    assert (broken["passed"], broken["failed"], broken["generation_failures"]) == (0, 0, 2)
    report = paths["html"].read_text()
    assert report.startswith("<!doctype html>") and "<b>50%</b>" in report
    assert "images/good__0000.png" in report and "unseeded" not in report


def test_backfill_fills_the_library_up_to_the_benchmark(tmp_path):
    library = ImageLibrary(tmp_path / "library")
    asyncio.run(generate_run(bench(tmp_path, "iterations = 1"), RunStore(tmp_path / "run"), AnnotationExtractor(), library))

    config = bench(tmp_path, "iterations = 3", models='["Good", "Broken"]')
    FakeBackend.requests.clear()
    assert asyncio.run(backfill(config, library)) == (2, 3)  # one failure per missing image, no retries
    assert [r.model for r in FakeBackend.requests] == ["good", "good", "broken", "broken", "broken"]
    assert len(library.images("fake", config.request(config.models["Good"]))) == 3

    FakeBackend.requests.clear()
    asyncio.run(generate_run(bench(tmp_path, "iterations = 3"), RunStore(tmp_path / "next"), AnnotationExtractor(), library))
    assert FakeBackend.requests == []  # the run finds everything in the library

    # With fixed seeds, the seeds that are missing are generated.
    FakeBackend.requests.clear()
    asyncio.run(backfill(bench(tmp_path, "iterations = 2\nseed = 5"), library))
    assert [r.seed for r in FakeBackend.requests] == [5, 6]


def test_runs_reuse_library_images_and_only_generate_new_models(tmp_path):
    library = ImageLibrary(tmp_path / "library")
    config = bench(tmp_path, "iterations = 2")
    FakeBackend.requests.clear()
    asyncio.run(generate_run(config, RunStore(tmp_path / "run1"), AnnotationExtractor(), library))
    assert len(FakeBackend.requests) == 2

    # A later benchmark adds a model and asks for one more image of the first.
    catalogue = MODELS + '"New" = "new"\n'
    config = bench(tmp_path, "iterations = 3", models='["Good", "New"]', catalogue=catalogue)
    FakeBackend.requests.clear()
    store = RunStore(tmp_path / "run2")
    asyncio.run(generate_run(config, store, AnnotationExtractor(), library))
    assert [r.model for r in FakeBackend.requests] == ["good", "new", "new", "new"]
    good = [g for g in store.generations() if g["model_name"] == "Good"]
    assert [bool(g.get("reused_from")) for g in good] == [True, True, False]
    assert all((store.root / g["image_path"]).exists() for g in store.generations())
    assert len(library.images("fake", config.request(config.models["Good"]))) == 3

    # A different prompt is a different set of images.
    config = bench(tmp_path, 'iterations = 1\nprompt = "another cube"')
    FakeBackend.requests.clear()
    asyncio.run(generate_run(config, RunStore(tmp_path / "run3"), AnnotationExtractor(), library))
    assert len(FakeBackend.requests) == 1


def test_fixed_seeds_reuse_only_the_same_seeds(tmp_path):
    library = ImageLibrary(tmp_path / "library")
    asyncio.run(generate_run(bench(tmp_path, "iterations = 2\nseed = 10"), RunStore(tmp_path / "a"),
                             AnnotationExtractor(), library))
    FakeBackend.requests.clear()
    asyncio.run(generate_run(bench(tmp_path, "iterations = 3\nseed = 11"), RunStore(tmp_path / "b"),
                             AnnotationExtractor(), library))
    assert [r.seed for r in FakeBackend.requests] == [12, 13]  # 11 is in the library, 10 is not asked for


def test_earlier_runs_can_be_imported_into_the_library(tmp_path):
    config = bench(tmp_path, "iterations = 2", models='["Good", "Broken"]')
    store = RunStore(tmp_path / "run")
    asyncio.run(generate_run(config, store, AnnotationExtractor(), ImageLibrary(tmp_path / "unused")))
    library = ImageLibrary(tmp_path / "library")
    assert import_run(store, library) == 2 and import_run(store, library) == 0  # failures and repeats are skipped
    FakeBackend.requests.clear()
    asyncio.run(generate_run(config, RunStore(tmp_path / "again"), AnnotationExtractor(), library))
    assert [r.model for r in FakeBackend.requests] == ["broken", "broken"]


def test_reference_photos_are_scored_as_a_separate_row(tmp_path):
    config = bench(tmp_path, "iterations = 1")
    store = RunStore(tmp_path / "run")
    asyncio.run(generate_run(config, store, AnnotationExtractor(), ImageLibrary(tmp_path / "library")))
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "reference_1.png").write_bytes(b"img")
    write(refs / "reference_1.json", json.dumps(REAL_CUBE))  # its hand annotation

    scores, paths = evaluate_run(store, AnnotationExtractor(), references=refs)
    ref = scores[REFERENCE_MODEL]
    assert ref["reference"] and ref["images"] == 1 and ref["passed"] == 1
    assert (store.root / "references" / "reference_1.png").exists()
    assert len(store.generations()) == 1  # the generation log is untouched
    html = paths["html"].read_text()
    assert "references/reference_1.png" in html and "real photo" in html


def test_site_has_sharing_tags_and_a_framed_preview(tmp_path):
    import cv2
    import numpy as np

    from rubiksbench.site import SHARE_SIZE, build_site

    config = bench(tmp_path, "iterations = 1")
    store = RunStore(tmp_path / "run")
    write(store.images / "good__0000.json", json.dumps(REAL_CUBE))
    asyncio.run(generate_run(config, store, AnnotationExtractor(), ImageLibrary(tmp_path / "library")))
    # A real picture: a colored square (the "cube") low on a gray background.
    picture = np.full((200, 200, 3), 127, np.uint8)
    picture[110:180, 70:140] = (40, 40, 220)
    cv2.imwrite(str(store.images / "good__0000.png"), picture)
    report_run(store)

    index = build_site(store.root, tmp_path / "site", url="https://example.org/bench/", title="RubiksBench",
                       description="How good are models at cubes?", share_sample="good__0000", share_alt="A cube")
    html = index.read_text()
    assert "<title>RubiksBench</title>" in html
    assert "<meta property='og:image' content='https://example.org/bench/share.jpg'>" in html
    assert "<meta name='twitter:card' content='summary_large_image'>" in html
    assert "images/good__0000.jpg" in html and (index.parent / "images" / "good__0000.jpg").exists()
    share = cv2.imread(str(index.parent / "share.jpg"))
    assert share.shape[1::-1] == SHARE_SIZE
    # The square is framed in the middle, not cut off at the bottom.
    red = np.argwhere(share[:, :, 2] > 180)
    assert 0.25 < red[:, 0].mean() / SHARE_SIZE[1] < 0.75


def test_valid_images_show_a_solution_with_a_3d_view():
    from urllib.parse import parse_qs, urlparse

    from rubiksbench.cube.observation import Observation
    from rubiksbench.cube.witness import apply_moves, invert
    from rubiksbench.evaluation.evaluator import validate
    from rubiksbench.evaluation.report import _solution

    record = validate(Observation.from_faces(REAL_CUBE)).to_dict()
    solution = _solution(record)
    witness = record["standard"]["witness"]
    # The moves undo the scramble that builds the completed cube.
    assert apply_moves(witness["scramble"] + " " + solution["moves"]) == apply_moves("")
    assert solution["count"] == len(solution["moves"].split()) and solution["moves"] == invert(witness["scramble"])
    query = parse_qs(urlparse(solution["viewer"]).query)
    assert query["alg"] == [solution["moves"]] and query["setup-alg"][0].endswith(witness["scramble"])
    assert _solution({"verdict": "INVALID"}) is None


def test_versions_share_major_minor():
    # A scoring change bumps the minor version; patch releases never change a score.
    from rubiksbench import EVALUATOR_VERSION, __version__

    assert EVALUATOR_VERSION.split(".")[:2] == __version__.split(".")[:2]


def test_rescoring_reports_progress_per_image(tmp_path, capsys):
    config = bench(tmp_path, "iterations = 2")
    store = RunStore(tmp_path / "run")
    write(store.images / "good__0000.json", json.dumps(REAL_CUBE))
    asyncio.run(generate_run(config, store, AnnotationExtractor(), ImageLibrary(tmp_path / "library")))
    capsys.readouterr()

    evaluate_run(store, AnnotationExtractor())
    lines = capsys.readouterr().err.splitlines()
    assert lines[0].startswith("scoring 2 images (annotation extractor;")
    assert lines[1].startswith("[Good] #0  1/2  ") and "VALID_STANDARD · validity 100% · 17/17" in lines[1]
    # No estimate from the first image alone (it includes loading the models), none after the last.
    assert "left" not in lines[1] and lines[2].startswith("[Good] #1  2/2  ") and "left" not in lines[2]


def test_gallery_shows_each_models_best_image_and_halls_leave_out_photos(tmp_path):
    import re

    config = bench(tmp_path, "iterations = 3", models='["Good", "New"]', catalogue=MODELS + '"New" = "new"\n')
    store = RunStore(tmp_path / "run")
    write(store.images / "good__0001.json", json.dumps(REAL_CUBE))  # the second image is the valid one
    asyncio.run(generate_run(config, store, AnnotationExtractor(), ImageLibrary(tmp_path / "library")))
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "reference_1.png").write_bytes(b"img")
    write(refs / "reference_1.json", json.dumps(REAL_CUBE))
    html = evaluate_run(store, AnnotationExtractor(), references=refs)[1]["html"].read_text()

    gallery = html.split("id='gallery'>", 1)[1]
    shown = re.findall(r"<div class='sample' data-model='([^']+)'[^>]*?(?<! hidden)>", gallery)
    assert sorted(shown) == ["Good", "New", "Reference photos (real cubes)"]  # the best of each model
    assert "src='images/good__0001.png'" in gallery  # the best Good image loads at once
    assert "data-src='images/good__0000.png'" in gallery  # the rest wait until shown
    assert len(re.findall(r"<div class='sample'[^>]* hidden>", gallery)) == 4
    halls = html.split("<h2>Hall of Fame</h2>", 1)[1].split("<h2>Samples</h2>", 1)[0]
    assert "images/good__0001.png" in halls and "references/" not in halls


def test_models_generate_side_by_side_but_each_model_one_at_a_time(tmp_path):
    @register_backend
    class SlowBackend(ImageBackend):
        name = "slow"
        running: dict[str, int] = {}
        peak = {"total": 0, "per_model": 0}

        async def generate(self, request: GenerationRequest) -> GenerationResult:
            self.running[request.model] = self.running.get(request.model, 0) + 1
            self.peak["total"] = max(self.peak["total"], sum(self.running.values()))
            self.peak["per_model"] = max(self.peak["per_model"], self.running[request.model])
            await asyncio.sleep(0.01)
            self.running[request.model] -= 1
            result = GenerationResult(request, self.name, utc_now(), 0.01)
            result.image, result.image_format = f"img {request.model} {request.seed}".encode(), "png"
            return result

    write(tmp_path / "models.toml", "[slow]\n" + "".join(f'"M{i}" = "m{i}"\n' for i in range(5)))
    config = BenchmarkConfig.load(write(tmp_path / "b.toml", "iterations = 3\nconcurrency = 3\n[models]\n"
                                        f"slow = {json.dumps([f'M{i}' for i in range(5)])}\n"))
    library = ImageLibrary(tmp_path / "library")
    assert asyncio.run(backfill(config, library)) == (15, 0)
    assert SlowBackend.peak == {"total": 3, "per_model": 1}

    store = RunStore(tmp_path / "run")
    asyncio.run(generate_run(config, store, AnnotationExtractor(), library))
    assert len(store.evaluations()) == 15  # everything reused from the library, then scored


def test_scored_images_are_cached_per_evaluator(tmp_path, monkeypatch, capsys):
    from rubiksbench.storage import evaluation_cache
    from rubiksbench.storage.evaluation_cache import EvaluationCache

    class CountingExtractor(AnnotationExtractor):
        """Hand annotations are never cached, so this one pretends to be a cacheable reader."""
        cache_variant = "counting"
        calls = 0

        def extract(self, path):
            CountingExtractor.calls += 1
            return super().extract(path)

    config = bench(tmp_path, "iterations = 2")
    store = RunStore(tmp_path / "run")
    write(store.images / "good__0000.json", json.dumps(REAL_CUBE))
    cache = EvaluationCache(tmp_path / "library" / "evaluations")
    asyncio.run(generate_run(config, store, CountingExtractor(), ImageLibrary(tmp_path / "library"), cache))
    first = store.evaluations()
    assert CountingExtractor.calls == 2

    capsys.readouterr()
    scores, _ = evaluate_run(store, CountingExtractor(), cache=cache)
    assert CountingExtractor.calls == 2 and store.evaluations() == first  # nothing read again, same results
    assert "2 of 2 images already scored by this evaluator" in capsys.readouterr().err

    monkeypatch.setattr(evaluation_cache, "code_fingerprint", lambda: "changed-code")
    evaluate_run(store, CountingExtractor(), cache=cache)
    assert CountingExtractor.calls == 4  # different scoring code: scored again

    evaluate_run(store, AnnotationExtractor(), cache=cache)  # no cache_variant: never cached
    assert not list((tmp_path / "library" / "evaluations").rglob("*annotation*"))
