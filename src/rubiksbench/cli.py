"""Command line entry point.

    rubiksbench validate observation.json
    rubiksbench validate-image cube.png
    rubiksbench models "Flux 3"
    rubiksbench backfill configs/benchmark.toml
    rubiksbench run configs/benchmark.toml
    rubiksbench library import runs/<run>
    rubiksbench score runs/<run>
    rubiksbench site runs/<run> site/ --url https://example.github.io/RubiksBench/ --share-image <sample>
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .backends import available_backends, get_backend
from .config import BenchmarkConfig
from .cube.observation import Observation
from .evaluation.evaluator import evaluate_extraction, validate
from .evaluation.rubric import CHECKS, FAIL
from .evaluation.scoring import format_table
from .site import build_site
from .runner import (
    backfill, evaluate_run, format_status, generate_run, import_references, import_run, new_run_dir, report_run,
)
from .storage.library import ImageLibrary
from .storage.results import RunStore
from .vision.extractor import EXTRACTORS


def cmd_site(args: argparse.Namespace) -> int:
    index = build_site(Path(args.run_dir), Path(args.out_dir), url=args.url, title=args.title,
                       description=args.description, share_sample=args.share_image, share_alt=args.share_alt)
    print(f"site written to {index.parent}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a JSON observation: {"U": "RWR BGB RYY", "F": ..., "R": ...}."""
    faces = json.loads(Path(args.observation).read_text())
    report = validate(Observation.from_faces(faces), with_scramble=not args.no_scramble)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


async def _models(args: argparse.Namespace) -> int:
    async with get_backend(args.backend) as backend:
        models = await backend.search_models(args.query, text_to_image_only=not args.all)
    if args.json:
        print(json.dumps([m.details | {"id": m.id} for m in models], indent=2))
        return 0
    if not models:
        print(f"no models match {args.query!r}", file=sys.stderr)
        return 1
    width = max(len(m.id) for m in models)
    for m in models:
        extra = ", ".join(x for x in (m.architecture, m.source) if x)
        name = f"  {m.name}" if m.name != m.id else ""
        print(f"{m.id:<{width}}{name}" + (f"  ({extra})" if extra else ""))
    return 0


def cmd_validate_image(args: argparse.Namespace) -> int:
    """Read and score one image using the benchmark's extraction pipeline."""
    image = Path(args.image)
    if not image.is_file():
        print(f"image file does not exist or is not a regular file: {image}", file=sys.stderr)
        return 1
    extraction = EXTRACTORS[args.extractor]().extract(image)
    report = evaluate_extraction(extraction, with_scramble=not args.no_scramble)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_status(report))
        if args.verbose:
            failed = [check for check in CHECKS if report["rubric"]["checks"].get(check.id) == FAIL]
            print("Failed rubric checks:" if failed else "No failed rubric checks.")
            for check in failed:
                print(f"  - {check.title}: {check.description}")
    return 0


def _run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = BenchmarkConfig.load(config_path)
    store = RunStore(Path(args.run_dir) if args.run_dir else new_run_dir(Path(args.out), config_path))
    print(f"run directory: {store.root}", file=sys.stderr)
    extractor = EXTRACTORS[args.extractor or config.extractor]()
    try:
        asyncio.run(generate_run(config, store, extractor, ImageLibrary(Path(config.library))))
    except KeyboardInterrupt:
        print("\ninterrupted: reporting the images generated so far", file=sys.stderr)
    if config.references:
        import_references(store, Path(config.references), extractor)
    return _report(*report_run(store))


def _backfill(args: argparse.Namespace) -> int:
    config = BenchmarkConfig.load(Path(args.config))
    generated, failed = asyncio.run(backfill(config, ImageLibrary(Path(config.library))))
    print(f"generated {generated} image(s) into {Path(config.library)}"
          + (f"; {failed} failed, backfill again to retry them" if failed else ""), file=sys.stderr)
    return 1 if failed else 0


def _score(store: RunStore, extractor: str | None, references: str | None = None) -> int:
    # Default to the extractor (and reference photos) the run was configured with.
    config = store.read_config().get("config", {})
    name = extractor or config.get("extractor", "vision")
    directory = references or config.get("references")
    return _report(*evaluate_run(store, EXTRACTORS[name](), Path(directory) if directory else None))


def _library_import(args: argparse.Namespace) -> int:
    library = ImageLibrary(Path(args.library))
    for run_dir in args.run_dirs:
        print(f"{run_dir}: {import_run(RunStore(Path(run_dir)), library)} new image(s)")
    return 0


def _library_list(args: argparse.Namespace) -> int:
    for entry, count in ImageLibrary(Path(args.library)).entries():
        others = {k: v for k, v in entry.items()
                  if k not in ("backend", "model", "prompt", "width", "height") and v not in (None, {})}
        settings = " ".join([f"{entry['width']}x{entry['height']}", *(f"{k}={v}" for k, v in others.items())])
        print(f"{count:4d}  {entry['backend']}  {entry['model']}  {settings}  “{entry['prompt']}”")
    return 0


def _report(scores: dict[str, Any], paths: dict[str, Path]) -> int:
    print(format_table(scores))
    print("\nresults written to:", *(f"  {p}" for p in paths.values()), sep="\n", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rubiksbench")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="validate a manually specified partial cube observation")
    p.add_argument("observation", help="JSON file mapping faces (U R F D L B) to 9-sticker grids")
    p.add_argument("--no-scramble", action="store_true", help="skip computing a witness scramble")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("validate-image", help="extract and score a cube directly from an image")
    p.add_argument("image", help="path to a cube image")
    p.add_argument("--extractor", default="vision", choices=sorted(EXTRACTORS),
                   help="sticker extractor (default: vision)")
    p.add_argument("--json", action="store_true", help="print the full report as JSON (overrides --verbose)")
    p.add_argument("-v", "--verbose", action="store_true", help="also describe failed rubric checks in text output")
    p.add_argument("--no-scramble", action="store_true", help="skip computing a witness scramble")
    p.set_defaults(func=cmd_validate_image)

    p = sub.add_parser("models", help="search a backend's models and print their ids")
    p.add_argument("query", help='name to search for, e.g. "Flux 3"')
    p.add_argument("--backend", default="runware", choices=available_backends())
    p.add_argument("--all", action="store_true", help="include LoRAs, video and non-text-to-image models")
    p.add_argument("--json", action="store_true", help="print full model records as JSON")
    p.set_defaults(func=lambda a: asyncio.run(_models(a)))

    p = sub.add_parser("run", help="score every model in a benchmark TOML, generating the images the library lacks")
    p.add_argument("config", help="benchmark TOML file, e.g. configs/benchmark.toml")
    p.add_argument("--out", default="runs", help="parent directory for run directories")
    p.add_argument("--run-dir", help="explicit run directory (default: runs/<timestamp>-<config name>)")
    p.add_argument("--extractor", choices=sorted(EXTRACTORS), help="override the TOML's extractor")
    p.set_defaults(func=_run)

    p = sub.add_parser("backfill", help="generate the images a benchmark's models lack in the library, without scoring")
    p.add_argument("config", help="benchmark TOML file, e.g. configs/benchmark.toml")
    p.set_defaults(func=_backfill)

    p = sub.add_parser("library", help="the image library that runs reuse: list it, or import earlier runs")
    library = p.add_subparsers(dest="library_command", required=True)
    q = library.add_parser("list", help="every group of images with the same settings, and its size")
    q.add_argument("--library", default="library")
    q.set_defaults(func=_library_list)
    q = library.add_parser("import", help="add the images of earlier run directories to the library")
    q.add_argument("run_dirs", nargs="+")
    q.add_argument("--library", default="library")
    q.set_defaults(func=_library_import)

    p = sub.add_parser("score", help="re-evaluate and re-score an existing run directory")
    p.add_argument("run_dir")
    p.add_argument("--extractor", choices=sorted(EXTRACTORS), help="override the run's extractor")
    p.add_argument("--references", help="directory of real cube photos to score alongside the run")
    p.set_defaults(func=lambda a: _score(RunStore(Path(a.run_dir)), a.extractor, a.references))

    p = sub.add_parser("site", help="build a shareable static site (e.g. for GitHub Pages) from a run's newest report")
    p.add_argument("run_dir")
    p.add_argument("out_dir", help="directory to write the site into (its contents are replaced, a .git is kept)")
    p.add_argument("--url", required=True, help="where the site will be published, for the sharing tags")
    p.add_argument("--share-image", required=True, metavar="SAMPLE", help="sample id whose image becomes the preview")
    p.add_argument("--share-alt", default="An AI-generated Rubik's Cube", help="description of the preview image")
    p.add_argument("--title", default="RubiksBench")
    p.add_argument("--description", default="How good are Text-to-Image models at generating valid Rubik's Cubes?")
    p.set_defaults(func=cmd_site)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
