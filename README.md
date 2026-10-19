# RubiksBench

[![tests](https://github.com/kvlxai/RubiksBench/actions/workflows/tests.yml/badge.svg)](https://github.com/kvlxai/RubiksBench/actions/workflows/tests.yml)

Can image models draw a **valid scrambled 3×3 Rubik's Cube**? RubiksBench generates images from a
prompt, reads the visible stickers, and checks whether a real, reachable cube state could look like
that.

**Latest results:** [kvlxai.github.io/RubiksBench](https://kvlxai.github.io/RubiksBench/) (15 models, 10 images each, plus real-cube reference photos).

[![RubiksBench results](https://kvlxai.github.io/RubiksBench/preview.png)](https://kvlxai.github.io/RubiksBench/)

```sh
uv sync
uv run pytest
uv run rubiksbench validate observation.json          # check a hand-read cube (see Observation format)
uv run rubiksbench validate-image cube.png            # extract and score an image directly
uv run --env-file .env rubiksbench models "Flux 3"   # find model ids (--all includes LoRAs/video)
uv run rubiksbench models flux --backend huggingface  # Hugging Face models served by Inference Providers
uv run --env-file .env rubiksbench run configs/benchmark.toml   # .env: RUNWARE_API_KEY / HF_TOKEN / FAL_KEY (or FAL_API_KEY);
                                                        #   generates only the images the library lacks
uv run rubiksbench library list                         # the stored images, per model and settings
uv run rubiksbench library import runs/<run>            # add an earlier run's images to the library
uv run rubiksbench score runs/<run>                     # re-evaluate + re-score an existing run (no API calls;
                                                        #   a line per image with the time left)
uv run --env-file .env rubiksbench backfill configs/benchmark.toml  # generate what the library lacks, no scoring
uv run rubiksbench site runs/<run> site/ --url https://<you>.github.io/RubiksBench/ --share-image <sample>
                                                  # shareable static site of the newest report (GitHub Pages)
```

`validate-image` prints the same text summary as a benchmark run: the verdict, failure flags,
validity percentage, and rubric points (for example, `INVALID (duplicate_centers) · validity 32% · 9/17`).
Add `-v` / `--verbose` to list failed rubric checks with their descriptions. Blocked or inconclusive
checks are not listed as failures. Use `--json` for the full JSON report instead; it overrides
`--verbose` so the output remains valid JSON. Use `--extractor sam` or `--extractor annotation` to select another reader,
and `--no-scramble` to skip computing a witness scramble. The default `vision` extractor expects
a three-face corner view and writes `<image stem>.extraction.jpg` beside the input. Its first
use downloads the background-removal model (see Sticker extraction below). An unreadable image
or uncertain extraction is reported as `INCONCLUSIVE`; a missing file exits with an error.

## Scoring

### Validity (the headline)

Validity says how close an image comes to a valid, scrambled, standard 3×3 cube, from 0% to 100%:

```
validity = (1 − (repairs + defects) / chance) × scramble factor        (clipped to 0–100%)
```

- **repairs** is the fewest visible stickers you would have to recolor to get a valid standard cube.
  It's exact: a dynamic program over every corner and edge slot, for each of the 24 orientations of the
  standard colors, keeping only reachable states (each piece once, twist, flip and parity). Hidden
  stickers are free, and the report lists which stickers to recolor.
- **defects** are stickers no recoloring fixes: letters, two colors on one sticker, or a second shade
  of a color.
- **chance** is the average repair distance of random colors on the 27 visible stickers: 4.45. The
  three hidden faces leave so much freedom that random stickers are only about four recolorings from a
  valid cube. So 0% means "no better than random", not "everything wrong".

- **scramble factor** holds a barely scrambled cube or a designed pattern in slightly less regard. It
  is 100% for a cube that looks like a random scramble, sliding down to 80% as fewer stickers differ
  from their face's center or more faces show a symmetric pattern (see the "scrambled" check below).

A valid, randomly scrambled cube scores 100%, and each sticker to fix costs about 22 points. An image
that isn't read as a 3×3 cube, or isn't scrambled at all (every face one color), scores 0%. Every
mistake costs only the stickers it takes to fix it, so nothing is counted twice. Each model's score is
the mean over its images, with a 95% CI. The models are ranked by it, and the gallery is sorted by it.

### Rubric (what is wrong)

Every image is graded on 17 checks. Each check earns 1 point when passed, ½ when inconclusive and 0
when failed. Every mistake costs its point once. A check is "blocked" (it scores 0, but doesn't count
as a mistake of its own) in two cases:
- **It can't be decided.** Sticker checks need the image read as a 3×3 cube, so a box-shaped cube
  stops at the 3×3 check. Color checks also need confidently named colors.
- **Its failure would only be a consequence of an earlier failure.** A duplicate center already rules
  out the standard center order, a seventh color rules out a standard palette, and an impossible piece
  rules out a solvable cube.

Independent mistakes always count, whatever failed before. A corner showing blue twice fails
"possible pieces" even next to two white centers. Opposite colors on side-by-side centers fail "center
order" even when another center is duplicated. So the same error is always reported the same way. The
checks live in `evaluation/rubric.py`, and every report lists them with what each one needs.

1. cube detected
2. cube shape
3. 3×3 grid
4. cubies in place (none sunk in, sticking out or deformed on its own)
5. plain stickers (no letters or markings; a logo on a center is allowed)
6. single-colored stickers (no sticker split between two clearly different colors; a logo on a
   center is allowed)
7. at most six colors (fewer is fine: a color can be entirely on the hidden faces)
8. one shade per color (on each face, all stickers of a color share a shade, so no lime next to
   green; real cubes stay within 3° of hue, a sticker 15° off is a second shade)
9. standard palette (every visible color is white, yellow, red, orange, green or blue)
10. standard color pairs (no piece joins white–yellow, red–orange or blue–green)
11. scrambled (looks like a random scramble: at least 15 of the 24 visible stickers differ from
    their face's center, and at most one face shows a symmetric pattern; fewer than 1% of random
    scrambles fail this, while a cube one move from solved or a designed pattern does)
12. distinct centers
13. standard center order
14. possible pieces
15. unique pieces
16. solvable
17. solvable (standard)

Where a failed check has a clear culprit, the rubric records it as a *mark*. Marked stickers are
named by facelet (`U9`, `F2`, ...), and faces that are not 3×3 are marked as a whole. Examples:
the two centers that share a color, the corner or edge that can't exist, both copies of a
duplicated piece, the stickers with letters on them, a sticker showing two colors or a second shade, a cubie out of place, the symmetric faces
of a designed pattern. Which piece counts as "impossible" can depend on the color scheme, so the smallest set of culprits is shown, among the schemes that got
furthest. Twist, flip and parity failures belong to the whole cube and are not marked. Blocked
checks are not marked either.

### Who it fools

Each check also says who would notice it failing: anyone at a glance (wrong grid, letters, an extra
color, not scrambled), anyone who has held a cube (two centers alike, a piece showing one color twice,
a second shade, a cubie out of place), a cuber looking carefully (opposite colors on a piece, impossible or duplicated
pieces, center order) or only a solver (twist, flip, parity). A designed or shallow scramble doesn't
count here: it is still a valid cube, with nothing to see through. An image fools everyone below the
first who would notice, so each image fools *no one*, *a casual viewer*, *a cube owner*, *a cuber* or
*everyone*. The report shows this per image and per model ("fools a cube owner" = gets past someone
who has held a cube).

### Solutions

For a valid image the report shows a solution: the moves that solve the cube the validator completed
the image to, found with Kociemba's two-phase solver and checked by replaying them in an independent
simulator (`magiccube`). It is *a* solution, not *the* solution: three faces are hidden, so many cubes
match the image, and the two-phase solver is not optimal, so shorter solutions likely exist. When the
colors are the standard ones, a link opens it in [Twizzle](https://alpha.twizzle.net/edit/), turned to
show the same three faces as the image.

### Reading the image

- A sticker whose color is close to a boundary (red or orange?) is evaluated both ways, for up to four
  such stickers. Whatever every reading agrees on is decided; a check that depends on the reading is
  inconclusive, and validity takes the more favourable reading, so uncertainty in reading the image
  never costs the model.
- A face that fails the 3×3 check while being much blurrier than the rest of the cube (a thin, out of
  focus top face at a shallow camera angle) is read as 3×3 like its sides, since a cube's faces share
  their grid. Cells that straddle two stickers in the stretched image are left unknown rather than
  guessed.

The HTML report shows each model's mean validity, who its images fool and its rubric score, plus the
share of points earned per check. The verdicts (`VALID_STANDARD`, …) are still reported next to them.

## Benchmark and models files

A benchmark file holds what is the same for every model, and picks models by name, per provider,
from a models file:

```toml
# configs/benchmark.toml
extractor = "vision"     # or "sam" / "annotation"; `score` reuses it, `--extractor` overrides
prompt = "A valid scrambled Rubik's Cube"
iterations = 10          # images per model; random seed per image unless `seed = N` is set
width = 1024
height = 1024
models_file = "models.toml"   # next to this file
library = "library"           # where generated images are kept and reused

[models]
runware = ["FLUX 3 Image", "Nano Banana 2.1"]
fal = ["Some fal-only model"]
```

The models file lists the models each provider serves (`runware` needs `RUNWARE_API_KEY`, `fal`
`FAL_KEY`, `huggingface` `HF_TOKEN`), with their generation settings. Anything left out takes the
benchmark's value, or the provider's default:

```toml
# configs/models.toml
[runware]
"FLUX 3 Image" = "bfl:flux@3-image"   # just the model id

[fal."Some fal-only model"]
id = "fal-ai/some-model"
steps = 30
cfg = 4.5
width = 1024             # overrides the benchmark's width and height
height = 1024
seed = 42                # fixed seeds 42, 43, ...
extra = { sampler = "euler" }   # provider-specific parameters, sent as they are
```

A name picked on two providers is shown as "Name (provider)".

### Image library

Every generated image is kept in the benchmark's `library` directory, grouped by provider, model id and every request setting
except the seed (prompt, size, steps, CFG, extra parameters):
`library/<provider>/<model id>/<settings hash>/` holds `settings.json`, `generations.jsonl` and
`images/`. A run takes each model's images from its group, oldest first (with fixed seeds, the
images with those seeds), and generates only the ones missing. Adding a new model to a benchmark
and running it again therefore pays only for the new model, and scores it alongside the earlier
ones in one report. Changing the prompt or a model's settings starts a new group. Reused samples
carry `reused_from` in the run's `generations.jsonl`, and their images are copied into the run, so
a run directory stays self-contained. `library import` adds the images of runs made before the
library existed (as copies; the run is left as it is).

`backfill configs/benchmark.toml` only fills the library: for each model it requests the images
still missing up to `iterations` (with fixed seeds, the missing seeds), once each, without scoring.
Failed requests are logged and requested again by the next `backfill`. A `run` afterwards then only
scores. A `run` without a backfill generates the missing images itself, and counts failures as
generation failures in its report.

Models run one after another, one request at a time, and each image is scored as soon as it is
generated or taken from the library (the verdict is printed on its log line). Ctrl-C stops the run and still reports what was
generated so far. Each run directory holds `run.json`,
`generations.jsonl` (every attempt, including failures, with seed and the exact request),
`images/` and `evaluations.jsonl`. Every scoring pass (at the end of `run`, or via `score`) also writes
timestamped results:
- `results-<UTC>.json` holds the raw data.
- `results-<UTC>.html` is a self-contained report with light and dark themes. It has:
  - the per-model scores: mean validity with an inline 95% CI strip, the rubric score and an outcome bar;
  - a points-per-check heatmap;
  - a gallery of every image, most valid first, with the stickers to fix (or, for a valid cube, a
    solution and a link to view it in 3D), its per-check results, its extraction overlay and the culprits of
    failed checks outlined on the image (hover an outline to see why; a checkbox hides them);
  - the failure flags and the list of checks.

Models that reject a `seed` (e.g. OpenAI, FLUX 3 Image) are sent without one. Their samples are
marked `seed_supported: false`.

Sticker extraction is pluggable (`extractor` in the TOML, or `--extractor` to override):

- `vision` (default) reads three-face corner views automatically:
  1. `rembg` (U2-Net) cuts out the cube; the first use downloads its ~176 MB model.
  2. The silhouette's hexagon and its vanishing points give the three faces. OpenCV flattens each face
     and counts its rows and columns, to check it is a 3×3 grid.
  3. An ideal cube of 27 cubies is posed in 3D (rotation, position and focal length) to match the
     image, starting from the hexagon: its predicted seams are pulled onto the image's black plastic
     and color edges, with the worst few cells ignored so a single deformed cubie cannot pull it off.
     All three faces are fitted together, so a thin or blurred face borrows the others' precision.
  4. Each sticker is read inside its fitted cell, and its color named by its CIELAB hue (white by low
     chroma). A lone cell that is mostly not sticker color (plastic or background) holds a cubie
     that is sunk in, sticking out or deformed: "cubies in place" fails.

  Whether the image shows a Rubik's cube at all is decided first:
  - **With the `ml` extra** (`uv sync --extra ml`), CLIP zero-shot recognition decides. An image that
    clearly shows something else (an animal, a die, a plain box) is `INVALID` with `not_a_cube` and
    scores 0.
  - **Without it**, the cube's silhouette has to fill its hexagonal outline (`hexagon_fit` ≥ 0.93).
    Anything else is inconclusive.

  Anything uncertain gives `INCONCLUSIVE` (`vision_low_confidence`), never a guess. Examples: no
  hexagonal silhouette, an inconsistent layout, one odd face, or a color near a hue boundary.
  The extractor writes `images/<sample>.extraction.jpg` with every label it read.
- `sam` is a sticker-first reader built on Meta's SAM 2.1 (PyTorch, via `transformers`).
  1. SAM proposes a mask for every sticker.
  2. Each sticker's edge directions say which face it is on.
  3. Its position along those directions gives its row and column.

  It doesn't use the silhouette, so it is an independent cross-check. It is optional
  (`uv sync --extra ml`) and takes ~15 s per image.
- `annotation` reads a hand annotation `images/<sample>.json` (same format as below), for checking
  the vision extractor or scoring images it cannot read.

## Layout

- `cube/`: the partial-state validator. Kociemba facelet conventions; an exact search for a
  reachable completion under each candidate color scheme; witness scrambles from `kociemba`,
  replayed independently in `magiccube`.
- `evaluation/`: verdicts (`VALID_STANDARD`, `VALID_NONSTANDARD`, `INVALID`, `INCONCLUSIVE`) and flags.
- `backends/`: `base.py` defines `ImageBackend` and a registry. `runware.py` uses the official
  `runware-sdk`. `huggingface.py` uses `huggingface_hub` (Inference Providers). `fal.py` uses `fal-client` and reads each
  endpoint's published input schema, so it sends only the fields that model accepts (size, seed, steps). To add a provider, subclass `ImageBackend`, decorate it with `@register_backend`
  and import it in `backends/__init__.py`.
- `storage/`: run directories.
- `vision/`: extractors (image → observation): `cube_detector.py`, `grid_extractor.py`,
  `color_classifier.py`, `extractor.py`, `sam_extractor.py`.
- `runner.py` / `evaluation/scoring.py`: serial generation, evaluation and per-model scores
  (standard-valid rate with 95% Wilson CI, abstract/invalid/inconclusive rates, flags, cost).

## Observation format

A JSON object mapping faces to 9-sticker grids, read row by row, with `.` for unknown. For a
three-face view, the top face is `U`, front-left is `F` and front-right is `R`. Each grid's top
row is the edge nearest the U face (for U itself, the edge nearest the back).

## Versioning

Any change that can alter a score bumps the **minor** version (0.4.x → 0.5.0), and with it the
evaluator version that every report and result file records. Scores are comparable only when the
major and minor versions match: results from 0.4.5 and 0.4.6 can be compared, results from 0.4.x and
0.5.x cannot. Patch releases change everything else (report layout, tooling, docs) but never a score.

## License

MIT, see [LICENSE](LICENSE).
