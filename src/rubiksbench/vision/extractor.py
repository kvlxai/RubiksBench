"""Sticker extraction: image -> partial cube observation.

Extractors are pluggable so automated vision can be compared against manual
annotation. An extractor never decides validity; it reports what it saw and
how sure it is, and the evaluator turns that into a verdict.
"""

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..cube.facelets import FACES
from ..cube.observation import Observation
from ..evaluation.taxonomy import Flag
from .color_classifier import StickerColor, classify, off_shade
from .cube_detector import CubeDetector, FaceLayout
from .cube_model import cell_residuals, fit_cube, out_of_place
from .extraction import FAIL, INCONCLUSIVE, PASS, Extraction, Extractor, geometry
from . import recognizer
from .grid_extractor import SIZE, read_face

# PyTorch for SAM (and CLIP) loads only when first used (`uv sync --extra ml`).
from .sam_extractor import SamExtractor



class AnnotationExtractor:
    """Reads a hand annotation stored next to the image as ``<image>.json``.

    The file holds face grids as accepted by ``Observation.from_faces``, plus
    optional ``"flags"`` (e.g. ``["wrong_grid_size"]``) for images that cannot
    be read as a 3x3 cube at all.
    """

    name = "annotation"

    def extract(self, image_path: Path) -> Extraction:
        path = image_path.with_suffix(".json")
        if not path.exists():
            return Extraction(self.name, None, details={"missing": str(path.name)}, quality={"cube": INCONCLUSIVE})
        data = json.loads(path.read_text())
        flags = [Flag(f) for f in data.pop("flags", [])]
        observation = Observation.from_faces(data) if data else None
        grid = FAIL if Flag.WRONG_GRID_SIZE in flags else PASS
        shape = FAIL if Flag.GEOMETRY_INVALID in flags else PASS
        stickers = FAIL if Flag.MARKED_STICKERS in flags else PASS
        single = FAIL if Flag.SPLIT_STICKERS in flags else PASS
        shades = FAIL if Flag.OFF_SHADE_STICKERS in flags else PASS
        in_place = FAIL if Flag.CUBIES_OUT_OF_PLACE in flags else PASS
        quality = {"cube": PASS, "shape": shape, "grid": grid, "in_place": in_place, "stickers": stickers,
                   "single_color": single, "shades": shades, "colors": PASS}
        return Extraction(self.name, observation, 1.0, flags, {"annotation": path.name}, quality)


class VisionExtractor:
    """Automatic extraction for a three-face corner view of the cube.

    Pipeline: background removal (rembg) -> hexagonal silhouette -> three face
    quads -> perspective-flattened faces (OpenCV) -> 3x3 grid check -> sticker
    colors named by hue (CIELAB). Anything it is unsure of makes the sample
    INCONCLUSIVE rather than guessed. Writes ``<image>.extraction.jpg`` showing
    what it read.
    """

    name = "vision"
    # Smallest hue/chroma margin accepted for a sticker's color (see color_classifier).
    min_color_margin = 0.5
    # Up to this many stickers below that margin are kept with both possible colors (the
    # evaluator decides what holds either way); more make the colors inconclusive.
    max_uncertain_stickers = 4
    # A face that does not read as 3x3 while this much blurrier than the cube's sharpest face
    # (Laplacian variance of the flattened face) cannot have its rows counted: a thin, out of
    # focus face at a shallow angle. When the other faces are 3x3 it is read as 3x3 like
    # them, since a cube's faces share their grid. (Ideogram 4.5's blurred top face, misread
    # as 2x3: 0.05; genuine non-3x3 faces: 0.24 up.)
    min_face_sharpness = 0.15
    # On such a face, a cell this mixed (share of a clearly different second color) straddles
    # two stickers, so it is left unknown rather than guessed.
    mixed_cell = 0.15
    # A face whose rows or columns clearly (by this factor over the runner-up) are not
    # 3 makes the cube INVALID; a weaker reading only makes it INCONCLUSIVE. Genuine
    # 3x3 faces have so far always read 3 on both axes.
    min_grid_dominance = 1.25
    # Largest geometric inconsistency of the face layout, relative to cube size
    # (0.001-0.03 on clean three-face views).
    max_inner_spread = 0.08
    # A sticker counts as marked above this share of sharp-edge pixels (plain: ~0-0.03).
    marked_sticker = 0.04
    # Letters on at least this many of the 24 visible non-center stickers make the image
    # invalid (lettered cubes: 21-22; clean cubes: 0-12). Centers may carry a brand logo.
    min_marked_stickers = 16
    # A non-center sticker shows two colors when this share of it is a clearly different
    # color (two-colored stickers: 0.20-0.34; plain ones up to 0.02).
    split_sticker = 0.10
    # Smallest silhouette/hexagon overlap for a clean cube view (cubes: 0.95-0.99;
    # a rabbit 0.83, a teapot 0.71, a cube held in a hand 0.71).
    min_hexagon_fit = 0.93

    def __init__(self, write_overlay: bool = True, recognize: bool | None = None, fit_model: bool = True):
        self.detector = CubeDetector()
        self.write_overlay = write_overlay
        # Pose an ideal cube to the image (cube_model) and read the faces from the fit.
        self.fit_model = fit_model
        # CLIP recognition runs when its optional dependencies are installed, unless disabled.
        use_clip = recognizer.available() if recognize is None else recognize
        self.recognizer = recognizer.CubeRecognizer() if use_clip else None

    @property
    def cache_variant(self) -> str:
        """What, besides the code, decides this extractor's readings (see storage/evaluation_cache)."""
        clip = f"clip_{self.recognizer.model_id}" if self.recognizer else "silhouette"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"vision-{clip}-{'fit' if self.fit_model else 'nofit'}"
                      f"{'' if self.write_overlay else '-nooverlay'}")

    def _recognize(self, image_path: Path) -> tuple[str | None, dict[str, Any]]:
        if self.recognizer is None:
            return None, {"method": "silhouette"}
        r = self.recognizer.recognize(image_path)
        return r.result, {"method": "clip", "model": self.recognizer.model_id, "rubiks_probability": r.rubiks_probability,
                          "top_label": r.top_label, "scores": r.scores}

    def extract(self, image_path: Path) -> Extraction:
        image = cv2.imread(str(image_path))
        if image is None:
            return Extraction(
                self.name, None, flags=[Flag.VISION_LOW_CONFIDENCE], details={"error": "unreadable image"},
                quality={"cube": INCONCLUSIVE},
            )
        recognized, recognition = self._recognize(image_path)
        if recognized == FAIL:
            return Extraction(
                self.name, None, flags=[Flag.NOT_A_CUBE], details={"recognition": recognition},
                quality={"cube": FAIL},
            )
        layout = self.detector.layout(self.detector.mask(image))
        if layout is None:
            return Extraction(
                self.name, None, flags=[Flag.VISION_LOW_CONFIDENCE],
                details={"recognition": recognition, "error": "no hexagonal silhouette: not a three-face corner view"},
                quality={"cube": recognized or INCONCLUSIVE, "shape": INCONCLUSIVE},
            )

        # The silhouette's faces count the grid (an ideal cube would make any box look 3x3); the
        # fitted cube's faces, when the fit improves on its start, read the stickers.
        fitted = layout
        misplaced: list[str] = []
        model: dict[str, Any] = {"used": False}
        if self.fit_model and (cube := fit_cube(image, layout.hexagon, layout.inner)) is not None:
            model = {"used": cube.score > cube.start_score, "score": round(cube.score, 3),
                     "start_score": round(cube.start_score, 3), "gap": round(cube.gap, 3)}
            if model["used"]:
                fitted = replace(layout, quads=cube.quads(), inner=cube.corner("C").astype(np.float32))
                model["quads"] = {f: q.round(1).tolist() for f, q in fitted.quads.items()}
                residuals = cell_residuals(image, cube)
                misplaced = out_of_place(residuals)
                model["residuals"] = {n: [round(a, 3), round(b, 3)] for n, (a, b) in residuals.items()}
        details: dict[str, Any] = {"recognition": recognition, "layout": layout.to_dict(), "model": model, "faces": {}}
        flags: list[Flag] = []
        clean = layout.hexagon_fit >= self.min_hexagon_fit
        # Without CLIP, a clean cube silhouette is the evidence that a cube is there at all.
        cube = recognized or (PASS if clean else INCONCLUSIVE)
        quality = {"cube": cube, "shape": PASS, "grid": PASS, "in_place": PASS, "stickers": PASS, "single_color": PASS,
                   "shades": PASS, "colors": PASS}
        if cube != PASS:
            flags.append(Flag.VISION_LOW_CONFIDENCE)
        if layout.inner_spread > self.max_inner_spread or not clean:
            flags.append(Flag.VISION_LOW_CONFIDENCE)
            quality["shape"] = INCONCLUSIVE
        faces: dict[str, list[str]] = {}
        colors: dict[str, list[StickerColor]] = {}
        grids: dict[str, tuple[int, int]] = {}
        flattened = {face: read_face(image, quad) for face, quad in layout.quads.items()}
        sampled = flattened if fitted is layout else {face: read_face(image, q) for face, q in fitted.quads.items()}
        sharpness = {face: sharpness_of(grid.image) for face, grid in flattened.items()}
        for face, grid in flattened.items():
            relative = sharpness[face] / (max(sharpness.values()) or 1.0)
            inferred = (grid.columns, grid.rows) != (3, 3) and relative < self.min_face_sharpness
            if inferred and not all((g.columns, g.rows) == (3, 3) for f, g in flattened.items() if f != face):
                details["faces"][face] = {
                    "skipped": f"too blurred to count its rows ({relative:.0%} of the sharpest face's detail)"}
                continue
            cells = sampled[face]
            colors[face] = [classify(rgb) for rgb in cells.rgb]
            faces[face] = [
                "." if inferred and mixed > self.mixed_cell else c.label
                for c, mixed in zip(colors[face], cells.second_color)
            ]
            grids[face] = (3, 3) if inferred else (grid.columns, grid.rows)
            for count, dominance in ((grid.columns, grid.column_dominance), (grid.rows, grid.row_dominance)):
                if count != 3 and not inferred:
                    # Rows are only counted on faces cut from a clean outline: a turned layer or a
                    # background the cut-out mistook for cube skews the faces, and their counts with them.
                    strong = dominance >= self.min_grid_dominance and quality["shape"] == PASS
                    flags.append(Flag.WRONG_GRID_SIZE if strong else Flag.VISION_LOW_CONFIDENCE)
                    if quality["grid"] != FAIL:
                        quality["grid"] = FAIL if strong else INCONCLUSIVE
            details["faces"][face] = {
                "columns": grid.columns,
                "rows": grid.rows,
                "column_dominance": round(grid.column_dominance, 2),
                "row_dominance": round(grid.row_dominance, 2),
                "marking": [round(m, 3) for m in cells.marking],
                "second_color": [round(s, 3) for s in cells.second_color],
                **({"inferred_grid": f"too blurred to count its rows ({relative:.0%} of the sharpest face's "
                    f"detail; measured {grid.columns}x{grid.rows}), read as 3x3 like the other faces"}
                   if inferred else {}),
                "stickers": [
                    {"label": c.label, "margin": round(c.margin, 2), "rgb": [round(v) for v in rgb],
                     "hue": round(c.hue, 1), "chroma": round(c.chroma, 1)}
                    for c, rgb in zip(colors[face], cells.rgb)
                ],
            }
        # Letters, split stickers and shades are judged on sharp faces only.
        read = {face: d for face, d in details["faces"].items() if "skipped" not in d and "inferred_grid" not in d}
        marked = sum(m > self.marked_sticker for f in read.values() for i, m in enumerate(f["marking"]) if i != 4)
        details["marked_stickers"] = marked
        details["geometry"] = geometry(
            (image.shape[1], image.shape[0]), fitted.quads,
            {f"{face}{i + 1}": cell for face, cells in sticker_quads(fitted, grids).items() for i, cell in enumerate(cells)},
            grids,
            [f"{face}{i + 1}" for face, d in read.items() for i, m in enumerate(d["marking"])
             if i != 4 and m > self.marked_sticker],
            # Centers are left out: they may carry a colored brand logo.
            # A displaced cubie's cell holds plastic or background, not a split or a second shade.
            split := [f"{face}{i + 1}" for face, d in read.items() if grids[face] == (3, 3)
                      for i, s in enumerate(d["second_color"]) if i != 4 and s > self.split_sticker
                      and f"{face}{i + 1}" not in misplaced],
            shades := [f"{face}{i + 1}" for face, face_colors in colors.items() if face in read
                       and grids[face] == (3, 3) for i in off_shade(face_colors, self.min_color_margin)
                       if f"{face}{i + 1}" not in misplaced],
            misplaced := [name for name in misplaced if name[0] in read and grids[name[0]] == (3, 3)],
        )
        # Defects measured on faces cut from a skewed outline are as unreliable as their grid counts.
        trusted = quality["shape"] == PASS
        for found, flag, check in ((misplaced, Flag.CUBIES_OUT_OF_PLACE, "in_place"),
                                   (shades, Flag.OFF_SHADE_STICKERS, "shades"),
                                   (split, Flag.SPLIT_STICKERS, "single_color"),
                                   (marked >= self.min_marked_stickers, Flag.MARKED_STICKERS, "stickers")):
            if found:
                flags.append(flag if trusted else Flag.VISION_LOW_CONFIDENCE)
                quality[check] = FAIL if trusted else INCONCLUSIVE
        known = {9 * FACES.index(face) + i: c for face, face_colors in colors.items()
                 for i, c in enumerate(face_colors) if faces[face][i] != "."}
        min_margin = min(c.margin for c in known.values())
        uncertain = {i: c for i, c in known.items() if c.margin < self.min_color_margin}
        alternatives: dict[int, str] = {}
        if len(uncertain) > self.max_uncertain_stickers:
            flags.append(Flag.VISION_LOW_CONFIDENCE)
            quality["colors"] = INCONCLUSIVE
        else:
            alternatives = {i: c.alternative for i, c in uncertain.items()}
            details["uncertain_stickers"] = {
                f"{FACES[i // 9]}{i % 9 + 1}": [c.label, c.alternative] for i, c in uncertain.items()
            }

        if self.write_overlay:
            overlay_path = image_path.with_suffix(".extraction.jpg")
            cv2.imwrite(str(overlay_path), draw_overlay(image, fitted, faces, grids))
            details["overlay"] = overlay_path.name
        observation = Observation.from_faces(faces)
        return Extraction(self.name, observation, min_margin, sorted(set(flags)), details, quality, alternatives)


def sharpness_of(face_bgr: np.ndarray) -> float:
    """Detail in a flattened face: variance of its Laplacian (low for blur or a stretched sliver)."""
    return float(cv2.Laplacian(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def sticker_quads(layout: FaceLayout, grids: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
    """Each 3x3 face's nine sticker cells in image pixels (9x4x2, row by row)."""
    target = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    unit = np.float32([[[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]] for r in range(3) for c in range(3)]) / 3
    out = {}
    for face, quad in layout.quads.items():
        if grids.get(face) == (3, 3):
            back = cv2.getPerspectiveTransform(target, np.float32(quad))
            out[face] = cv2.perspectiveTransform(unit.reshape(-1, 1, 2), back).reshape(9, 4, 2)
    return out


def draw_overlay(
    image: np.ndarray, layout: FaceLayout, faces: dict[str, list[str]], grids: dict[str, tuple[int, int]]
) -> np.ndarray:
    """The detected faces and the color label read for every sticker.

    A face that does not read as 3x3 shows its measured columns x rows instead.
    """
    out = image.copy()
    for quad in layout.quads.values():
        cv2.polylines(out, [quad.astype(np.int32)], True, (0, 255, 0), 2)
    target = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])
    centres = np.float32([[(c + 0.5) * SIZE / 3, (r + 0.5) * SIZE / 3] for r in range(3) for c in range(3)])
    scale = max(image.shape[:2]) / 1024
    for face, quad in layout.quads.items():
        if face not in grids:  # left unread (seen edge-on)
            continue
        back = cv2.getPerspectiveTransform(target, np.float32(quad))
        points = cv2.perspectiveTransform(centres.reshape(-1, 1, 2), back).reshape(-1, 2)
        if grids[face] != (3, 3):
            x, y = points[4]
            text = f"{face}: {grids[face][0]}x{grids[face][1]}"
            org = (int(x - 60 * scale), int(y + 12 * scale))
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4 * scale, (0, 0, 0), int(9 * scale))
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4 * scale, (0, 0, 255), int(3 * scale))
            continue
        for (x, y), label in zip(points, faces[face]):
            org = (int(x - 9 * scale), int(y + 9 * scale))
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 0, 0), int(5 * scale))
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 255, 255), int(2 * scale))
    return out


EXTRACTORS: dict[str, type[Extractor]] = {
    "vision": VisionExtractor,
    "sam": SamExtractor,
    "annotation": AnnotationExtractor,
}
