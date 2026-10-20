"""Sticker-first extraction with Meta's Segment Anything 2.1 (PyTorch, via transformers).

Independent of the silhouette-based ``vision`` extractor, so the two can
cross-check each other:

1. SAM 2.1 "segment everything" proposes masks; sticker-like ones (convex,
   four-cornered, plausibly sized) become quads.
2. Stickers on a corner view have edges in three image directions: ``a``
   (down-right, along T-UR), ``b`` (down-left, along T-UL) and ``c`` (down,
   the vertical edges). A sticker's pair of directions names its face:
   {a, b} = U, {a, c} = F, {b, c} = R.
3. Within a face, positions along its two directions give row and column:
   U col along a, row along b; F col along a, row along c; R col along -b,
   row along c (Kociemba facelet order, as in ``cube_detector``).

Needs the optional ``ml`` dependencies: ``uv sync --extra ml``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..cube.observation import Observation
from ..evaluation.taxonomy import Flag
from .color_classifier import classify, off_shade, second_color_share
from .extraction import FAIL, INCONCLUSIVE, PASS, Extraction, geometry
from .grid_extractor import _STROKE

MODEL_ID = "facebook/sam2.1-hiera-small"
# Sticker area as a fraction of the image.
MIN_AREA, MAX_AREA = 0.001, 0.04
MIN_SOLIDITY = 0.9


@dataclass
class Sticker:
    quad: np.ndarray  # 4x2 corners
    centre: np.ndarray
    rgb: tuple[float, float, float]
    directions: tuple[np.ndarray, np.ndarray]  # unit vectors of its two edge pairs
    size: float  # mean edge length
    marking: float  # share of sharp-edge pixels inside the sticker (letters, logos)
    second_color: float  # share of the sticker in a clearly different second color


def _sticker(mask: np.ndarray, image: np.ndarray) -> Sticker | None:
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if not MIN_AREA * m.size < area < MAX_AREA * m.size:
        return None
    hull = cv2.convexHull(contour)
    if area / max(cv2.contourArea(hull), 1.0) < MIN_SOLIDITY:
        return None
    approx = cv2.approxPolyDP(hull, 0.06 * cv2.arcLength(hull, True), True)
    if len(approx) != 4:
        return None
    quad = approx.reshape(4, 2).astype(np.float32)
    edges = [quad[(i + 1) % 4] - quad[i] for i in range(4)]
    # Opposite edges point opposite ways; average each pair as an undirected direction.
    directions = []
    for e1, e2 in ((edges[0], -edges[2]), (edges[1], -edges[3])):
        d = e1 / np.linalg.norm(e1) + e2 / np.linalg.norm(e2)
        directions.append(d / np.linalg.norm(d))
    inner = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=2).astype(bool)
    pixels = image[inner if inner.any() else m.astype(bool)]
    b, g, r = np.median(pixels, axis=0)
    size = float(np.mean([np.linalg.norm(e) for e in edges]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    magnitude = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    core = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=4).astype(bool)
    marking = float((magnitude[core] > _STROKE).mean()) if core.any() else 0.0
    second = second_color_share(image[core]) if core.any() else 0.0
    return Sticker(quad, quad.mean(axis=0), (float(r), float(g), float(b)), tuple(directions), size, marking, second)


def _dedupe(stickers: list[Sticker]) -> list[Sticker]:
    kept: list[Sticker] = []
    for s in sorted(stickers, key=lambda s: s.size):
        if all(np.linalg.norm(s.centre - k.centre) > 0.4 * k.size for k in kept):
            kept.append(s)
    return kept


def _main_cluster(stickers: list[Sticker]) -> list[Sticker]:
    """Keep the stickers connected to each other (drops stray background quads)."""
    if not stickers:
        return []
    n = len(stickers)
    linked = [[j for j in range(n) if j != i and np.linalg.norm(stickers[i].centre - stickers[j].centre)
               < 1.8 * max(stickers[i].size, stickers[j].size)] for i in range(n)]
    best: list[int] = []
    seen: set[int] = set()
    for start in range(n):
        if start in seen:
            continue
        group, stack = [], [start]
        seen.add(start)
        while stack:
            i = stack.pop()
            group.append(i)
            for j in linked[i]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        best = max(best, group, key=len)
    return [stickers[i] for i in best]


def _axis_name(d: np.ndarray) -> str:
    """Classify an undirected edge direction as a (down-right), b (down-left) or c (vertical)."""
    x, y = d if d[1] >= 0 else -d  # point it downwards
    angle = np.degrees(np.arctan2(y, x))  # 0 = right, 90 = down, 180 = left
    if 60 <= angle <= 120:
        return "c"
    return "a" if angle < 90 else "b"


def _canonical(d: np.ndarray, name: str) -> np.ndarray:
    """Orient a direction: a points right, b points left, c points down."""
    if name == "a":
        return d if d[0] >= 0 else -d
    if name == "b":
        return d if d[0] <= 0 else -d
    return d if d[1] >= 0 else -d


FACE_OF = {frozenset("ab"): "U", frozenset("ac"): "F", frozenset("bc"): "R"}
# (column axis, column sign, row axis) per face.
GRID_AXES = {"U": ("a", 1, "b"), "F": ("a", 1, "c"), "R": ("b", -1, "c")}


def _bins(values: np.ndarray, pitch: float) -> np.ndarray:
    """Group sorted positions into rows/columns split by gaps of at least half a sticker."""
    order = np.argsort(values)
    labels = np.zeros(len(values), int)
    current = 0
    for prev, nxt in zip(order, order[1:]):
        if values[nxt] - values[prev] > 0.5 * pitch:
            current += 1
        labels[nxt] = current
    return labels


class SamExtractor:
    name = "sam"
    # Missing stickers make the observation weaker; more than this is not trusted.
    max_missing = 3
    min_color_margin = 0.5

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None, write_overlay: bool = True):
        self.model_id = model_id
        self.device = device
        self.write_overlay = write_overlay
        self._generator = None

    @property
    def cache_variant(self) -> str:
        """What, besides the code, decides this extractor's readings (see storage/evaluation_cache)."""
        return "sam-" + self.model_id.replace("/", "_") + ("" if self.write_overlay else "-nooverlay")

    def _masks(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        if self._generator is None:
            import torch
            from transformers import pipeline

            device = self.device or ("mps" if torch.backends.mps.is_available() else "cpu")
            self._generator = pipeline("mask-generation", model=self.model_id, device=device)
        from PIL import Image

        result = self._generator(Image.fromarray(image_bgr[:, :, ::-1]), points_per_batch=64, pred_iou_thresh=0.86)
        h, w = image_bgr.shape[:2]
        return [cv2.resize(np.asarray(m, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for m in result["masks"]]

    def extract(self, image_path: Path) -> Extraction:
        image = cv2.imread(str(image_path))
        if image is None:
            return Extraction(
                self.name, None, flags=[Flag.VISION_LOW_CONFIDENCE], details={"error": "unreadable image"},
                quality={"cube": INCONCLUSIVE},
            )
        stickers = _main_cluster(_dedupe([s for m in self._masks(image) if (s := _sticker(m, image)) is not None]))
        details: dict[str, Any] = {"model": self.model_id, "stickers_found": len(stickers)}
        if len(stickers) < 9:
            details["error"] = "too few sticker-like regions"
            return Extraction(
                self.name, None, flags=[Flag.VISION_LOW_CONFIDENCE], details=details, quality={"cube": INCONCLUSIVE}
            )

        # Global edge directions, averaged per axis over all stickers.
        axes: dict[str, list[np.ndarray]] = {"a": [], "b": [], "c": []}
        faces_of: list[str | None] = []
        for s in stickers:
            names = [_axis_name(d) for d in s.directions]
            face = FACE_OF.get(frozenset(names)) if len(set(names)) == 2 else None
            faces_of.append(face)
            if face:
                for d, n in zip(s.directions, names):
                    axes[n].append(_canonical(d, n))
        if any(not v for v in axes.values()):
            details["error"] = "stickers do not show three face orientations"
            return Extraction(
                self.name, None, flags=[Flag.VISION_LOW_CONFIDENCE], details=details,
                quality={"cube": PASS, "shape": INCONCLUSIVE},
            )
        axis = {n: (lambda m: m / np.linalg.norm(m))(np.mean(v, axis=0)) for n, v in axes.items()}

        flags: list[Flag] = []
        grids: dict[str, list[str | None]] = {}
        placed: list[tuple[Sticker, str, int]] = []
        bins_per_face: dict[str, tuple[int, int]] = {}
        for face, (col_axis, sign, row_axis) in GRID_AXES.items():
            members = [s for s, f in zip(stickers, faces_of) if f == face]
            cells: list[str | None] = [None] * 9
            if members:
                pitch = float(np.median([s.size for s in members]))
                # The face's axes are oblique in the image, so express each centre in
                # that basis instead of projecting (projection would mix rows and columns).
                basis = np.column_stack([sign * axis[col_axis], axis[row_axis]])
                coords = np.array([np.linalg.solve(basis, s.centre) for s in members])
                cols = _bins(coords[:, 0], pitch)
                rows = _bins(coords[:, 1], pitch)
                bins_per_face[face] = (int(rows.max()) + 1, int(cols.max()) + 1)
                if rows.max() < 3 and cols.max() < 3:
                    for s, r, c in zip(members, rows, cols):
                        # A face missing its whole first row/column would shift indices;
                        # that is caught below by the requirement that every face spans 3x3.
                        cells[3 * r + c] = classify(s.rgb).label
                        placed.append((s, face, 3 * r + c))
            grids[face] = cells
        details["bins_per_face"] = bins_per_face
        quality = {"cube": PASS, "shape": PASS, "grid": PASS, "stickers": PASS, "single_color": PASS, "shades": PASS,
                   "colors": PASS}
        # Same rule as the vision extractor: letters on most non-center stickers.
        marked = sum(s.marking > 0.04 for s, _, index in placed if index != 4)
        details["marked_stickers"] = marked
        outlines = {
            face: cv2.convexHull(np.concatenate([s.quad for s, f in zip(stickers, faces_of) if f == face])).reshape(-1, 2)
            for face in bins_per_face
        }
        details["geometry"] = geometry(
            (image.shape[1], image.shape[0]), outlines,
            {f"{face}{index + 1}": s.quad for s, face, index in placed},
            {face: (cols, rows) for face, (rows, cols) in bins_per_face.items()},
            [f"{face}{index + 1}" for s, face, index in placed if index != 4 and s.marking > 0.04],
            split := [f"{face}{index + 1}" for s, face, index in placed if index != 4 and s.second_color > 0.10],
            shades := [f"{face}{members[k][0] + 1}" for face in GRID_AXES
                       if (members := [(index, classify(s.rgb)) for s, f, index in placed if f == face])
                       for k in off_shade([c for _, c in members], self.min_color_margin)],
        )
        if shades:  # same rule as the vision extractor
            flags.append(Flag.OFF_SHADE_STICKERS)
            quality["shades"] = FAIL
        if split:  # same rule as the vision extractor
            flags.append(Flag.SPLIT_STICKERS)
            quality["single_color"] = FAIL
        if marked >= 16:
            flags.append(Flag.MARKED_STICKERS)
            quality["stickers"] = FAIL
        spans = list(bins_per_face.values())
        if sum(1 for r, c in spans if r > 3 or c > 3) >= 2:
            flags.append(Flag.WRONG_GRID_SIZE)
            quality["grid"] = FAIL
        elif len(spans) < 3 or any(span != (3, 3) for span in spans):
            flags.append(Flag.VISION_LOW_CONFIDENCE)
            quality["grid"] = INCONCLUSIVE

        missing = sum(c is None for g in grids.values() for c in g)
        details["missing_stickers"] = missing
        if missing > self.max_missing:
            flags.append(Flag.VISION_LOW_CONFIDENCE)
            quality["grid"] = quality["grid"] if quality["grid"] == FAIL else INCONCLUSIVE
        margins = [classify(s.rgb).margin for s, _, _ in placed]
        min_margin = min(margins) if margins else 0.0
        if min_margin < self.min_color_margin:
            flags.append(Flag.VISION_LOW_CONFIDENCE)
            quality["colors"] = INCONCLUSIVE

        if self.write_overlay:
            overlay_path = image_path.with_suffix(".sam.jpg")
            cv2.imwrite(str(overlay_path), _overlay(image, placed))
            details["overlay"] = overlay_path.name
        observation = Observation.from_faces({f: [c or "." for c in g] for f, g in grids.items()})
        return Extraction(self.name, observation, min_margin, sorted(set(flags)), details, quality)


def _overlay(image: np.ndarray, placed: list[tuple[Sticker, str, int]]) -> np.ndarray:
    out = image.copy()
    colors = {"U": (0, 255, 0), "F": (255, 128, 0), "R": (255, 0, 255)}
    scale = max(image.shape[:2]) / 1024
    for s, face, index in placed:
        cv2.polylines(out, [s.quad.astype(np.int32)], True, colors[face], 2)
        text = f"{face}{index + 1}:{classify(s.rgb).label}"
        org = (int(s.centre[0] - 28 * scale), int(s.centre[1] + 6 * scale))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (0, 0, 0), int(4 * scale))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (255, 255, 255), max(1, int(1.5 * scale)))
    return out
