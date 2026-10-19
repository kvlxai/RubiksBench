"""Fit an ideal 3x3x3 cube, seen through a pinhole camera, to the image (analysis by synthesis).

Instead of reading faces one by one from the silhouette, an ideal cube of 27 cubies
is posed in 3D: rotation, position and focal length, seven numbers in all. The
silhouette's hexagon gives the starting pose (``cv2.solvePnP``); the pose is then
refined so the cube's predicted seams land where the image has them, with black
plastic and color edges on the seams and plain sticker color inside each cell.
Because all three faces come from one rigid cube, they are fitted together: shared
edges, right angles and equal cubies hold by construction, and a thin or blurred face
borrows its precision from the others.

The fit is robust: the few worst-fitting cells are ignored while fitting, so a single
deformed cubie cannot pull the cube off. After fitting, each cell's residual (how much
of it is not one sticker color) shows such cubies.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize

# World coordinates in cubies: the origin is the bottom corner nearest the camera (B), a runs
# towards the lower left corner (LL), b towards the lower right (LR), z up.
CORNERS = {"T": (3, 3, 3), "UR": (0, 3, 3), "LR": (0, 3, 0), "B": (0, 0, 0), "LL": (3, 0, 0), "UL": (3, 0, 3),
           "C": (0, 0, 3)}
_HEXAGON = ("T", "UR", "LR", "B", "LL", "UL")


def face_point(face: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """A point on a visible face, from face coordinates (u: column, v: row, 0-3 from the face's
    top left as read, i.e. facelet 1) to world coordinates."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    zero, three = np.zeros_like(u), np.full_like(u, 3)
    return {
        "U": lambda: np.stack([3 - u, 3 - v, three], -1),
        "F": lambda: np.stack([3 - u, zero, 3 - v], -1),
        "R": lambda: np.stack([zero, u, 3 - v], -1),
    }[face]()


# Points sampled over each face: their distance to the nearest seam and the cell they fall in.
_STEP = 0.05
_GRID = np.arange(_STEP / 2, 3, _STEP)
_U, _V = (a.ravel() for a in np.meshgrid(_GRID, _GRID))
_SEAM = np.minimum(np.abs(_U - np.round(_U)), np.abs(_V - np.round(_V)))
_CELL = np.floor(_V).astype(int) * 3 + np.floor(_U).astype(int)
_WORLD = {f: face_point(f, _U, _V).astype(np.float32) for f in "UFR"}
# Cells this far from the seams count as the sticker's inside (in cubies, beyond the gap).
_INSIDE = 0.1
# Cells left out of the fit: the worst few may be deformed cubies.
_TRIM = 3


@dataclass
class FittedCube:
    rotation: np.ndarray  # Rodrigues vector
    translation: np.ndarray
    camera: np.ndarray  # 3x3 intrinsics
    gap: float  # half the seam width, in cubies
    score: float  # seam contrast; higher is a better fit
    start_score: float  # the same for the starting pose

    def project(self, world: np.ndarray) -> np.ndarray:
        return project_points(world, self.rotation, self.translation, self.camera)

    def quads(self) -> dict[str, np.ndarray]:
        """Each visible face's corners in the image, in facelet order 1, 3, 9, 7."""
        u, v = np.array([0, 3, 3, 0]), np.array([0, 0, 3, 3])
        return {f: self.project(face_point(f, u, v)).astype(np.float32) for f in "UFR"}

    def corner(self, name: str) -> np.ndarray:
        return self.project(np.array([CORNERS[name]], np.float32))[0]


def project_points(world: np.ndarray, rotation: np.ndarray, translation: np.ndarray, camera: np.ndarray) -> np.ndarray:
    """Pinhole projection of world points to pixels. cv2.projectPoints gives the same, but also
    computes its Jacobian, which made it most of the fitting time."""
    r, _ = cv2.Rodrigues(np.asarray(rotation, np.float64))
    p = np.asarray(world, np.float64) @ r.T + np.asarray(translation, np.float64).ravel()
    return np.stack([camera[0, 0] * p[:, 0] / p[:, 2] + camera[0, 2],
                     camera[1, 1] * p[:, 1] / p[:, 2] + camera[1, 2]], 1)


def _camera(log_focal: float, shape: tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    f = float(np.exp(log_focal)) * w
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    x = points[:, 0].astype(np.float32).reshape(1, -1)
    y = points[:, 1].astype(np.float32).reshape(1, -1)
    return cv2.remap(image, x, y, cv2.INTER_LINEAR, borderValue=0).ravel()


def _dark(lab: np.ndarray) -> np.ndarray:
    """Black plastic: dark and colorless."""
    return ((lab[..., 0] < 25) & (np.hypot(lab[..., 1], lab[..., 2]) < 20)).astype(np.float32)


def _edges(lab: np.ndarray) -> np.ndarray:
    """Color edge strength, scaled so its 99th percentile is 1."""
    g = np.sqrt(sum(cv2.Sobel(lab[..., k], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lab[..., k], cv2.CV_32F, 0, 1) ** 2
                    for k in range(3)))
    return np.clip(g / (np.percentile(g, 99) + 1e-6), 0, 1).astype(np.float32)


def _contrast(maps: list[np.ndarray], project, gap: float) -> float:
    """For each map: its mean on the predicted seams minus its trimmed mean inside the cells."""
    total = 0.0
    points = {f: project(_WORLD[f]) for f in "UFR"}
    for m in maps:
        on_seam, inside = [], []
        for f in "UFR":
            values = _sample(m, points[f])
            on_seam.append(values[_SEAM < gap])
            inner = _SEAM > gap + _INSIDE
            inside.append(np.bincount(_CELL[inner], weights=values[inner], minlength=9)
                          / np.maximum(np.bincount(_CELL[inner], minlength=9), 1))
        cells = np.sort(np.concatenate(inside))[: 27 - _TRIM]
        total += float(np.concatenate(on_seam).mean() - cells.mean())
    return total


def _start(image_shape: tuple[int, ...], outline: np.ndarray) -> np.ndarray | None:
    """Rotation, translation and log focal length best matching the outline's seven corners."""
    world = np.array([CORNERS[n] for n in (*_HEXAGON, "C")], np.float64)
    best = None
    for log_focal in np.linspace(np.log(0.8), np.log(8), 25):
        camera = _camera(log_focal, image_shape)
        ok, rvec, tvec = cv2.solvePnP(world, np.asarray(outline, np.float64), camera, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        err = float(np.linalg.norm(project_points(world, rvec, tvec, camera) - outline, axis=1).mean())
        if best is None or err < best[0]:
            best = (err, np.r_[rvec.ravel(), tvec.ravel(), log_focal])
    return None if best is None else best[1]


def fit_cube(image: np.ndarray, hexagon: np.ndarray, inner: np.ndarray) -> FittedCube | None:
    """Pose the ideal cube to match the image, starting from the silhouette's hexagon
    (clockwise from the top: T, UR, LR, B, LL, UL) and its inner vertex C."""
    outline = np.vstack([hexagon, inner[None]]).astype(np.float32)
    p0 = _start(image.shape, outline)
    if p0 is None:
        return None
    lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    maps = [_dark(lab), _edges(lab)]
    cubie = float(np.linalg.norm(np.ptp(outline, axis=0))) / 4.5
    # Search in scaled steps around the start: rotation 0.05 rad, position 2% of the distance,
    # log focal 0.1, half seam width 0.03 cubie (around 0.08).
    scale = np.r_[[0.05] * 3, [abs(p0[3:6]).max() * 0.02] * 3, 0.1, 0.03]
    base = np.r_[p0, 0.08]
    world_outline = np.array([CORNERS[n] for n in (*_HEXAGON, "C")], np.float32)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, float]:
        q = base + x * scale
        return q[:7], float(np.clip(q[7], 0.03, 0.2))

    def projector(pose: np.ndarray):
        camera = _camera(pose[6], image.shape)
        return lambda world: project_points(world, pose[:3], pose[3:6], camera)

    def objective(x: np.ndarray, blurred: list[np.ndarray]) -> float:
        pose, gap = unpack(x)
        project = projector(pose)
        # The outline's corners are known to within about a quarter cubie: stay near them.
        drift = np.maximum(0, np.linalg.norm(project(world_outline) - outline, axis=1) / cubie - 0.25)
        return -_contrast(blurred, project, gap) + 20.0 * float(np.mean(drift**2))

    x = np.zeros(8)
    start_score = _contrast(maps, projector(unpack(x)[0]), unpack(x)[1])
    for sigma in (cubie / 6, cubie / 15, cubie / 40):
        blurred = [cv2.GaussianBlur(m, (0, 0), sigma) for m in maps]
        x = minimize(objective, x, args=(blurred,), method="Nelder-Mead",
                     options={"initial_simplex": np.vstack([x, x + np.eye(8)]), "xatol": 0.01, "fatol": 1e-5,
                              "maxfev": 2500}).x
    pose, gap = unpack(x)
    return FittedCube(pose[:3], pose[3:6], _camera(pose[6], image.shape), gap,
                      _contrast(maps, projector(pose), gap), start_score)


def cell_residuals(image: np.ndarray, cube: FittedCube) -> dict[str, tuple[float, float]]:
    """Per sticker cell ("U1" ...): the share of its inside that is not sticker color (black
    plastic or background), and the share in a second sticker color."""
    lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    out = {}
    inner = _SEAM > cube.gap + _INSIDE
    for f in "UFR":
        points = cube.project(_WORLD[f])
        pixels = np.stack([_sample(lab[..., k], points) for k in range(3)], 1)
        for cell in range(9):
            px = pixels[inner & (_CELL == cell)]
            off = np.linalg.norm(px - np.median(px, axis=0), axis=1) > 25
            chroma = np.hypot(px[:, 1], px[:, 2])
            out[f"{f}{cell + 1}"] = (float((off & (chroma < 20)).mean()), float((off & (chroma >= 35)).mean()))
    return out


# A cell this full of non-sticker pixels holds a displaced or deformed cubie (real photos: at
# most 0.04; a cubie sunk into the cube: 0.31) ...
OUT_OF_PLACE = 0.25
# ... unless other cells in its row or column show it too, which means the fit is off along
# an edge rather than one cubie being out of place.
ALONG_AN_EDGE = 0.10


def out_of_place(residuals: dict[str, tuple[float, float]]) -> list[str]:
    """Cells ("R6") whose cubie is displaced or deformed: isolated, clearly non-sticker cells.
    Centers are left out (they may carry a logo)."""
    found = []
    for name, (plastic, _) in residuals.items():
        face, index = name[0], int(name[1:]) - 1
        if index == 4 or plastic < OUT_OF_PLACE:
            continue
        row, column = index // 3, index % 3
        neighbours = [f"{face}{3 * row + c + 1}" for c in range(3) if c != column]
        neighbours += [f"{face}{3 * r + column + 1}" for r in range(3) if r != row]
        if sum(residuals[n][0] >= ALONG_AN_EDGE for n in neighbours) <= 1:
            found.append(name)
    return found
