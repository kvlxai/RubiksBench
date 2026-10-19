"""Automatic extraction against hand labels of real benchmark images (tests/fixtures).

The first run downloads rembg's U2-Net model (~176 MB) into ~/.u2net.
"""

from pathlib import Path

import pytest

from rubiksbench.cube.observation import Observation
from rubiksbench.evaluation.evaluator import evaluate_extraction
from rubiksbench.evaluation.taxonomy import Flag
from rubiksbench.vision.color_classifier import classify
from rubiksbench.vision.extractor import VisionExtractor

FIXTURES = Path(__file__).parent / "fixtures"

# Hand labels: (faces, expected verdict). Top face = U, front-left = F, front-right = R.
LABELLED = {
    "nano-banana-2-1__0000": ({"U": "GYO RYB BYR", "F": "GRY YBO OWG", "R": "RGB GRW YBY"}, "INVALID"),
    "nano-banana-2-gemini-3-1-flash-image__0000": (
        {"U": "WOY GRY BBO", "F": "RYB GOW YBR", "R": "WGO RYG BOW"},
        "INVALID",
    ),
    "qwen-image-3-0__0000": ({"U": "YWO YWY WBG", "F": "YRB OGB BYO", "R": "YBO RRR BOG"}, "INVALID"),
    "qwen-image-3-0-pro__0000": ({"U": "BRO YWG YRB", "F": "WWY YRG OBG", "R": "WWG YBR ORO"}, "INVALID"),
}


@pytest.fixture(scope="module")
def extractor():
    return VisionExtractor(write_overlay=False)


@pytest.mark.parametrize("name", sorted(LABELLED))
def test_reads_every_sticker_like_the_hand_label(extractor, name):
    faces, verdict = LABELLED[name]
    extraction = extractor.extract(FIXTURES / f"{name}.jpg")
    assert extraction.flags == []
    assert extraction.observation == Observation.from_faces(faces)
    assert evaluate_extraction(extraction, with_scramble=False)["verdict"] == verdict


@pytest.mark.parametrize(
    "name, grids",
    [
        # A 4x4-ish mosaic cube: no face is 3x3, though the exact counts are noisy (None).
        ("flux-3-image-klein__0000", {"U": None, "F": None, "R": None}),
        # A box: 3 wide, 4 tall on both side faces.
        ("flux-3-image-klein__0001", {"U": (3, 3), "F": (3, 4), "R": (3, 4)}),
        # Only the front face is wrong: 4 columns x 3 rows (12 fields).
        ("qwen-image-3-0-pro__0001", {"U": (3, 3), "F": (4, 3), "R": (3, 3)}),
    ],
)
def test_faces_that_are_not_3x3_are_wrong_grid_size(extractor, name, grids):
    extraction = extractor.extract(FIXTURES / f"{name}.jpg")
    measured = {f: (d["columns"], d["rows"]) for f, d in extraction.details["faces"].items()}
    assert {f: g if grids[f] else g != (3, 3) for f, g in measured.items()} == {f: g or True for f, g in grids.items()}
    assert Flag.WRONG_GRID_SIZE in extraction.flags
    record = evaluate_extraction(extraction)
    assert record["verdict"] == "INVALID"
    # Each face that is not 3x3 is outlined as a whole, unless an earlier check blocked the grid check.
    outlined = [f for m in record["rubric"].get("marks", []) if m["check"] == "grid_3x3" for f in m["faces"]]
    failed = record["rubric"]["checks"]["grid_3x3"] == "fail"
    assert outlined == ([f for f, g in grids.items() if g != (3, 3)] if failed else [])


def test_sticker_outlines_follow_the_facelet_order(extractor):
    extraction = extractor.extract(FIXTURES / "qwen-image-3-0__0000.jpg")
    geometry = extraction.details["geometry"]
    assert geometry["size"] == [768, 768] and len(geometry["stickers"]) == 27

    def centre(name):
        xs, ys = zip(*geometry["stickers"][name])
        return sum(xs) / 4, sum(ys) / 4

    # U1 is the back corner of the top face, U9 the front one; F7 sits below F1, R3 right of R1.
    assert centre("U1")[1] < centre("U9")[1]
    assert centre("F1")[1] < centre("F7")[1] and centre("R1")[0] < centre("R3")[0]


def test_image_without_a_cube_is_inconclusive(extractor, tmp_path):
    import cv2
    import numpy as np

    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((256, 256, 3), 40, np.uint8))
    assert evaluate_extraction(extractor.extract(blank))["verdict"] == "INCONCLUSIVE"


@pytest.mark.parametrize(
    "rgb, label",
    [((235, 235, 235), "W"), ((150, 150, 150), "W"), ((240, 200, 20), "Y"), ((200, 30, 30), "R"),
     ((250, 120, 20), "O"), ((20, 160, 70), "G"), ((20, 80, 200), "B"), ((150, 40, 180), "X")],
)
def test_color_names(rgb, label):
    assert classify(rgb).label == label


class FakeClassifier:
    def __init__(self, scores):
        self.scores = scores

    def __call__(self, image, candidate_labels, hypothesis_template):
        ranked = sorted(self.scores.items(), key=lambda kv: -kv[1])
        return [{"label": label, "score": score} for label, score in ranked]


@pytest.mark.parametrize(
    "scores, result",
    [
        ({"a Rubik's cube": 0.9, "an animal": 0.1}, "pass"),
        ({"a Rubik's cube": 0.001, "an animal": 0.999}, "fail"),
        ({"a Rubik's cube": 0.2, "a cube-shaped box": 0.8}, "inconclusive"),
    ],
)
def test_recognizer_thresholds(scores, result, tmp_path):
    import cv2
    import numpy as np

    from rubiksbench.vision.recognizer import CubeRecognizer

    image = tmp_path / "x.png"
    cv2.imwrite(str(image), np.zeros((8, 8, 3), np.uint8))
    recognizer = CubeRecognizer()
    recognizer._classifier = FakeClassifier(scores)
    assert recognizer.recognize(image).result == result


def test_image_recognized_as_not_a_cube_scores_zero(tmp_path):
    from rubiksbench.vision.recognizer import CubeRecognizer

    extractor = VisionExtractor(write_overlay=False, recognize=True)
    extractor.recognizer = CubeRecognizer()
    extractor.recognizer._classifier = FakeClassifier({"a Rubik's cube": 0.0003, "an animal": 0.9997})
    record = evaluate_extraction(extractor.extract(FIXTURES / "qwen-image-3-0__0000.jpg"))
    assert record["verdict"] == "INVALID" and record["flags"] == ["not_a_cube"]
    assert record["rubric"]["points"] == 0 and record["rubric"]["checks"]["cube_detected"] == "fail"


def test_clean_cube_silhouette_fills_its_hexagon(extractor):
    extraction = extractor.extract(FIXTURES / "qwen-image-3-0__0000.jpg")
    assert extraction.details["layout"]["hexagon_fit"] >= VisionExtractor.min_hexagon_fit


@pytest.mark.parametrize("name", ["nano-banana-2-lettered__0001", "nano-banana-2-lettered__0002"])
def test_letters_on_stickers_are_3x3_but_invalid(extractor, name):
    # Nano Banana 2 printed each sticker's color initial on it: still a 3x3 grid
    # (the letters must not be read as extra cells), but not a plain Rubik's cube.
    extraction = extractor.extract(FIXTURES / f"{name}.jpg")
    assert all((d["columns"], d["rows"]) == (3, 3) for d in extraction.details["faces"].values())
    assert Flag.MARKED_STICKERS in extraction.flags and extraction.details["marked_stickers"] >= 16
    record = evaluate_extraction(extraction, with_scramble=False)
    assert record["verdict"] == "INVALID" and "marked_stickers" in record["flags"]
    assert record["rubric"]["checks"]["plain_stickers"] == "fail"
    lettered = next(m for m in record["rubric"]["marks"] if m["check"] == "plain_stickers")
    assert len(lettered["facelets"]) >= 16 and "U5" not in lettered["facelets"]


@pytest.mark.parametrize("name", sorted(LABELLED))
def test_plain_stickers_are_not_marked(extractor, name):
    extraction = extractor.extract(FIXTURES / f"{name}.jpg")
    assert extraction.quality["stickers"] == "pass" and extraction.details["marked_stickers"] < 16


def test_two_colored_stickers_make_the_image_invalid(extractor):
    # Qwen-Image-3.0 drew three stickers half one color, half another (blue/yellow, orange/blue).
    extraction = extractor.extract(FIXTURES / "qwen-image-3-0-split__0002.jpg")
    assert Flag.SPLIT_STICKERS in extraction.flags
    assert sorted(extraction.details["geometry"]["split"]) == ["F2", "F8", "U8"]
    record = evaluate_extraction(extraction, with_scramble=False)
    assert record["verdict"] == "INVALID" and "split_stickers" in record["flags"]
    assert record["rubric"]["checks"]["single_color"] == "fail"
    assert sorted(m["facelets"][0] for m in record["rubric"]["marks"] if m["check"] == "single_color") == [
        "F2", "F8", "U8"]


@pytest.mark.parametrize("name", sorted(LABELLED))
def test_plain_stickers_are_single_colored(extractor, name):
    extraction = extractor.extract(FIXTURES / f"{name}.jpg")
    assert extraction.details["geometry"]["split"] == [] and extraction.quality["single_color"] == "pass"


def test_a_blurred_face_is_read_like_its_sides(extractor):
    # Ideogram 4.5 #9: a shallow camera angle and lens blur leave the top face a thin, blurred
    # sliver whose rows cannot be counted. Its sides are 3x3, so it is read as 3x3 too, and the
    # fitted cube places its stickers precisely enough to read all nine.
    extraction = extractor.extract(FIXTURES / "ideogram-4-5-edge-on__0009.jpg")
    assert "inferred_grid" in extraction.details["faces"]["U"]
    assert Flag.WRONG_GRID_SIZE not in extraction.flags and extraction.quality["grid"] == "pass"
    assert extraction.observation.to_faces() == {"U": "WRWGWGYRW", "R": "GWRYBYRBG", "F": "WYRRWYBRW"}


def test_a_displaced_cubie_is_out_of_place_not_two_colored(extractor):
    # Nano Banana 2.1 #5: one cubie on the right face is sunk into the cube and its neighbour
    # sticks out, which no turn can do. The fitted cube finds the sunk one (R6) as a lone cell
    # of plastic and background; the protruding red is read as red, not as a split sticker.
    extraction = extractor.extract(FIXTURES / "nano-banana-2-1-displaced__0005.jpg")
    assert extraction.details["model"]["used"]
    assert extraction.details["geometry"]["out_of_place"] == ["R6"]
    assert Flag.CUBIES_OUT_OF_PLACE in extraction.flags and Flag.SPLIT_STICKERS not in extraction.flags
    assert extraction.observation.to_faces()["R"][8] == "R"
    record = evaluate_extraction(extraction, with_scramble=False)
    assert record["verdict"] == "INVALID" and record["rubric"]["checks"]["cubies_in_place"] == "fail"


@pytest.mark.parametrize("name", sorted(LABELLED))
def test_well_formed_cubes_have_every_cubie_in_place(extractor, name):
    extraction = extractor.extract(FIXTURES / f"{name}.jpg")
    assert extraction.details["geometry"]["out_of_place"] == []


def test_only_a_lone_misfit_cell_is_out_of_place():
    from rubiksbench.vision.cube_model import out_of_place

    clean = {f"{f}{i}": (0.0, 0.0) for f in "UFR" for i in range(1, 10)}
    assert out_of_place(clean | {"R6": (0.31, 0.0)}) == ["R6"]
    # A whole column off means the fit is off along an edge, not three displaced cubies.
    assert out_of_place(clean | {"R3": (0.25, 0.0), "R6": (0.25, 0.0), "R9": (0.25, 0.0)}) == []
    assert out_of_place(clean | {"U5": (0.4, 0.0)}) == []  # centers may carry a logo


def test_projection_matches_opencv():
    import cv2
    import numpy as np

    from rubiksbench.vision.cube_model import project_points

    world = np.random.default_rng(0).uniform(-1.5, 1.5, (50, 3)).astype(np.float32)
    rotation, translation = np.array([0.4, -0.7, 0.2]), np.array([0.1, -0.2, 9.0])
    camera = np.array([[1800, 0, 512], [0, 1800, 512], [0, 0, 1]], np.float64)
    expected = cv2.projectPoints(world, rotation, translation, camera, None)[0].reshape(-1, 2)
    assert np.allclose(project_points(world, rotation, translation, camera), expected, atol=1e-3)
