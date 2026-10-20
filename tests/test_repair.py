import random

import pytest

from rubiksbench.cube.facelets import facelet_index
from rubiksbench.cube.observation import Observation
from rubiksbench.cube.repair import CHANCE_DISTANCE, chance_distance, repair
from rubiksbench.evaluation.evaluator import validate
from rubiksbench.evaluation.validity import HALF_LIFE, validity
from rubiksbench.vision.extraction import Extraction

from .test_validator import REAL_CUBE, corner_view, scrambled


def repaired(observation: Observation) -> Observation:
    return observation.with_stickers({facelet_index(f["facelet"]): f["color"] for f in
                                      (fix.to_dict() for fix in repair(observation).fixes)})


def test_a_valid_cube_needs_no_repair():
    result = repair(Observation.from_faces(REAL_CUBE))
    assert result.distance == 0 and result.fixes == ()


def test_a_duplicate_center_is_one_sticker_to_fix():
    observation = Observation.from_faces(dict(REAL_CUBE, R="ORG WGG OYB"))  # R's center copies U's green
    result = repair(observation)
    assert result.distance == 1 and [f.to_dict()["facelet"] for f in result.fixes] in (["U5"], ["R5"])


@pytest.mark.parametrize("seed", range(12))
def test_repair_agrees_with_the_validator(seed):
    # Recolor a few random stickers of a valid corner view: the distance is 0 exactly when
    # the validator still finds a standard cube, never more than the stickers changed, and
    # applying the fixes always gives a valid standard cube.
    rng = random.Random(seed)
    observation = corner_view(scrambled(seed))
    shown = [i for i, s in enumerate(observation.stickers) if s is not None]
    changed = rng.sample(shown, rng.randint(1, 4))
    observation = observation.with_stickers({i: rng.choice("WYROGB") for i in changed})
    distance = repair(observation).distance
    assert (distance == 0) == validate(observation, with_scramble=False).standard.valid
    assert distance <= len(changed)
    assert validate(repaired(observation), with_scramble=False).standard.valid


def test_chance_distance_matches_its_constant():
    assert chance_distance(40, seed=1) == pytest.approx(CHANCE_DISTANCE, abs=0.75)


QUALITY = {"cube": "pass", "shape": "pass", "grid": "pass", "in_place": "pass", "stickers": "pass", "single_color": "pass", "shades": "pass", "colors": "pass"}


def scored(faces, quality=QUALITY, details=None):
    from rubiksbench.evaluation.rubric import grade

    observation = Observation.from_faces(faces)
    extraction = Extraction("annotation", observation, details=details or {}, quality=quality)
    checks = grade(extraction, validate(observation, with_scramble=False).to_dict())["checks"]
    return validity(extraction, checks)


def test_validity_is_full_for_a_valid_cube_and_drops_per_sticker():
    assert scored(REAL_CUBE)["score"] == 1.0
    one_off = scored(dict(REAL_CUBE, R="ORG WGG OYB"))
    assert one_off["repairs"] == 1 and one_off["score"] == pytest.approx(0.5 ** (1 / HALF_LIFE))
    # Every half-life of stickers to fix halves the score; random colors land at 25%.
    assert HALF_LIFE == pytest.approx(CHANCE_DISTANCE / 2)


def test_sticker_defects_count_as_stickers_to_fix():
    # A second shade of a color takes someone who has held a cube to notice.
    quality = QUALITY | {"shades": "fail"}
    result = scored(REAL_CUBE, quality, {"geometry": {"off_shade": ["F2", "U8"]}})
    assert result["repairs"] == 0 and result["defects"] == 2
    assert result["score"] == pytest.approx(0.5 ** (2 / HALF_LIFE))


def test_a_flaw_seen_at_a_glance_scores_zero_but_keeps_its_stickers_to_fix():
    # Two-colored stickers are plain to anyone at a glance: the image fools no one.
    result = scored(REAL_CUBE, QUALITY | {"single_color": "fail"}, {"geometry": {"split": ["F2", "U8"]}})
    assert result["score"] == 0.0 and "fools no one" in result["reason"]
    assert result["defects"] == 2 and result["defective"] == ["F2", "U8"]


def test_an_unreadable_or_solved_cube_scores_zero():
    assert scored(REAL_CUBE, QUALITY | {"grid": "fail"}) == {"score": 0.0, "reason": "not read as a 3x3 cube"}
    solved = {"U": "G" * 9, "F": "R" * 9, "R": "W" * 9}
    assert scored(solved)["score"] == 0.0


def test_a_designed_pattern_is_valid_but_held_in_slightly_less_regard():
    import magiccube

    from rubiksbench.evaluation.rubric import FOOLS, grade

    # The checkerboard pattern: a valid cube, but every visible face is symmetric.
    cube = magiccube.Cube(3)
    cube.rotate("R2 L2 U2 D2 F2 B2")
    observation = Observation.from_facelet_string(cube.get_kociemba_facelet_colors()).restricted_to("UFR")
    extraction = Extraction("annotation", observation, quality=QUALITY)
    rubric = grade(extraction, validate(observation, with_scramble=False).to_dict())
    result = validity(extraction, rubric["checks"])
    assert rubric["checks"]["scrambled"] == "fail" and result["repairs"] == 0
    assert 0.95 <= result["score"] < 1.0 and result["score"] == result["scramble_factor"]
    assert FOOLS[rubric["fools"]] == "everyone"  # it is a valid cube: nothing to see through
