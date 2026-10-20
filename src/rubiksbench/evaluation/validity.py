"""Validity: how close an image comes to a valid, scrambled, standard 3x3 cube, from 0% to 100%.

    validity = 0.5 ** ((repairs + defects) / half-life) * scramble factor,   half-life = chance / 2

``repairs`` is the fewest observed stickers to recolor for a valid standard cube
(``cube.repair``), ``defects`` the stickers no recoloring fixes (letters, two
colors on one sticker, a second shade of a color, a cubie out of place) and
``chance`` the repair distance of random colors. Every sticker to fix costs the
same share of what is left, so 100% is a valid cube, coloring the stickers at
random lands at 25%, and a cube that still fools someone never reaches 0%.
The scramble factor holds a barely scrambled cube or a designed pattern in slightly
less regard: from 100% for a random-looking scramble down to 95%.

0% is kept for images that fool no one: not read as a 3x3 cube, not scrambled at all
(every face one color), or with a flaw anyone sees at a glance (letters, a seventh color).

The rubric explains what is wrong; this measures how much.
"""

from typing import Any

from ..cube.repair import CHANCE_DISTANCE, repair
from ..vision.extraction import Extraction
from .rubric import FAIL, PASS, READ_AS_CUBE, fools, scramble

# How much a barely scrambled cube (scramble score 0) loses against a random-looking one.
SCRAMBLE_WEIGHT = 0.05
# Stickers to fix that halve the score: random colors (chance) then score 25%.
HALF_LIFE = CHANCE_DISTANCE / 2


def validity(extraction: Extraction, checks: dict[str, str]) -> dict[str, Any]:
    """Score one image from its extraction and its rubric results."""
    if not all(checks.get(c) == PASS for c in READ_AS_CUBE) or extraction.observation is None:
        return {"score": 0.0, "reason": "not read as a 3x3 cube"}
    mixed = scramble(extraction.observation)
    if mixed["of"] and not mixed["off_center"]:
        return {"score": 0.0, "reason": "not scrambled"}
    factor = 1 - SCRAMBLE_WEIGHT * (1 - mixed["score"])
    geometry = extraction.details.get("geometry", {})
    defects = sorted(
        set(geometry.get("marked", []) if checks.get("plain_stickers") == FAIL else [])
        | set(geometry.get("split", []) if checks.get("single_color") == FAIL else [])
        | set(geometry.get("off_shade", []) if checks.get("one_shade") == FAIL else [])
        | set(geometry.get("out_of_place", []) if checks.get("cubies_in_place") == FAIL else [])
    )
    nearest = repair(extraction.observation)
    cost = nearest.distance + len(defects)
    out: dict[str, Any] = {
        "score": 0.5 ** (cost / HALF_LIFE) * factor,
        "repairs": nearest.distance,
        "defects": len(defects),
        "chance": CHANCE_DISTANCE,
        "scramble": mixed["score"],
        "scramble_factor": round(factor, 3),
        "fixes": [f.to_dict() for f in nearest.fixes],
        "defective": defects,
    }
    if fools(checks, extraction.observation) == 0:
        # The stickers to fix stay listed as evidence; the image itself fools no one.
        out |= {"score": 0.0, "reason": "fools no one: a flaw anyone sees at a glance"}
    return out

