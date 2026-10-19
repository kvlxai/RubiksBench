"""Timestamped result files for a scored run.

    results-<UTC timestamp>.json   everything needed to recompute the report
    results-<UTC timestamp>.html   self-contained report: tables, inline SVG, sample gallery

The HTML follows the dataviz guidance: colors as light/dark tokens, one
sequential hue for magnitudes, reserved status colors for check results, hover
titles on every mark, and tables as the accessible view of every chart.
"""

import json
import re
from urllib.parse import urlencode
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from .. import EVALUATOR_VERSION, __version__

REPOSITORY = "https://github.com/kvlxai/RubiksBench"
from .rubric import CHECKS, FOOLS, MAX_POINTS, NOTICED_BY
from .scoring import ranked
from ..cube.color_scheme import ColorScheme, rotation_moves
from ..cube.repair import CHANCE_DISTANCE
from ..cube.witness import invert
from .taxonomy import Verdict

# Outcome colors (dataviz reference palette, validated as an adjacent set in this order).
OUTCOMES: tuple[tuple[str, str, str], ...] = (
    (Verdict.VALID_STANDARD, "Valid (standard)", "--outcome-1"),
    (Verdict.VALID_NONSTANDARD, "Valid (non-standard)", "--outcome-2"),
    (Verdict.INVALID, "Invalid", "--outcome-3"),
    (Verdict.INCONCLUSIVE, "Inconclusive", "--outcome-4"),
    ("generation_failed", "Generation failed", "--outcome-5"),
)
CHECK_RESULTS = {
    "pass": ("Pass", "--status-good"),
    "inconclusive": ("Inconclusive", "--status-warning"),
    "fail": ("Fail", "--status-critical"),
    "blocked": ("Blocked", "--blocked"),
}
# Heatmap buckets 0..10 (0%..100%): light steps run the blue ramp 100 -> 700;
# dark steps start near the dark surface and climb to a light blue.
_HEAT_LIGHT = ["#e6f0fc", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
               "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_HEAT_DARK = ["#1d2430", "#1c2c44", "#1b3352", "#1c3d63", "#1d4775", "#1f5389",
              "#2461a0", "#2d71b9", "#3987e5", "#5598e7", "#86b6ef"]
_HEAT_INK_LIGHT = ["#0b0b0b"] * 7 + ["#ffffff"] * 4
_HEAT_INK_DARK = ["#c3c2b7"] * 7 + ["#0b0b0b"] * 4


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _ranked(scores: dict[str, Any]) -> list[str]:
    return ranked(scores)


def _outcome_counts(s: dict[str, Any]) -> list[int]:
    return [s["verdicts"].get(str(v), 0) for v, _, _ in OUTCOMES[:-1]] + [s["generation_failures"]]


# ----------------------------------------------------------------------------- inline SVG

STRIP_W, STRIP_H, PAD = 200, 26, 12


def _x(value: float, maximum: float) -> float:
    return PAD + (STRIP_W - 2 * PAD) * value / maximum


# Validity is a share (0-1), shown as percent; ticks every 25%.
VALIDITY_TICKS = [(v / 4, f"{25 * v}") for v in range(5)]


def _axis(maximum: float, ticks: list[tuple[float, str]]) -> str:
    marks = "".join(
        f'<line x1="{_x(v, maximum):.1f}" x2="{_x(v, maximum):.1f}" y1="14" y2="18" class="axis"/>'
        f'<text x="{_x(v, maximum):.1f}" y="11" class="tick">{label}</text>'
        for v, label in ticks
    )
    return (
        f'<svg class="strip" width="{STRIP_W}" height="20" viewBox="0 0 {STRIP_W} 20" aria-hidden="true">'
        f'<line x1="{_x(0, maximum)}" x2="{_x(maximum, maximum)}" y1="18" y2="18" class="axis"/>{marks}</svg>'
    )


def _strip(mean: float | None, ci: list[float] | None, maximum: float, ticks: list[tuple[float, str]],
           label: str) -> str:
    """A mean as a dot on a 0..maximum scale, with its 95% interval as a line."""
    if mean is None:
        return '<span class="muted">no images</span>'
    grid = "".join(
        f'<line x1="{_x(v, maximum):.1f}" x2="{_x(v, maximum):.1f}" y1="3" y2="{STRIP_H - 3}" class="grid"/>'
        for v, _ in ticks
    )
    interval = (
        f'<line x1="{_x(ci[0], maximum):.1f}" x2="{_x(ci[1], maximum):.1f}" y1="{STRIP_H / 2}" '
        f'y2="{STRIP_H / 2}" class="ci"/>' if ci else ""
    )
    return (
        f'<svg class="strip" width="{STRIP_W}" height="{STRIP_H}" viewBox="0 0 {STRIP_W} {STRIP_H}" role="img" '
        f'aria-label="{escape(label)}"><title>{escape(label)}</title>{grid}{interval}'
        f'<circle cx="{_x(mean, maximum):.1f}" cy="{STRIP_H / 2}" r="5" class="dot"/></svg>'
    )


def _validity_strip(s: dict[str, Any]) -> str:
    mean, ci = s["validity_mean"], s["validity_ci95"]
    label = "" if mean is None else f"validity {_pct(mean)}" + (f", 95% CI {_pct(ci[0])}–{_pct(ci[1])}" if ci else "")
    return _strip(mean, ci, 1.0, VALIDITY_TICKS, label)


def _bar(counts: list[int], items: list[tuple[str, str]], width: int = 80) -> str:
    """A stacked bar of counts, one segment per (label, color token)."""
    total = sum(counts)
    if not total:
        return ""
    x, parts = 0.0, []
    for count, (label, token) in zip(counts, items):
        if count:
            w = width * count / total
            parts.append(
                f'<rect x="{x:.1f}" y="2" width="{max(w - 2, 1):.1f}" height="12" rx="2" style="fill:var({token})">'
                f"<title>{escape(label)}: {count} of {total}</title></rect>"
            )
            x += w
    return f'<svg class="bar" width="{width}" height="16" viewBox="0 0 {width} 16">{"".join(parts)}</svg>'


def _outcome_bar(s: dict[str, Any]) -> str:
    return _bar(_outcome_counts(s), [(label, token) for _, label, token in OUTCOMES])


# From "fools no one" (red) to "fools everyone" (green), then images that could not be fully checked.
FOOLS_ITEMS = [(f"fools {who}", token) for who, token in zip(
    FOOLS, ("--status-critical", "--status-warning", "--outcome-2", "--outcome-5", "--status-good"))]


def _fools_bar(s: dict[str, Any]) -> str:
    counts = [s["fools"][who] for who in FOOLS] + [s["fools"]["unclear"]]
    return _bar(counts, FOOLS_ITEMS + [("not fully checked", "--blocked")], width=64)


def _check_strip(checks: dict[str, str] | None) -> str:
    if not checks:
        return ""
    cells = []
    for i, c in enumerate(CHECKS):
        label, token = CHECK_RESULTS[checks.get(c.id, "blocked")]
        cells.append(
            f'<rect x="{i * 13}" y="0" width="11" height="11" rx="2" style="fill:var({token})">'
            f"<title>{i + 1}. {escape(c.title)}: {label}</title></rect>"
        )
    return f'<svg class="checks" width="{len(CHECKS) * 13}" height="11">{"".join(cells)}</svg>'


def _legend(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:var({token})"></i>{escape(label)}</span>' for label, token in items
    ) + "</div>"


# ----------------------------------------------------------------------------- page

CSS = """
:root { color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --grid:#e1e0d9;
  --axis:#c3c2b7; --border:rgba(11,11,11,.10); --series:#2a78d6;
  --outcome-1:#1baf7a; --outcome-2:#2a78d6; --outcome-3:#e34948; --outcome-4:#eda100; --outcome-5:#6250d6;
  --mark:#ff2bd6; --status-good:#0ca30c; --status-warning:#fab219; --status-critical:#d03b3b; --blocked:#e1e0d9; LIGHT_HEAT }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { DARK } }
:root[data-theme="dark"] { DARK }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 40px 0 8px; }
p, .meta { color: var(--ink-2); } .meta { margin: 0 0 8px; } .muted { color: var(--muted); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px;
  overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 8px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 600; font-size: 12px; vertical-align: bottom; }
td.wrap, th.wrap { white-space: normal; text-align: left; }
.strip line.grid { stroke: var(--grid); stroke-width: 1; } .strip line.axis { stroke: var(--axis); }
.strip line.ci { stroke: var(--axis); stroke-width: 2; stroke-linecap: round; }
.strip circle.dot { fill: var(--series); stroke: var(--surface); stroke-width: 2; }
.strip text.tick { fill: var(--muted); font-size: 10px; text-anchor: middle; }
.heat td.h { text-align: center; min-width: 38px; padding: 6px 2px; font-size: 12px; border: 2px solid var(--surface); }
.heat th.c { text-align: center; white-space: normal; min-width: 38px; max-width: 64px; padding: 4px 2px;
  font-size: 11px; font-weight: 500; line-height: 1.25; }
.heat td.m { white-space: normal; min-width: 140px; max-width: 200px; }
HEAT_CLASSES
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--ink-2); font-size: 12px; margin: 8px 0; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px;
  vertical-align: -1px; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 12px; }
.sample { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.sample img { display: block; width: 100%; aspect-ratio: 1; object-fit: cover; background: var(--grid); }
.frame { position: relative; display: block; }
.frame svg.marks { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
svg.marks polygon { fill: transparent; stroke-linejoin: round; vector-effect: non-scaling-stroke; pointer-events: all; }
svg.marks .halo { stroke: rgba(0,0,0,.75); stroke-width: 5; }
svg.marks .line { stroke: var(--mark); stroke-width: 2.5; }
svg.marks g:hover .line { stroke-width: 4; }
#show-marks:not(:checked) ~ .gallery svg.marks { display: none; }
.sample ul.marks { margin: 4px 0 0; padding-left: 16px; color: var(--ink); }
.sample .repairs { margin-top: 4px; color: var(--ink); }
.sample details.solution { margin-top: 4px; color: var(--ink); }
.sample details.solution summary { cursor: pointer; }
.sample details.solution code { display: block; margin: 4px 0; word-spacing: 2px; }
.sample .body { padding: 8px 10px 10px; font-size: 12px; color: var(--ink-2); }
.sample .head { display: flex; justify-content: space-between; gap: 8px; color: var(--ink); font-weight: 600;
  font-size: 13px; }
.sample .head > :first-child { min-width: 0; }
.sample .head > :last-child { flex-shrink: 0; white-space: nowrap; }
.sample svg.checks { display: block; margin: 6px 0; }
.sample .flags { margin-top: 4px; word-break: break-word; }
.noimg { aspect-ratio: 1; display: grid; place-items: center; color: var(--muted); background: var(--grid); }
a { color: var(--series); }
code { font-size: 12px; }
.badge { font-size: 11px; color: var(--ink-2); border: 1px solid var(--border); border-radius: 4px; padding: 0 5px; }
tr.reference td { background: color-mix(in srgb, var(--surface) 92%, var(--series)); }
/* The first column (model names) stays put while wide tables scroll sideways. */
.card th:first-child, .card td:first-child { position: sticky; left: 0; z-index: 1; background: var(--surface);
  box-shadow: 1px 0 0 var(--grid); }
tr.reference td:first-child { background: color-mix(in srgb, var(--surface) 92%, var(--series)); }
table.scores td { vertical-align: top; }
table.scores td.wrap { min-width: 180px; }
@media (max-width: 900px) {
  table.scores .chart { display: none; }
  table.scores th, table.scores td { padding: 6px 4px; }
}
"""
DARK_TOKENS = (
    "color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;"
    " --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10); --series:#3987e5;"
    " --outcome-1:#199e70; --outcome-2:#3987e5; --outcome-3:#e66767; --outcome-4:#c98500; --outcome-5:#9085e9;"
    " --blocked:#383835;"
)


def _css() -> str:
    light_heat = " ".join(f"--h{i}:{c}; --h{i}-ink:{k};" for i, (c, k) in enumerate(zip(_HEAT_LIGHT, _HEAT_INK_LIGHT)))
    dark_heat = " ".join(f"--h{i}:{c}; --h{i}-ink:{k};" for i, (c, k) in enumerate(zip(_HEAT_DARK, _HEAT_INK_DARK)))
    classes = "\n".join(f".h{i} {{ background: var(--h{i}); color: var(--h{i}-ink); }}" for i in range(11))
    return (
        CSS.replace("LIGHT_HEAT", light_heat)
        .replace("DARK", DARK_TOKENS + " " + dark_heat)
        .replace("HEAT_CLASSES", classes)
    )


def _scores_section(scores: dict[str, Any]) -> str:
    rows = []
    for model in _ranked(scores):
        s = scores[model]
        ci = s["validity_ci95"]
        rubric_ci = s["rubric_ci95"]
        rubric = "–" if s["rubric_mean"] is None else f"{s['rubric_mean']:.1f}"
        rubric_title = "" if rubric_ci is None else f" title='95% CI {rubric_ci[0]:.1f}–{rubric_ci[1]:.1f}'"
        cost = "–" if s["total_cost"] is None else f"${s['total_cost']:.3f}"
        latency = "" if s["mean_latency_s"] is None else f"{s['mean_latency_s']:.0f}s each"
        interval = "" if ci is None else f"<div class='muted'>{_pct(ci[0])}–{_pct(ci[1])}</div>"
        errors = s["generation_failures"]
        images_note = f"{s['passed']} passed" + (f" · {errors} error{'s' if errors != 1 else ''}" if errors else "")
        rows.append(
            f"<tr{' class=reference' if s.get('reference') else ''}><td class='wrap'>{escape(model)}"
            + (" <span class='badge'>reference</span>" if s.get("reference") else "")
            + f"<div class='muted'><code>{escape(s['model'])}</code></div></td>"
            f"<td title='95% CI'><b>{_pct(s['validity_mean'])}</b>{interval}</td>"
            f"<td class='chart'>{_validity_strip(s)}</td><td>{_fools_bar(s)}</td>"
            f"<td{rubric_title}>{rubric}<span class='muted'>/{MAX_POINTS:.0f}</span></td>"
            f"<td title='Images passing all {MAX_POINTS:.0f} checks; requests that returned no image'>"
            f"{s['images']}<div class='muted'>{images_note}</div></td>"
            f"<td class='chart'>{_outcome_bar(s)}</td><td title='Total cost; mean generation time per image'>{cost}<div class='muted'>{latency}</div></td></tr>"
        )
    return (
        "<h2>Scores</h2><p><b>Validity</b> says how close an image comes to a valid, scrambled, standard 3x3 "
        "cube: the fewest stickers you would have to recolor to get one (plus any sticker no recoloring fixes, "
        "such as one with letters or two colors), compared with random colors. 100% is a valid cube, 0% is no "
        "better than coloring the stickers at random (which needs about "
        f"{CHANCE_DISTANCE:.1f} recolorings, because the three hidden faces leave a lot of freedom). A barely "
        "scrambled cube or a designed pattern is held in slightly less regard: its validity is multiplied by a "
        "scramble factor between 80% and 100%. An image that is not read as a 3x3 cube, or is not scrambled at "
        "all, scores 0%.</p>"
        "<p><b>Who it fools</b> says who would be taken in by each image: every failed check is one that "
        f"{escape(NOTICED_BY[0])}, {escape(NOTICED_BY[1])}, {escape(NOTICED_BY[2])} or {escape(NOTICED_BY[3])} "
        "would notice, and an image fools everyone below the first who notices. An image that passes every "
        "check fools everyone.</p>"
        f"<p>The <b>rubric</b> explains what is wrong: up to {MAX_POINTS:.0f} points, 1 per check passed, ½ when "
        "inconclusive, 0 when failed or blocked (see <a href='#checks'>Checks</a>). An image <b>passes</b> only "
        "if it passes all checks. Scores are over generated images; requests that returned no image are counted "
        "separately as generation errors.</p>"
        "<div class='card'><table class='scores'><thead><tr><th>Model</th><th>Validity</th>"
        f"<th class='chart'>{_axis(1.0, VALIDITY_TICKS)}</th><th class='wrap'>Who it fools</th><th>Rubric</th><th>Images</th>"
        "<th class='chart'>Outcomes</th><th>Cost</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
        + _legend(FOOLS_ITEMS + [("not fully checked", "--blocked")])
        + _legend([(label, token) for _, label, token in OUTCOMES]) + "</div>"
    )


def _heatmap_section(scores: dict[str, Any]) -> str:
    head = "".join(
        f"<th class='c' title='{escape(c.description)}'>{i}. {escape(c.title)}</th>" for i, c in enumerate(CHECKS, 1)
    )
    rows = []
    for model in _ranked(scores):
        by_check = scores[model]["rubric_by_check"]
        cells = []
        for c in CHECKS:
            value = by_check.get(c.id)
            if value is None:
                cells.append("<td class='h muted'>–</td>")
            else:
                cells.append(
                    f"<td class='h h{round(value * 10)}' title='{escape(model)} · {escape(c.title)}: "
                    f"{_pct(value)} of points'>{_pct(value)}</td>"
                )
        rows.append(f"<tr><td class='m'>{escape(model)}</td>{''.join(cells)}</tr>")
    return (
        "<h2>Points per check</h2><p>Share of the available points each model earned on each check. "
        "A check that cannot be decided (e.g. sticker colors on a face that is not 3x3) is blocked and scores 0.</p>"
        f"<div class='card heat'><table><thead><tr><th>Model</th>{head}</tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def _checks_section() -> str:
    rows = "".join(
        f"<tr><td>{i}</td><td class='wrap'>{escape(c.title)}</td><td class='wrap'>{escape(c.description)}</td>"
        f"<td class='wrap'>{escape(c.needs) or '–'}</td></tr>"
        for i, c in enumerate(CHECKS, 1)
    )
    return (
        "<h2 id='checks'>Checks</h2><div class='card'><table><thead><tr><th>#</th><th class='wrap'>Check</th>"
        "<th class='wrap'>Passes when</th><th class='wrap'>Decided when read</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _details_section(scores: dict[str, Any]) -> str:
    parts = []
    flagged = [(m, scores[m]["flags"]) for m in _ranked(scores) if scores[m]["flags"]]
    if flagged:
        items = "".join(
            f"<li><b>{escape(m)}:</b> "
            + ", ".join(f"<code>{escape(f)}</code> × {n}" for f, n in flags.items()) + "</li>"
            for m, flags in flagged
        )
        parts.append(f"<h2>Failure flags</h2><p>Images carrying each flag (one image can carry several).</p><ul>{items}</ul>")
    failed = [(m, scores[m]["generation_failure_codes"]) for m in _ranked(scores) if scores[m]["generation_failure_codes"]]
    if failed:
        items = "".join(
            f"<li><b>{escape(m)}:</b> "
            + ", ".join(f"<code>{escape(c)}</code> × {n}" for c, n in codes.items()) + "</li>"
            for m, codes in failed
        )
        parts.append(f"<h2>Generation failures</h2><ul>{items}</ul>")
    return "".join(parts)


def _gallery_order(samples: list[dict[str, Any]], scores: dict[str, Any]) -> list[dict[str, Any]]:
    """Most valid first across all models, then by rubric points; ties by leaderboard rank,
    failed generations last."""
    rank = {model: i for i, model in enumerate(_ranked(scores))}

    def key(s: dict[str, Any]) -> tuple[float, float, int, int]:
        validity = (s.get("validity") or {}).get("score")
        return (-(validity if validity is not None else -1), -(s["points"] if s["points"] is not None else -1),
                rank.get(s["model_name"], len(rank)), s["iteration"])

    return sorted(samples, key=key)


def _text(label: str) -> str:
    """Escaped label whose color pairs ("B-G") never break across lines."""
    return re.sub(r"\b([WYROGBX])-(?=[WYROGBX]\b)", "\\1\u2011", escape(label))


def _solution_html(solution: dict[str, Any] | None) -> str:
    if not solution:
        return ""
    viewer = (f" <a href='{escape(solution['viewer'])}' target='_blank' rel='noopener'>View in 3D</a>"
              if solution["viewer"] else "")
    return (
        f"<details class='solution'><summary>Solution: {solution['count']} moves</summary>"
        f"<code>{escape(solution['moves'])}</code>"
        "<div class='muted'>A solution, not the only one: three faces are hidden, so many cubes match this "
        "image and the solver picked one of them. Found with Kociemba's two-phase solver, so shorter "
        f"solutions likely exist.{viewer}</div></details>"
    )


def _fools(level: int | None) -> str:
    if level is None:
        return ""
    return f"<div class='repairs'>Fools <b>{escape(FOOLS[level])}</b></div>"


def _repairs(validity: dict[str, Any] | None) -> str:
    """One line on how far the image is from a valid cube, e.g. "2 stickers to fix: U5 R→G, F3 B→G"."""
    if not validity:
        return ""
    if "repairs" not in validity:
        return f"<div class='repairs'>{escape(validity['reason'])}</div>"
    parts = [f"{f['facelet']} {f['seen']}→{f['color']}" for f in validity["fixes"]]
    parts += [f"{name} (sticker defect)" for name in validity["defective"]]
    count = len(parts)
    factor = validity.get("scramble_factor", 1.0)
    scramble = (f" · <span title='Barely scrambled or a designed pattern (scramble score {validity['scramble']:g})'>"
                f"scramble ×{factor:.2f}</span>" if factor < 1 else "")
    if not count:
        return f"<div class='repairs'>valid: nothing to fix{scramble}</div>"
    noun = "sticker" if count == 1 else "stickers"
    title = "Recoloring these stickers gives the nearest valid standard cube (one of them, when several tie)"
    return f"<div class='repairs' title='{title}'>{count} {noun} to fix: {_text(', '.join(parts))}{scramble}</div>"


def _inset(polygon: list[list[float]], keep: float = 0.88) -> list[tuple[float, float]]:
    """The polygon shrunk towards its center."""
    cx = sum(x for x, _ in polygon) / len(polygon)
    cy = sum(y for _, y in polygon) / len(polygon)
    return [(cx + (x - cx) * keep, cy + (y - cy) * keep) for x, y in polygon]


def _marks_svg(s: dict[str, Any]) -> str:
    """Outlines of the stickers or faces behind each failed check, over the image (hover for why)."""
    if not (marks := [m for m in s.get("marks") or [] if m["polygons"]]):
        return ""
    width, height = s["image_size"]
    groups = []
    for m in marks:
        # Each outline sits slightly inside its sticker (or face), so neighbouring outlines stay apart.
        points = [" ".join(f"{x:.1f},{y:.1f}" for x, y in _inset(poly)) for poly in m["polygons"]]
        halo = "".join(f"<polygon class='halo' points='{p}'/>" for p in points)
        line = "".join(f"<polygon class='line' points='{p}'/>" for p in points)
        groups.append(f"<g><title>{escape(m['label'])}</title>{halo}{line}</g>")
    # "slice" matches the image's object-fit: cover, so outlines stay on their stickers.
    return (f"<svg class='marks' viewBox='0 0 {width} {height}' preserveAspectRatio='xMidYMid slice'>"
            f"{''.join(groups)}</svg>")


def _samples_section(samples: list[dict[str, Any]], scores: dict[str, Any]) -> str:
    cards = []
    for s in _gallery_order(samples, scores):
        if s["image_path"]:
            src = escape(s["image_path"])
            image = (f"<a class='frame' href='{src}'><img loading='lazy' src='{src}' alt='{escape(s['sample_id'])}'>"
                     f"{_marks_svg(s)}</a>")
        else:
            image = f"<div class='noimg'>{escape(s.get('error') or 'no image')}</div>"
        if s.get("reference"):
            seed = "real photo"
        else:
            seed = f"seed {escape(str(s['seed']))}" if s["seed_supported"] else "unseeded"
        if s.get("backfilled"):
            seed += " · <span title='Generated after the run, when the original request failed'>backfilled</span>"
        overlay = f" · <a href='{escape(s['overlay_path'])}'>extraction</a>" if s.get("overlay_path") else ""
        points = "–" if s["points"] is None else f"{s['points']:g}/{MAX_POINTS:.0f}"
        validity = s.get("validity")
        headline = points if validity is None else f"{_pct(validity['score'])} · {points}"
        flags = ", ".join(escape(f) for f in s["flags"]) or "–"
        problems = "".join(f"<li>{_text(m['label'])}</li>" for m in s.get("marks") or [])
        problems = f"<ul class='marks'>{problems}</ul>" if problems else ""
        cards.append(
            f"<div class='sample'>{image}<div class='body'><div class='head'><span>{escape(s['model_name'])} "
            f"#{s['iteration']}</span><span>{headline}</span></div>{_check_strip(s.get('checks'))}"
            f"<div>{escape(s['verdict'])} · {seed}{overlay}</div>{_fools(s.get('fools'))}{_repairs(validity)}"
            f"{_solution_html(s.get('solution'))}{problems}"
            f"<div class='flags muted'>{flags}</div></div></div>"
        )
    legend = _legend([(label, token) for label, token in CHECK_RESULTS.values()] + [("problem outline", "--mark")])
    return (
        f"<h2>Samples</h2><p>Every attempted image, most valid first. Each card shows its validity and rubric "
        "points, and which stickers to recolor for the nearest valid cube. The squares are the "
        f"{len(CHECKS)} checks in order; hover for names. Where a failed check has a clear culprit (a duplicate center, an impossible piece, a "
        "face that is not 3x3, ...) it is outlined on the image and listed below it. Twist, flip and parity "
        "failures belong to the whole cube, so they are not outlined.</p>"
        "<input type='checkbox' id='show-marks' checked> <label for='show-marks'>Show problem outlines</label>"
        f"{legend}<div class='gallery'>{''.join(cards)}</div>"
    )


def render_html(stamp: str, run: dict[str, Any], scores: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    config = run.get("config", {})
    seeds = "random" if config.get("seed") is None else f"from {config['seed']}"
    extractor = next((s["extractor"] for s in samples if s["extractor"]), config.get("extractor", "–"))
    # Older runs name one backend; newer ones name it per model.
    backends = config.get("backend") or ", ".join(
        sorted({m["backend"] for m in config.get("models", {}).values() if isinstance(m, dict)})) or "?"
    meta = (
        f"<p class='meta'>Prompt: “{escape(str(config.get('prompt', '?')))}”</p>"
        f"<p class='meta'>Backend {escape(str(backends))} · "
        f"{escape(str(config.get('iterations', '?')))} images per model · seeds {seeds} · "
        f"extractor {escape(str(extractor))} · evaluator {EVALUATOR_VERSION} · "
        f"run started {escape(str(run.get('started_at', '?')))} · scored {escape(stamp)}</p>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>RubiksBench results</title><style>{_css()}</style></head><body><main>"
        f"<h1>Rubik's Cube benchmark results</h1>{meta}"
        f"<p class='meta'>Made with <a href='{REPOSITORY}'>RubiksBench</a> {__version__}: the benchmark's code, "
        "method and how to run it.</p>"
        + _scores_section(scores) + _heatmap_section(scores) + _samples_section(samples, scores)
        + _details_section(scores) + _checks_section()
        + "</main></body></html>\n"
    )


# Twizzle (cubing.js) shows a 3D cube in the standard colors with a setup and a solution.
VIEWER = "https://alpha.twizzle.net/edit/"


def _solution(evaluation: dict[str, Any]) -> dict[str, Any] | None:
    """For a valid image: the moves that solve the cube the solver completed it to, and a 3D
    viewer link when its colors are the standard ones (the viewer shows only those)."""
    side = {"VALID_STANDARD": "standard", "VALID_NONSTANDARD": "abstract"}.get(evaluation.get("verdict", ""))
    witness = (evaluation.get(side) or {}).get("witness") if side else None
    if not witness or witness.get("scramble") is None or not witness.get("scramble_verified"):
        return None
    moves = invert(witness["scramble"]) if witness["scramble"] else ""
    rotation = rotation_moves(ColorScheme.from_mapping(witness["scheme"])) if side == "standard" else None
    viewer = None
    if rotation is not None:
        # Turn the viewer's cube to this image's colors (top, front-left, front-right), then scramble.
        viewer = VIEWER + "?" + urlencode({"setup-alg": f"{rotation} {witness['scramble']}".strip(), "alg": moves})
    return {"moves": moves, "count": len(moves.split()), "viewer": viewer}


def _sample_marks(evaluation: dict[str, Any]) -> dict[str, Any]:
    """The rubric's marks with their outlines in image pixels (empty when nothing is marked)."""
    geometry = evaluation.get("extraction_details", {}).get("geometry")
    marks = evaluation.get("rubric", {}).get("marks")
    if not geometry or not marks:
        return {"marks": marks or []}
    return {
        "image_size": geometry["size"],
        "marks": [
            m | {"polygons": [geometry["stickers"][f] for f in m["facelets"] if f in geometry["stickers"]]
                 + [geometry["faces"][f] for f in m["faces"] if f in geometry["faces"]]}
            for m in marks
        ],
    }


def write_results(
    root: Path, run: dict[str, Any], generations: list[dict[str, Any]], evaluations: dict[str, dict[str, Any]],
    scores: dict[str, Any],
) -> dict[str, Path]:
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    samples = []
    for g in generations:
        e = evaluations.get(g["sample_id"], {})
        samples.append(
            {
                "sample_id": g["sample_id"],
                "model_name": g["model_name"],
                "model": g["request"]["model"],
                "iteration": g["iteration"],
                "seed": g["request"]["seed"],
                "seed_supported": g.get("seed_supported", True),
                "backfilled": "backfill_of" in g,
                "image_path": g.get("image_path"),
                "error": g["error"]["code"] if g.get("error") else None,
                "verdict": e.get("verdict", "generation_failed"),
                "flags": e.get("flags", []),
                "extractor": e.get("extractor"),
                "points": e.get("rubric", {}).get("points"),
                "checks": e.get("rubric", {}).get("checks"),
                "validity": e.get("validity"),
                "fools": e.get("rubric", {}).get("fools"),
                "solution": _solution(e),
                "reference": bool(g.get("reference")),
                **_sample_marks(e),
                "overlay_path": (
                    str(Path(g["image_path"]).parent / o)
                    if g.get("image_path") and (o := e.get("extraction_details", {}).get("overlay")) else None
                ),
            }
        )
    paths = {ext: root / f"results-{stamp}.{ext}" for ext in ("json", "html")}
    raw = {
        "scored_at": now.isoformat(),
        "evaluator_version": EVALUATOR_VERSION,
        "run": run,
        "scores": scores,
        "samples": samples,
        "evaluations": evaluations,
    }
    paths["json"].write_text(json.dumps(raw, indent=2))
    paths["html"].write_text(render_html(now.strftime("%Y-%m-%d %H:%M:%S UTC"), run, scores, samples))
    return paths
