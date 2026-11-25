"""Draw the animations used in the README.

Three short loops come first. One puts the three scalars beside a Transformer
block, one follows the leading cost as the series lengthens, and one shows
the dynamic weights under the priors. The long tour then keeps a single block
on screen and opens each part in turn. Three further animations isolate
locality, causality and location invariance. The tour's arithmetic is a real
forward pass on a short series.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2] / "docs" / "figures"
INK = "#e8edf2"
MUTED = "#93a0b0"
BG = "#10141b"
PANEL = "#181e28"
CHANNEL = ("#e0a15a", "#5ec8c5", "#e07a7a", "#8d7bff", "#8fce7a")
STAGE = "#090d12"
STAGE_INK = "#f4f7fa"
STAGE_MUTED = "#93a4b8"
AMBER = "#e4a45c"
TEAL = "#8fd0c4"
BLUE = "#8eb6e0"
LEARN = LinearSegmentedColormap.from_list("learn", ("#1a120c", "#7a4e22", "#e4a45c", "#f8ecd9"))
SIGNED = LinearSegmentedColormap.from_list("signed", ("#1a6f86", "#14343e", "#10161c", "#5c3d22", "#e4a45c", "#f6e7d0"))
FIELD = LinearSegmentedColormap.from_list("field", ("#10161c", "#12343a", "#1c6d74", "#7ad7cf", "#f2fffd"))
WARM = LinearSegmentedColormap.from_list("warm", ("#10161c", "#3a2416", "#a86b32", "#e4a45c", "#fff6ea"))
ATTENTION = LinearSegmentedColormap.from_list("attention", ("#0e141c", "#16324d", "#3d7eb8", "#d7e8f8"))
# A simplified block diagram, laid out left to right so it can sit above the matrices.
# Coordinates are figure fractions.
SCHEME = {
    "input": (0.05, 0.74, "Input"),
    "p1": (0.15, 0.645, "P1"),
    "local": (0.16, 0.835, "Locality"),
    "causal": (0.29, 0.835, "Causality"),
    "wdyn": (0.24, 0.74, "Wdyn"),
    "wmask": (0.37, 0.74, "Wmask"),
    "xt": (0.48, 0.835, "XT"),
    "zmix": (0.48, 0.74, "Zmix"),
    "shared": (0.58, 0.835, "Shared"),
    "indep": (0.58, 0.645, "Indep."),
    "p2": (0.68, 0.645, "P2"),
    "zproj": (0.78, 0.74, "Zproj"),
    "bias": (0.78, 0.645, "bias"),
    "gelu": (0.875, 0.74, "GELU"),
    "output": (0.955, 0.74, "Output"),
}
ORDER = ("input", "p1", "wdyn", "local", "causal", "wmask", "zmix", "p2", "zproj", "gelu", "output")
RELATED = {"zmix": ("xt",), "p2": ("shared", "indep"), "zproj": ("bias",)}
EDGES = (
    ("input", "local", None),
    ("input", "causal", None),
    ("input", "wdyn", None),
    ("input", "xt", "over"),
    ("p1", "wdyn", None),
    ("local", "wmask", "mid"),
    ("causal", "wmask", None),
    ("wdyn", "wmask", None),
    ("xt", "zmix", None),
    ("wmask", "zmix", None),
    ("shared", "p2", "drop"),
    ("indep", "p2", None),
    ("zmix", "zproj", None),
    ("p2", "zproj", None),
    ("bias", "zproj", None),
    ("zproj", "gelu", None),
    ("gelu", "output", None),
)
PANE = {"left": 0.07, "right": 0.93, "top": 0.45, "bottom": 0.075}
BOX_W, BOX_H = 0.064, 0.042


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    _dials(ROOT / "dials.gif")
    _scaling(ROOT / "scaling.gif")
    _fields(ROOT / "fields.gif")
    _block(ROOT / "block.gif")
    _locality(ROOT / "locality.gif")
    _causality(ROOT / "causality.gif")
    _location(ROOT / "location.gif")
    print(f"wrote {ROOT}")


def _block(path: Path) -> None:
    previous = plt.rcParams["font.family"]
    plt.rcParams["font.family"] = "Segoe UI"
    data = _pipeline()
    shots = _timeline()
    figure = plt.figure(figsize=(12.6, 7.05))
    frames: list[Image.Image] = []
    durations: list[int] = []
    try:
        for shot in shots:
            figure.clear()
            _scene(figure, data, shot)
            frames.append(_grab(figure))
            durations.append(shot["ms"])
        _write_gif(path, frames, durations)
    finally:
        plt.close(figure)
        plt.rcParams["font.family"] = previous
    seconds = sum(durations) / 1000
    print(f"block.gif  {len(frames)} frames  {seconds:.0f}s")


def _pipeline() -> dict:
    """One forward pass on a 12-step, 5-channel series with rank 4.

    The numbers are the ones the frames draw. A later stage never invents a
    matrix the earlier stages did not already produce.
    """
    length, channels, rank = 12, 5, 4
    clock = np.linspace(0, 1, length)
    series = np.stack(
        [
            np.sin(2 * np.pi * clock),
            0.7 * np.cos(4 * np.pi * clock),
            np.exp(-0.5 * ((clock - 0.42) / 0.07) ** 2),
            1.35 * (clock - 0.5),
            0.55 * np.sin(6 * np.pi * clock + 0.4),
        ],
        axis=1,
    )
    series = (series - series.mean(0)) / (series.std(0) + 1e-6)
    generator = np.random.default_rng(1)
    weights = generator.normal(0, 0.35, (channels, rank))
    weights[2, 1] += 1.35
    weights[0, 0] += 0.85
    weights[4, 3] -= 0.7
    dynamic = series @ weights
    centers = np.arange(rank) * (length - 1) / (rank - 1)
    distance = (np.arange(length)[:, None] - centers[None, :]) ** 2
    present = np.round(centers).astype(int)

    def locality(progress: float) -> np.ndarray:
        # progress 0 is almost flat (α large); progress 1 is local (α small).
        spread = (0.28 + 1.02 * (1 - progress)) * length
        return np.exp(-distance / (2 * spread**2))

    def causality(phi: float) -> np.ndarray:
        mask = np.ones((length, rank))
        for unit, token in enumerate(present):
            mask[token + 1 :, unit] = 1 - phi
        return mask

    local = locality(1.0)
    causal = causality(0.82)
    combined = local * causal
    masked = combined * dynamic
    mixed = series.T @ masked
    kernel = np.array([0.02, 1.0, 0.16, 0.0])
    shared_time = np.zeros((length, rank))
    for token, center in enumerate(np.round(np.arange(length) * (rank - 1) / (length - 1)).astype(int)):
        for unit in range(rank):
            shared_time[token, unit] = kernel[(unit - center) % rank]
    independent = np.clip(generator.normal(0.4, 0.22, (rank, length)), 0, 1)
    gate = 0.45
    projection = (1 - gate) * shared_time.T + gate * independent
    bias = 0.28 * np.sin(2 * np.pi * clock)
    raw = mixed @ projection
    projected = raw + bias
    activated = _gelu(projected.T)
    output = series + activated
    weight_start = generator.normal(0, 0.55, weights.shape)
    return {
        "series": series,
        "weights": weights,
        "weight_start": weight_start,
        "weight_delay": generator.uniform(0, 0.55, weights.shape),
        "dynamic": dynamic,
        "locality": locality,
        "causality": causality,
        "local": local,
        "causal": causal,
        "combined": combined,
        "masked": masked,
        "mixed": mixed,
        "shared": shared_time.T,
        "independent": independent,
        "projection": projection,
        "projected": projected,
        "activated": activated,
        "output": output,
        "span": {
            "series": _span(series),
            "weights": _span(np.stack([weights, weight_start])),
            "dynamic": _span(dynamic),
            "mixed": _span(mixed),
            "projection": (0.0, 1.0),
            "projected": _span(np.stack([raw, projected])),
            "activated": _span(np.stack([projected.T, activated])),
        },
    }


def _timeline() -> list[dict]:
    """The whole diagram stays on screen. Each part lights up, then its matrices play.

    Motion frames are finer than the holds, so the change reads smoothly and the result sits.
    """
    shots: list[dict] = []
    intro = "A series enters, is mixed and regulated, and comes back in its original shape."
    for index, node in enumerate(ORDER):
        shots.append({"layout": "map", "node": node, "sentence": intro, "ms": 320, "scan": index})
    shots.append({"layout": "map", "node": None, "sentence": intro, "ms": 2600, "scan": None})

    chapters = (
        ("input", "series", "Series", "Input", "Several channels, one series", "It starts as several channels sharing one clock.", 14, 170, 2200, 700),
        ("input", "matrix", "Matrix", "Matrix", "Rows are time, columns are channels", "The channels are stacked into a matrix. Rows are time. Columns are channels.", 12, 170, 2400, 600),
        ("p1", "weights", "Learnable", "Learnable weights", "Each cell is trained on its own", "A small matrix, P1, is learned. Each cell changes on its own.", 16, 170, 2400, 800),
        ("wdyn", "dynamic", "Product", "Dynamic weights", "S × r, not an S × S attention map", "Multiplying by P1 builds the mixing weights. One slim column per unit, not a full attention map.", 16, 170, 2800, 900),
        ("local", "locality", "Locality", "Locality", "How far each unit is allowed to see", "Locality sets how far from its own center each unit can see.", 16, 170, 2600, 900),
        ("causal", "causality", "Causality", "Causality", "Future tokens are turned down", "Causality turns down every token that lies in the future.", 16, 170, 2400, 600),
        ("wmask", "masks", "Combined", "One mask", "Locality and causality, cell by cell", "The two masks multiply into a single regulator.", 10, 170, 2400, 700),
        ("wmask", "masked", "Gated", "Gated weights", "The mask softens the mixer", "The regulator softens the dynamic weights before they are used.", 10, 170, 2600, 600),
        ("zmix", "tokens", "Tokens", "Token mixing", "Tokens are summed into each unit", "The gated weights mix the tokens. The channels are already combined.", 16, 170, 2800, 900),
        ("p2", "shared", "Pattern", "Shared pattern", "One pattern, shifted, or a free one", "One shared pattern, shifted in time, is blended with weights that are free.", 14, 170, 2500, 900),
        ("zproj", "project", "Project", "Project back", "Onto the length of the series, plus a bias", "The mixed tokens are laid back along the series, and a bias is added.", 10, 170, 2400, 800),
        ("gelu", "transpose", "Transpose", "Transpose", "Time runs down the rows again", "The result is turned so time runs down the rows again.", 1, 170, 2400, 700),
        ("gelu", "gelu", "GELU", "GELU", "Negatives are pulled toward zero", "GELU bends every value and pulls negatives toward zero.", 1, 170, 2400, 600),
        ("output", "residual", "Residual", "Residual", "The input is added back", "Adding the input back returns a series of the original shape.", 1, 170, 3200, 900),
    )
    for node, kind, chip, card, card_line, sentence, frames, frame_ms, hold_ms, card_ms in chapters:
        base = {"layout": "part", "node": node, "kind": kind, "chip": chip, "card": card, "card_line": card_line, "sentence": sentence}
        for index in range(frames):
            progress = 1.0 if frames == 1 else index / (frames - 1)
            duration = hold_ms if index == frames - 1 else frame_ms
            if index == 0 and frames > 1:
                duration = max(duration, card_ms)
            shots.append({**base, "phase": "detail", "zoom": 1.0, "progress": progress, "ms": duration})
    return shots


def _scene(figure, data, shot: dict) -> None:
    figure.patch.set_facecolor(STAGE)
    _heading(figure, shot)
    _scheme(figure, shot)
    if shot["layout"] == "map":
        note = "The block, simplified. Each part opens in turn." if shot["scan"] is None else "The highlight walks the diagram. Next we open that part."
        figure.text(0.5, 0.40, note, color=STAGE_MUTED, fontsize=13, ha="center", va="center")
        return
    _brackets(figure)
    label = SCHEME[shot["node"]][2]
    chip = label.upper() if shot["chip"].casefold() == label.casefold() else f"{label.upper()}    ·    {shot['chip'].upper()}"
    figure.text(0.5, 0.528, chip, color=STAGE_INK, fontsize=11, ha="center", va="center")
    {
        "series": _draw_series,
        "matrix": _draw_matrix,
        "weights": _draw_weights,
        "dynamic": _draw_dynamic,
        "locality": _draw_locality,
        "causality": _draw_causality,
        "masks": _draw_masks,
        "masked": _draw_masked,
        "tokens": _draw_tokens,
        "shared": _draw_shared,
        "project": _draw_project,
        "transpose": _draw_transpose,
        "gelu": _draw_gelu,
        "residual": _draw_residual,
    }[shot["kind"]](figure, data, shot["progress"])


def _heading(figure, shot: dict) -> None:
    figure.text(0.045, 0.972, "CLAIM", color=AMBER, fontsize=12, ha="left", va="center", fontweight="medium")
    figure.text(0.115, 0.972, "BLOCK", color=STAGE_MUTED, fontsize=12, ha="left", va="center")
    if shot.get("node") in ORDER:
        number = ORDER.index(shot["node"]) + 1
        figure.text(0.955, 0.972, f"{number}  /  {len(ORDER)}", color=STAGE_MUTED, fontsize=12, ha="right", va="center")
    figure.text(0.5, 0.932, shot["sentence"], color=STAGE_INK, fontsize=14, ha="center", va="center")


def _scheme(figure, shot: dict) -> None:
    focus = None if shot["layout"] == "map" and shot.get("scan") is None else shot["node"]
    _draw_scheme(figure, focus)


def _draw_scheme(figure, focus: str | None) -> None:
    done = set(ORDER[: ORDER.index(focus)]) if focus in ORDER else set()
    aspect = figure.get_size_inches()[1] / figure.get_size_inches()[0]
    _residual(figure, done, focus)
    for source, target, route in EDGES:
        _edge(figure, source, target, route, _edge_color(source, target, done, focus))
    for node, (x, y, label) in SCHEME.items():
        state = _state(node, focus, done)
        scale = 1.08 if state == "active" else 1.0
        _pill(figure, x, y, BOX_W * scale, BOX_H * scale, label, state, aspect)


def _state(node: str, focus: str | None, done: set[str]) -> str:
    if focus is None:
        return "calm"
    if node == focus or node in RELATED.get(focus, ()):
        return "active"
    if node in done or (node == "xt" and "zmix" in done) or (node in ("shared", "indep") and "p2" in done) or (node == "bias" and "zproj" in done):
        return "done"
    return "idle"


def _passed(node: str, done: set[str]) -> bool:
    if node in done:
        return True
    return (node == "xt" and "zmix" in done) or (node in ("shared", "indep") and "p2" in done) or (node == "bias" and "zproj" in done)


def _live(node: str, focus: str | None) -> bool:
    return node == focus or (focus is not None and node in RELATED.get(focus, ()))


def _edge_color(source: str, target: str, done: set[str], focus: str | None) -> str:
    if _passed(source, done) and _passed(target, done):
        return "#3d8f84"
    live = _live(source, focus) or _live(target, focus)
    known = live or _passed(source, done) or _passed(target, done)
    if live and known:
        return AMBER
    return "#2a3848"


def _edge(figure, source: str, target: str, route, color: str) -> None:
    start = SCHEME[source][:2]
    end = SCHEME[target][:2]
    if route == "over":
        points = [start, (start[0], 0.90), (end[0], 0.90), end]
    elif route == "mid":
        points = [start, (start[0], 0.788), (end[0], 0.788), end]
    elif route == "drop":
        points = [start, (end[0], start[1]), end]
    else:
        points = [start, end]
    points[0] = _leave(points[0], points[1], BOX_W, BOX_H)
    points[-1] = _leave(points[-1], points[-2], BOX_W, BOX_H)
    for first, second in zip(points, points[1:-1]):
        figure.add_artist(plt.Line2D([first[0], second[0]], [first[1], second[1]], transform=figure.transFigure, color=color, lw=1.05, solid_capstyle="round", zorder=1))
    figure.add_artist(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=8, lw=1.05, color=color, shrinkA=0, shrinkB=0, transform=figure.transFigure, zorder=1))


def _leave(origin, toward, box_w: float, box_h: float):
    dx, dy = toward[0] - origin[0], toward[1] - origin[1]
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return origin
    tx = (box_w / 2) / abs(dx) if dx else 1e9
    ty = (box_h / 2) / abs(dy) if dy else 1e9
    step = min(tx, ty, 0.98)
    return origin[0] + dx * step, origin[1] + dy * step


def _residual(figure, done: set[str], focus: str | None) -> None:
    color = "#3d8f84" if "output" in done or focus == "output" else "#243140"
    rail = 0.598
    x0, y0 = SCHEME["input"][0], SCHEME["input"][1]
    x1, y1 = SCHEME["output"][0], SCHEME["output"][1]
    figure.add_artist(plt.Line2D([x0, x0, x1, x1], [y0, rail, rail, y1], transform=figure.transFigure, color=color, lw=1.0, solid_capstyle="round", zorder=0))
    figure.text(0.30, rail - 0.016, "residual", color=STAGE_MUTED, fontsize=8, ha="center", va="top")


def _pill(figure, x, y, width, height, label, state, aspect) -> None:
    if state == "active":
        face, edge, text, line = "#2a2118", AMBER, "#f7f1e6", 1.6
        glow = FancyBboxPatch((x - width * 0.62, y - height * 0.72), width * 1.24, height * 1.44, boxstyle="round,pad=0.001,rounding_size=0.012", transform=figure.transFigure, facecolor=AMBER, edgecolor="none", alpha=0.14, mutation_aspect=aspect, zorder=2)
        figure.add_artist(glow)
    elif state == "done":
        face, edge, text, line = "#10211f", "#3d8f84", TEAL, 1.15
    elif state == "calm":
        face, edge, text, line = "#141b24", "#3d5166", "#d7e0ea", 1.15
    else:
        face, edge, text, line = "#10161d", "#2a3848", "#8b9caf", 1.0
    patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height, boxstyle="round,pad=0.001,rounding_size=0.008", transform=figure.transFigure, facecolor=face, edgecolor=edge, linewidth=line, mutation_aspect=aspect, zorder=3)
    figure.add_artist(patch)
    figure.text(x, y, label, ha="center", va="center", fontsize=8, color=text, fontweight="medium", zorder=4)


def _brackets(figure) -> None:
    left, right, bottom, top = 0.04, 0.96, 0.04, 0.548
    color = "#7d92a6"
    arm = 0.018
    corners = ((left, bottom, 1, 1), (right, bottom, -1, 1), (left, top, 1, -1), (right, top, -1, -1))
    for origin_x, origin_y, sign_x, sign_y in corners:
        figure.add_artist(plt.Line2D([origin_x, origin_x + sign_x * arm], [origin_y, origin_y], transform=figure.transFigure, color=color, lw=1.45))
        figure.add_artist(plt.Line2D([origin_x, origin_x], [origin_y, origin_y + sign_y * arm], transform=figure.transFigure, color=color, lw=1.45))


def _draw_series(figure, data, progress: float) -> None:
    axis = _panels(figure, [1])[0]
    _lines(axis, data["series"], progress, "input")


def _draw_matrix(figure, data, progress: float) -> None:
    left, arrow, right = _panels(figure, [1.15, 0.16, 0.72])
    _lines(left, data["series"], 1, "input")
    _symbol(arrow, "→")
    _heat(right, data["series"], "S × C", SIGNED, *data["span"]["series"], reveal=progress, along="row")


def _draw_weights(figure, data, progress: float) -> None:
    left, symbol, right = _panels(figure, [0.85, 0.16, 0.62])
    _heat(left, data["series"], "S × C", SIGNED, *data["span"]["series"])
    _symbol(symbol, "×")
    delay = data["weight_delay"]
    eased = np.clip((progress - delay) / np.maximum(1 - delay, 1e-6), 0, 1)
    eased = eased**2 * (3 - 2 * eased)
    shown = data["weight_start"] + (data["weights"] - data["weight_start"]) * eased
    _heat(right, shown, "learnable weights", LEARN, *data["span"]["weights"])
    right.set_xlabel("C × r", color=MUTED, fontsize=9, labelpad=6)


def _draw_dynamic(figure, data, progress: float) -> None:
    left, times, right, equals, result = _panels(figure, [0.72, 0.14, 0.5, 0.14, 0.62])
    contracting = _cursor(progress, data["weights"].shape[0])
    _heat(left, data["series"], "S × C", SIGNED, *data["span"]["series"], mark=("col", contracting) if contracting is not None else None)
    _symbol(times, "×")
    _heat(right, data["weights"], "learnable weights", LEARN, *data["span"]["weights"], mark=("row", contracting) if contracting is not None else None)
    right.set_xlabel("C × r", color=MUTED, fontsize=9, labelpad=6)
    _symbol(equals, "=")
    _heat(result, data["dynamic"], "dynamic weights", SIGNED, *data["span"]["dynamic"], reveal=0 if contracting is not None else (progress - 0.58) / 0.42)
    result.set_xlabel("S × r", color=MUTED, fontsize=9, labelpad=6)


def _draw_locality(figure, data, progress: float) -> None:
    axis = _panels(figure, [0.7])[0]
    _heat(axis, data["locality"](_settle(progress)), "S × r", FIELD, 0, 1)


def _draw_causality(figure, data, progress: float) -> None:
    axis = _panels(figure, [0.7])[0]
    _heat(axis, data["causality"](0.82 * _settle(progress)), "S × r", FIELD, 0, 1)


def _draw_masks(figure, data, progress: float) -> None:
    left, times, right, equals, result = _panels(figure, [0.55, 0.14, 0.55, 0.14, 0.55])
    _heat(left, data["local"], "locality", FIELD, 0, 1)
    left.set_xlabel("S × r", color=MUTED, fontsize=9, labelpad=6)
    _symbol(times, "$\\odot$")
    _heat(right, data["causal"], "causality", FIELD, 0, 1)
    right.set_xlabel("S × r", color=MUTED, fontsize=9, labelpad=6)
    _symbol(equals, "=")
    _heat(result, data["combined"], "combined mask", FIELD, 0, 1, reveal=max(progress, 0.04))
    result.set_xlabel("S × r", color=MUTED, fontsize=9, labelpad=6)


def _draw_masked(figure, data, progress: float) -> None:
    left, times, right, equals, result = _panels(figure, [0.55, 0.14, 0.55, 0.14, 0.55])
    low, high = data["span"]["dynamic"]
    _heat(left, data["combined"], "combined mask", FIELD, 0, 1)
    _symbol(times, "$\\odot$")
    _heat(right, data["dynamic"], "dynamic weights", SIGNED, low, high)
    _symbol(equals, "=")
    _heat(result, data["masked"], "regulated weights", SIGNED, low, high, reveal=max(progress, 0.04))


def _draw_tokens(figure, data, progress: float) -> None:
    left, times, right, equals, result = _panels(figure, [1.05, 0.14, 0.62, 0.14, 0.48])
    contracting = _cursor(progress, data["masked"].shape[0], stops=4)
    low, high = data["span"]["dynamic"]
    _heat(left, data["series"].T, "transposed input", SIGNED, *data["span"]["series"], mark=("col", contracting) if contracting is not None else None)
    left.set_xlabel("C × S", color=MUTED, fontsize=9, labelpad=6)
    _symbol(times, "×")
    _heat(right, data["masked"], "regulated weights", SIGNED, low, high, mark=("row", contracting) if contracting is not None else None)
    right.set_xlabel("S × r", color=MUTED, fontsize=9, labelpad=6)
    _symbol(equals, "=")
    _heat(result, data["mixed"], "mixed tokens", SIGNED, *data["span"]["mixed"], reveal=0 if contracting is not None else (progress - 0.58) / 0.42)
    result.set_xlabel("C × r", color=MUTED, fontsize=9, labelpad=6)


def _draw_shared(figure, data, progress: float) -> None:
    gate = 0.45 * progress
    left, middle, right = _panels(figure, [1, 1, 1])
    shown = (1 - gate) * data["shared"] + gate * data["independent"]
    _heat(left, data["shared"], "shared", FIELD, 0, 1, aspect="auto")
    _heat(middle, shown, f"blend  {gate:.2f}", FIELD, 0, 1, aspect="auto")
    _heat(right, data["independent"], "independent", FIELD, 0, 1, aspect="auto")


def _draw_project(figure, data, progress: float) -> None:
    left, times, right, equals, result = _panels(figure, [0.48, 0.14, 1.15, 0.14, 1.05])
    _heat(left, data["mixed"], "mixed tokens", SIGNED, *data["span"]["mixed"])
    left.set_xlabel("C × r", color=MUTED, fontsize=9, labelpad=6)
    _symbol(times, "×")
    _heat(right, data["projection"], "P2", FIELD, 0, 1)
    right.set_xlabel("r × S", color=MUTED, fontsize=9, labelpad=6)
    _symbol(equals, "=")
    raw = data["mixed"] @ data["projection"]
    shown = raw + progress * (data["projected"] - raw)
    _heat(result, shown, "C × S  + bias" if progress > 0.45 else "C × S", SIGNED, *data["span"]["projected"])


def _draw_transpose(figure, data, progress: float) -> None:
    del progress
    left, arrow, right = _panels(figure, [1.05, 0.22, 0.85])
    _heat(left, data["projected"], "C × S", SIGNED, *data["span"]["projected"])
    _symbol(arrow, "transpose")
    _heat(right, data["projected"].T, "S × C", SIGNED, *data["span"]["projected"])


def _draw_gelu(figure, data, progress: float) -> None:
    del progress
    low, high = data["span"]["activated"]
    left, arrow, right = _panels(figure, [0.8, 0.2, 0.8])
    _heat(left, data["projected"].T, "before GELU", SIGNED, low, high)
    _symbol(arrow, "→")
    _heat(right, data["activated"], "after GELU", SIGNED, low, high)


def _draw_residual(figure, data, progress: float) -> None:
    del progress
    left, symbol, right = _panels(figure, [1, 0.14, 1])
    _lines(left, data["series"], 1, "input")
    _symbol(symbol, "+")
    _lines(right, data["output"], 1, "output")


def _panels(figure, ratios: list[float]):
    grid = figure.add_gridspec(1, len(ratios), left=PANE["left"], right=PANE["right"], top=PANE["top"], bottom=PANE["bottom"], wspace=0.28, width_ratios=ratios)
    return [figure.add_subplot(grid[0, index]) for index in range(len(ratios))]


def _lines(axis, series: np.ndarray, progress: float, title: str) -> None:
    axis.set_facecolor("#0e141c")
    count = max(2, int(np.ceil(progress * series.shape[0])))
    abscissa = np.arange(count)
    for channel in range(series.shape[1]):
        color = CHANNEL[channel]
        axis.plot(abscissa, series[:count, channel], color=color, lw=5.2, alpha=0.16, solid_capstyle="round")
        axis.plot(abscissa, series[:count, channel], color=color, lw=1.7, solid_capstyle="round")
    axis.axhline(0, color="#243140", lw=0.8)
    axis.set_xlim(-0.4, series.shape[0] - 0.6)
    low, high = series.min(), series.max()
    pad = 0.18 * (high - low)
    axis.set_ylim(low - pad, high + pad)
    axis.set_title(title, color=STAGE_INK, fontsize=12, pad=8)
    axis.set_xlabel("time", color=STAGE_MUTED, fontsize=10)
    axis.tick_params(colors=STAGE_MUTED, labelsize=8)
    _box(axis)


def _heat(axis, matrix, title, cmap, vmin, vmax, reveal=1.0, along="col", mark=None, aspect="equal") -> None:
    shown = np.array(matrix, dtype=float)
    if reveal <= 0:
        shown[:] = np.nan
    elif reveal < 0.999:
        extent = shown.shape[1] if along == "col" else shown.shape[0]
        keep = max(1, int(np.ceil(reveal * extent)))
        if along == "col":
            shown[:, keep:] = np.nan
        else:
            shown[keep:, :] = np.nan
    rows, cols = matrix.shape
    axis.set_facecolor("#0e141c")
    if not np.all(np.isnan(shown)):
        palette = cmap.copy()
        palette.set_bad("#0e141c")
        axis.imshow(shown, cmap=palette, vmin=vmin, vmax=vmax, aspect=aspect, interpolation="nearest")
    else:
        axis.set_xlim(-0.5, cols - 0.5)
        axis.set_ylim(rows - 0.5, -0.5)
    if aspect == "equal":
        axis.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        axis.grid(which="minor", color=STAGE, lw=1.05)
        axis.tick_params(which="minor", length=0)
    axis.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)
    _box(axis)
    axis.set_title(title, color=STAGE_INK, fontsize=12, pad=8)
    if mark is not None and reveal >= 1:
        kind, index = mark
        patch = Rectangle((index - 0.5, -0.5), 1, rows, fill=False, edgecolor="#f6e7d0", lw=2.0, zorder=4) if kind == "col" else Rectangle((-0.5, index - 0.5), cols, 1, fill=False, edgecolor="#f6e7d0", lw=2.0, zorder=4)
        axis.add_patch(patch)
    axis.set_anchor("C")


def _symbol(axis, text: str) -> None:
    axis.axis("off")
    axis.text(0.5, 0.5, text, ha="center", va="center", color=STAGE_INK, fontsize=14 if len(text) > 2 else 22)


def _box(axis) -> None:
    for spine in axis.spines.values():
        spine.set_color("#314556")


def _span(matrix: np.ndarray) -> tuple[float, float]:
    limit = float(np.max(np.abs(matrix)))
    limit = max(limit, 1e-3)
    return -limit, limit


def _settle(progress: float) -> float:
    eased = float(np.clip((progress - 0.12) / 0.68, 0, 1))
    return eased * eased * (3 - 2 * eased)


def _cursor(progress: float, length: int, stops: int | None = None, until: float = 0.58):
    if progress >= until:
        return None
    stops = min(length, stops or length)
    slot = min(stops - 1, int(progress / until * stops))
    if stops == length:
        return slot
    return int(round(slot * (length - 1) / max(stops - 1, 1)))


def _gelu(values: np.ndarray) -> np.ndarray:
    inner = np.sqrt(2 / np.pi) * (values + 0.044715 * values**3)
    return 0.5 * values * (1 + np.tanh(inner))


def _grab(figure, dpi: int = 100) -> Image.Image:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor=figure.get_facecolor())
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB")


def _write_gif(path: Path, frames: list[Image.Image], durations: list[int]) -> None:
    sample = frames[:: max(1, len(frames) // 10)][:10]
    sheet = Image.new("RGB", (frames[0].width, sum(image.height for image in sample)))
    offset = 0
    for image in sample:
        sheet.paste(image, (0, offset))
        offset += image.height
    palette = sheet.quantize(colors=256)
    encoded = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in frames]
    # disposal 1 keeps unchanged pixels, and optimize stores only the region that moved.
    encoded[0].save(path, save_all=True, append_images=encoded[1:], duration=durations, loop=0, disposal=1, optimize=True)


def _dials(path: Path, *, sheet: Path | None = None, write: bool = True) -> None:
    """CLAIM beside a Transformer block, with the three scalars as dials.

    The heatmaps are the mixing weights and the projection of one short
    series. α and β change the mask. γ blends the projection. The attention
    map on the right is the same series and does not move.
    """
    length, rank = 16, 4
    demo = _demo(length, rank)
    _play(path, _scalar_beats(_dial_rows()), lambda figure, shot: _draw_dials(figure, shot, demo), (12.4, 6.7), sheet=sheet, write=write, dpi=82)


def _scaling(path: Path, *, sheet: Path | None = None, write: bool = True) -> None:
    """Leading cost against sequence length.

    The matrices share a token scale, so the drawn area is the number of
    entries. The curves are the leading terms per channel, S² and S r,
    divided by r² so they meet at 1 when S = r.
    """
    _play(path, _scale_shots(), _draw_scaling, (12.6, 6.3), sheet=sheet, write=write, dpi=82)


def _fields(path: Path, *, sheet: Path | None = None, write: bool = True) -> None:
    """Dynamic weights, and the priors that gate them.

    Arcs are the weights that remain after the mask. Teal fields are the
    locality and causality prior. The strip below is the projection γ blends.
    """
    length, rank = 22, 4
    demo = _demo(length, rank)
    shots = _scalar_beats(_field_rows())
    cycle = len(shots)
    for index, shot in enumerate(shots):
        shot["pulse"] = 0.5 - 0.5 * np.cos(2 * np.pi * index / cycle)
    _play(path, shots, lambda figure, shot: _draw_fields(figure, shot, demo), (12.4, 6.4), sheet=sheet, write=write, dpi=82)


def _demo(length: int, rank: int) -> dict:
    """One short series shared by the dial and field loops.

    The event is where the dynamic weights sit. The carrier only draws the
    sparkline, so the weights and the line agree about which tokens matter.
    """
    tokens = np.arange(length, dtype=float)
    clock = tokens / max(length - 1, 1)
    event = np.exp(-0.5 * ((tokens - 0.32 * (length - 1)) / 1.45) ** 2)
    event += 0.58 * np.exp(-0.5 * ((tokens - 0.70 * (length - 1)) / 1.85) ** 2)
    carrier = 0.22 * np.sin(2 * np.pi * clock * 1.5) + 0.12 * np.sin(2 * np.pi * clock * 3.2 + 0.5)
    gain = np.linspace(0.72, 1.0, rank)
    kernel = np.array([0.04, 1.0, 0.16, 0.0])[:rank]
    generator = np.random.default_rng(0)
    independent = np.clip(generator.normal(0.42, 0.2, (rank, length)), 0, 1)
    centers = np.arange(rank) * (length - 1) / max(rank - 1, 1)
    shifts = np.round(np.arange(length) * (rank - 1) / max(length - 1, 1)).astype(int)
    shared = np.zeros((rank, length))
    for time, center in enumerate(shifts):
        for unit in range(rank):
            shared[unit, time] = kernel[(unit - center) % rank]
    queries = np.arange(length)
    band = np.exp(-0.5 * ((queries[:, None] - queries[None, :]) / 1.7) ** 2)
    peaks = np.exp(-0.5 * ((queries[None, :] - 0.32 * (length - 1)) / 1.3) ** 2)
    peaks += 0.7 * np.exp(-0.5 * ((queries[None, :] - 0.70 * (length - 1)) / 1.6) ** 2)
    attention = 0.58 * band + 0.42 * peaks
    attention /= attention.max()
    return {
        "length": length,
        "rank": rank,
        "tokens": tokens,
        "event": event,
        "series": carrier + event,
        "content": event[None, :] * gain[:, None],
        "centers": centers,
        "shared": shared,
        "independent": independent,
        "attention": attention,
    }


def _mask(demo: dict, alpha: float, beta: float) -> np.ndarray:
    """Locality times causality, the same two factors as the block.

    α is σ(α) in the paper: the spread is 3 S σ(α). β is σ(β): future tokens
    are scaled by 1 − σ(β). A floor on the spread keeps the Gaussian finite
    when the dial sits at zero.
    """
    length, rank = demo["length"], demo["rank"]
    tokens = demo["tokens"]
    centers = demo["centers"]
    spread = 3.0 * length * max(alpha, 0.018)
    local = np.exp(-((tokens[None, :] - centers[:, None]) ** 2) / (2 * spread**2))
    causal = np.ones((rank, length))
    for unit, present in enumerate(np.round(centers).astype(int)):
        causal[unit, present + 1 :] = 1.0 - beta
    return local * causal


def _projection(demo: dict, gamma: float) -> np.ndarray:
    return (1.0 - gamma) * demo["shared"] + gamma * demo["independent"]


def _dial_rows() -> list[tuple]:
    poster = "CLAIM mixes with a short rank. Three dials shape it. Attention uses one full map."
    local = "α is locality. Each unit reads the tokens around its own center."
    wide = "As α grows the field widens, until the mask is nearly flat across the series."
    back = "Brought back down, the same unit is local again."
    causal = "β is causality. Tokens after a unit's present are turned down."
    opened = "With β at zero, those future tokens are visible again."
    free = "γ is location invariance. A shared kernel gives way to a free weight at every time."
    home = poster
    # a, b, g, focus, sentence, frames, frame_ms, hold_ms, ease
    return [
        (0.05, 0.0, 0.0, "none", poster, 1, 70, 1500, "smooth"),
        (0.05, 0.0, 0.0, "a", local, 1, 70, 700, "smooth"),
        (0.92, 0.0, 0.0, "a", wide, 16, 65, 1100, "open"),
        (0.05, 0.0, 0.0, "a", back, 12, 60, 650, "shut"),
        (0.05, 1.0, 0.0, "b", causal, 13, 65, 1100, "smooth"),
        (0.05, 0.0, 0.0, "b", opened, 9, 55, 500, "smooth"),
        (0.05, 0.0, 1.0, "g", free, 13, 65, 1100, "smooth"),
        (0.05, 0.0, 0.0, "none", home, 11, 55, 80, "smooth"),
    ]


def _field_rows() -> list[tuple]:
    poster = "Dynamic weights follow the event. Teal is the prior that multiplies them."
    local = "α is locality. Each teal field covers the tokens around one center."
    wide = "As α grows, those fields open and the weights reach across the series."
    causal = "β is causality. The prior drops every token after the present."
    free = "γ moves the projection from one shifted kernel to a weight of its own."
    return [
        (0.032, 0.0, 0.0, "none", poster, 1, 70, 1400, "smooth"),
        (0.032, 0.0, 0.0, "a", local, 1, 70, 500, "smooth"),
        (0.90, 0.0, 0.0, "a", wide, 16, 60, 1000, "open"),
        (0.032, 0.0, 0.0, "a", poster, 12, 55, 400, "shut"),
        (0.032, 1.0, 0.0, "b", causal, 13, 60, 1000, "smooth"),
        (0.032, 0.0, 0.0, "b", poster, 9, 50, 350, "smooth"),
        (0.032, 0.0, 1.0, "g", free, 13, 60, 1000, "smooth"),
        (0.032, 0.0, 0.0, "none", poster, 11, 50, 80, "smooth"),
    ]


def _scalar_beats(rows: list[tuple]) -> list[dict]:
    """Walk a list of poses. Each row is the end of a move from the previous one."""
    shots: list[dict] = []
    opening = rows[0]
    shots.append(_pose(opening, opening, 1.0))
    shots[-1]["ms"] = opening[7]
    start = opening
    for row in rows[1:]:
        frames = row[5]
        if frames == 1:
            shots.append(_pose(row, row, 1.0))
            shots[-1]["ms"] = row[7]
        else:
            for index in range(1, frames):
                progress = _ease(index / (frames - 1), row[8])
                shot = _pose(start, row, progress)
                shot["ms"] = row[7] if index == frames - 1 else row[6]
                shots.append(shot)
        start = row
    return shots


def _pose(start: tuple, end: tuple, progress: float) -> dict:
    return {
        "a": _mix(start[0], end[0], progress),
        "b": _mix(start[1], end[1], progress),
        "g": _mix(start[2], end[2], progress),
        "focus": end[3],
        "sentence": end[4],
        "pulse": 0.0,
        "ms": end[6],
    }


def _scale_shots() -> list[dict]:
    """Grow from half the rank to five times the rank, hold, then rewind."""
    rank = 48
    start, stop = rank / 2, 5 * rank
    frames = 44
    shots = []
    for index in range(frames):
        progress = _ease(index / (frames - 1), "smooth")
        shots.append({"s": start + (stop - start) * progress, "rank": rank, "ms": 80})
    shots[-1]["ms"] = 1500
    for index in range(1, 10):
        progress = _ease(1 - index / 9, "smooth")
        shots.append({"s": start + (stop - start) * progress, "rank": rank, "ms": 45})
    shots[-1]["ms"] = 200
    return shots


def _draw_dials(figure, shot: dict, demo: dict) -> None:
    _banner(figure, "THREE SCALARS", shot["sentence"])
    _panel(figure, (0.025, 0.035, 0.605, 0.855))
    _panel(figure, (0.65, 0.035, 0.325, 0.855))
    figure.text(0.05, 0.845, "CLAIM", color=AMBER, fontsize=15, fontweight="medium", va="center")
    figure.text(0.135, 0.845, "rank r", color=STAGE_MUTED, fontsize=12, va="center")
    figure.text(0.675, 0.845, "Transformer", color=BLUE, fontsize=15, fontweight="medium", va="center")
    figure.text(0.86, 0.845, "attention", color=STAGE_MUTED, fontsize=12, va="center")

    spark = figure.add_axes((0.05, 0.785, 0.30, 0.045))
    weights = figure.add_axes((0.05, 0.555, 0.30, 0.145))
    projection = figure.add_axes((0.05, 0.365, 0.30, 0.13))
    for axis in (spark, weights, projection):
        axis.set_zorder(2)
    _spark(spark, demo["series"])
    _matrix(weights, demo["content"] * _mask(demo, shot["a"], shot["b"]), WARM, 0.0, float(demo["content"].max()), shot["focus"] in ("a", "b"))
    _matrix(projection, _projection(demo, shot["g"]), FIELD, 0.0, 1.0, shot["focus"] == "g")
    _caption(figure, 0.05, 0.742, "mixing weights", "S × r", shot["focus"] in ("a", "b"))
    _caption(figure, 0.05, 0.525, "projection", "r × S", shot["focus"] == "g")

    dials = figure.add_axes((0.04, 0.05, 0.57, 0.28))
    dials.set_zorder(2)
    dials.set_xlim(0, 3.74)
    dials.set_ylim(0, 1)
    dials.set_aspect("equal", adjustable="box")
    dials.axis("off")
    _dial(dials, 0.68, 0.56, shot["a"], shot["focus"] == "a", r"$\alpha$", _alpha_word(shot["a"]))
    _dial(dials, 1.87, 0.56, shot["b"], shot["focus"] == "b", r"$\beta$", _beta_word(shot["b"]))
    _dial(dials, 3.06, 0.56, shot["g"], shot["focus"] == "g", r"$\gamma$", _gamma_word(shot["g"]))

    series = figure.add_axes((0.70, 0.785, 0.23, 0.045))
    attention = figure.add_axes((0.692, 0.20, 0.246, 0.436))
    series.set_zorder(2)
    attention.set_zorder(2)
    _spark(series, demo["series"])
    _matrix(attention, demo["attention"], ATTENTION, 0.0, 1.0, False)
    attention.set_aspect("equal", adjustable="box")
    figure.text(0.815, 0.145, "attention map    S × S", color=STAGE_INK, fontsize=12, ha="center", va="center")
    figure.text(0.815, 0.095, "one weight for every pair of tokens", color=STAGE_MUTED, fontsize=10, ha="center", va="center")


def _draw_scaling(figure, shot: dict) -> None:
    length = int(round(shot["s"]))
    rank = int(shot["rank"])
    ratio = length / rank
    _banner(
        figure,
        "LEADING COST",
        "Per channel, attention follows S² and CLAIM follows S. The rank stays fixed.",
    )
    limit = 5 * rank
    attention = figure.add_axes((0.05, 0.30, 0.26, 0.48))
    claim = figure.add_axes((0.345, 0.30, 0.26, 0.48))
    _growing(attention, _attention_texture(limit)[:length, :length], length, length, limit, BLUE, ATTENTION)
    _growing(claim, _claim_texture(limit, rank)[:length, :rank], rank, length, limit, TEAL, FIELD)
    attention.set_title("attention    S × S", color=BLUE, fontsize=13, pad=8)
    claim.set_title("CLAIM    S × r", color=TEAL, fontsize=13, pad=8)
    figure.text(0.18, 0.22, f"{length * length:,}", color=BLUE, fontsize=18, ha="center", va="center")
    figure.text(0.18, 0.155, "entries", color=STAGE_MUTED, fontsize=10, ha="center", va="center")
    figure.text(0.475, 0.22, f"{length * rank:,}", color=TEAL, fontsize=18, ha="center", va="center")
    figure.text(0.475, 0.155, "entries", color=STAGE_MUTED, fontsize=10, ha="center", va="center")
    _meters(figure, ratio)

    chart = figure.add_axes((0.66, 0.24, 0.30, 0.54))
    _cost_chart(chart, length, rank)
    ratio_color = TEAL if ratio < 0.97 else (AMBER if ratio < 1.08 else BLUE)
    chart.set_title(f"S = {length}      r = {rank}      S / r = {ratio:.2f}", color=ratio_color, fontsize=12, pad=8)


def _draw_fields(figure, shot: dict, demo: dict) -> None:
    _banner(figure, "WEIGHTS", shot["sentence"])
    _sliders(figure, shot)
    stage = figure.add_axes((0.06, 0.30, 0.90, 0.44))
    _field_stage(stage, demo, shot)
    strip = figure.add_axes((0.06, 0.08, 0.90, 0.15))
    _matrix(strip, _projection(demo, shot["g"]), FIELD, 0.0, 1.0, shot["focus"] == "g")
    figure.text(0.06, 0.245, "projection", color=AMBER if shot["focus"] == "g" else STAGE_MUTED, fontsize=11, va="center")
    figure.text(0.16, 0.245, "r  ×  S", color=STAGE_MUTED, fontsize=11, va="center")
    figure.text(0.96, 0.245, _gamma_word(shot["g"]), color=STAGE_INK, fontsize=11, ha="right", va="center")


def _field_stage(axis, demo: dict, shot: dict) -> None:
    length = demo["length"]
    tokens = demo["tokens"]
    centers = demo["centers"]
    pulse = 0.84 + 0.16 * shot["pulse"]
    event = demo["event"] * pulse
    energy = event / demo["event"].max()
    mask = _mask(demo, shot["a"], shot["b"])
    weights = demo["content"] * mask * pulse
    peak = float(demo["content"].max())

    axis.set_facecolor(STAGE)
    axis.set_xlim(-1.3, length - 0.3)
    axis.set_ylim(1.6, 9.2)
    axis.axis("off")

    _prior_fields(axis, demo, shot["a"], shot["b"])
    _arcs(axis, tokens, centers, weights / peak)
    ink = np.zeros((length, 4))
    ink[:, 0], ink[:, 1], ink[:, 2] = 0.894, 0.643, 0.361
    ink[:, 3] = 0.30 + 0.70 * energy
    axis.scatter(tokens, np.full(length, 7.15), s=42 + 170 * energy, c=ink, linewidths=0, zorder=5)
    axis.scatter(centers, np.full(demo["rank"], 2.85), s=110, marker="D", c="#10211f", edgecolors=TEAL, linewidths=1.4, zorder=6)
    axis.text(-1.05, 7.15, "tokens", color=STAGE_MUTED, fontsize=9, ha="right", va="center")
    axis.text(-1.05, 2.85, "units", color=STAGE_MUTED, fontsize=9, ha="right", va="center")


def _prior_fields(axis, demo: dict, alpha: float, beta: float) -> None:
    length = demo["length"]
    grid = np.linspace(-0.6, length - 0.4, 400)
    spread = 3.0 * length * max(alpha, 0.018)
    for center in demo["centers"]:
        present = round(float(center))
        local = np.exp(-((grid - center) ** 2) / (2 * spread**2))
        causal = np.where(grid <= present + 0.5, 1.0, 1.0 - beta)
        field = local * causal
        axis.fill_between(grid, 5.85, 5.85 + 1.15 * field, color=TEAL, alpha=0.55, linewidth=0, zorder=2)


def _arcs(axis, tokens: np.ndarray, centers: np.ndarray, strength: np.ndarray) -> None:
    from matplotlib.collections import LineCollection

    segments = []
    colors = []
    widths = []
    amber = (228 / 255, 164 / 255, 92 / 255)
    for unit, center in enumerate(centers):
        for token in range(len(tokens)):
            weight = float(np.round(strength[unit, token] * 6) / 6)
            if weight < 0.18:
                continue
            span = abs(float(tokens[token]) - float(center))
            samples = np.linspace(0, 1, 12)
            control_x = (tokens[token] + center) / 2
            control_y = 4.7 + min(1.35, 0.25 + 0.10 * span)
            abscissa = (1 - samples) ** 2 * tokens[token] + 2 * (1 - samples) * samples * control_x + samples**2 * center
            ordinate = (1 - samples) ** 2 * 6.55 + 2 * (1 - samples) * samples * control_y + samples**2 * 3.15
            segments.append(np.column_stack((abscissa, ordinate)))
            colors.append((*amber, 0.15 + 0.85 * weight))
            widths.append(0.6 + 1.8 * weight)
    if segments:
        axis.add_collection(LineCollection(segments, colors=colors, linewidths=widths, capstyle="round", zorder=3))


def _sliders(figure, shot: dict) -> None:
    specs = (
        (0.08, "a", r"$\alpha$", "locality", "local", "global", shot["a"]),
        (0.38, "b", r"$\beta$", "causality", "open", "causal", shot["b"]),
        (0.68, "g", r"$\gamma$", "location", "shared", "free", shot["g"]),
    )
    for origin, focus, glyph, name, left, right, value in specs:
        active = shot["focus"] == focus
        color = AMBER if active else "#5c6e80"
        figure.text(origin, 0.785, glyph, color=color, fontsize=14, va="center")
        figure.text(origin + 0.025, 0.785, name, color=color, fontsize=11, va="center")
        figure.text(origin + 0.20, 0.785, f"{value:.2f}", color=STAGE_INK if active else STAGE_MUTED, fontsize=11, va="center", ha="right")
        y = 0.755
        width = 0.20
        figure.add_artist(plt.Line2D([origin, origin + width], [y, y], transform=figure.transFigure, color="#243140", lw=4, solid_capstyle="round"))
        figure.add_artist(plt.Line2D([origin, origin + width * value], [y, y], transform=figure.transFigure, color=color, lw=4, solid_capstyle="round"))
        figure.add_artist(plt.Line2D([origin + width * value], [y], transform=figure.transFigure, marker="o", markersize=10, color="#f7f1e6" if active else color, markeredgecolor=color, markeredgewidth=0))
        figure.text(origin, 0.728, left, color=STAGE_MUTED, fontsize=8, ha="left", va="top")
        figure.text(origin + width, 0.728, right, color=STAGE_MUTED, fontsize=8, ha="right", va="top")


def _dial(axis, cx: float, cy: float, value: float, active: bool, glyph: str, caption: str) -> None:
    radius = 0.34
    color = AMBER if active else "#5c6e80"
    if active:
        axis.add_patch(plt.Circle((cx, cy), radius * 1.22, color=AMBER, alpha=0.10, zorder=0))
    axis.add_patch(plt.Circle((cx, cy), radius, facecolor="#10161d", edgecolor=color, lw=1.7 if active else 1.05, zorder=1))
    track = np.linspace(np.deg2rad(206), np.deg2rad(206 - 232), 90)
    axis.plot(cx + radius * 0.74 * np.cos(track), cy + radius * 0.74 * np.sin(track), color="#243140", lw=3.2, solid_capstyle="round", zorder=2)
    end = np.deg2rad(206 - 232 * float(np.clip(value, 0, 1)))
    filled = np.linspace(np.deg2rad(206), end, max(3, int(48 * value + 2)))
    axis.plot(cx + radius * 0.74 * np.cos(filled), cy + radius * 0.74 * np.sin(filled), color=color, lw=3.3, solid_capstyle="round", zorder=3)
    axis.plot([cx, cx + radius * 0.52 * np.cos(end)], [cy, cy + radius * 0.52 * np.sin(end)], color="#f4f7fa" if active else "#9aabbc", lw=1.5, solid_capstyle="round", zorder=4)
    axis.add_patch(plt.Circle((cx, cy), 0.035, color=color, zorder=5))
    axis.text(cx, cy - 0.12, glyph, color="#f7f1e6" if active else "#d5dee8", fontsize=13, ha="center", va="center", zorder=6)
    axis.text(cx, cy - radius - 0.08, caption, color=color, fontsize=8.5, ha="center", va="top", zorder=6)


def _alpha_word(value: float) -> str:
    if value < 0.18:
        return "local"
    if value < 0.45:
        return "opening"
    return "nearly global"


def _beta_word(value: float) -> str:
    if value < 0.12:
        return "future open"
    if value < 0.82:
        return "future dimmed"
    return "causal"


def _gamma_word(value: float) -> str:
    if value < 0.12:
        return "shared kernel"
    if value < 0.82:
        return "blended"
    return "independent"


def _attention_texture(limit: int) -> np.ndarray:
    index = np.arange(limit)
    band = np.exp(-0.5 * ((index[:, None] - index[None, :]) / 7.0) ** 2)
    peaks = np.exp(-0.5 * ((index[None, :] - 0.30 * (limit - 1)) / 9.0) ** 2)
    peaks += 0.75 * np.exp(-0.5 * ((index[None, :] - 0.62 * (limit - 1)) / 12.0) ** 2)
    texture = 0.55 * band + 0.45 * peaks
    return 0.22 + 0.78 * texture / texture.max()


def _claim_texture(limit: int, rank: int) -> np.ndarray:
    time = np.arange(limit)
    event = np.exp(-0.5 * ((time - 0.30 * (limit - 1)) / 10.0) ** 2)
    event += 0.6 * np.exp(-0.5 * ((time - 0.62 * (limit - 1)) / 14.0) ** 2)
    return 0.28 + 0.72 * event[:, None] * np.linspace(0.85, 1.0, rank)[None, :]


def _growing(axis, matrix: np.ndarray, width: int, height: int, limit: int, edge: str, cmap) -> None:
    axis.set_facecolor("#0e141c")
    axis.imshow(matrix, origin="lower", cmap=cmap, vmin=0, vmax=1, interpolation="nearest", extent=(-0.5, width - 0.5, -0.5, height - 0.5))
    axis.set_xlim(-0.5, limit - 0.5)
    axis.set_ylim(-0.5, limit - 0.5)
    axis.set_aspect("equal", adjustable="box")
    axis.add_patch(Rectangle((-0.5, -0.5), limit, limit, fill=False, edgecolor="#243140", lw=1.0, zorder=2))
    axis.add_patch(Rectangle((-0.5, -0.5), width, height, fill=False, edgecolor=edge, lw=1.6, zorder=3))
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#314556")


def _meters(figure, ratio: float) -> None:
    """Two bars on one scale: each cost divided by the larger of the two."""
    attention = ratio / max(ratio, 1.0)
    claim = 1.0 / max(ratio, 1.0)
    _bar(figure, 0.05, 0.26, attention, BLUE)
    _bar(figure, 0.345, 0.26, claim, TEAL)


def _bar(figure, x: float, width: float, fill: float, color: str) -> None:
    y = 0.075
    figure.add_artist(FancyBboxPatch((x, y), width, 0.018, boxstyle="round,pad=0,rounding_size=0.006", transform=figure.transFigure, facecolor="#141b24", edgecolor="#243140", linewidth=0.8, mutation_aspect=figure.get_figheight() / figure.get_figwidth()))
    if fill > 0.01:
        figure.add_artist(FancyBboxPatch((x, y), width * fill, 0.018, boxstyle="round,pad=0,rounding_size=0.006", transform=figure.transFigure, facecolor=color, edgecolor="none", alpha=0.9, mutation_aspect=figure.get_figheight() / figure.get_figwidth()))


def _cost_chart(axis, length: int, rank: int) -> None:
    grid = np.linspace(rank / 2, 5 * rank, 400)
    claim = grid / rank
    attention = claim**2
    shown = grid <= length
    axis.set_facecolor("#0e141c")
    axis.plot(grid, attention, color="#24384c", lw=1.4)
    axis.plot(grid, claim, color="#1d3330", lw=1.4)
    axis.fill_between(grid[shown], attention[shown], color=BLUE, alpha=0.13, linewidth=0)
    axis.plot(grid[shown], attention[shown], color=BLUE, lw=2.1, label="attention    S²")
    axis.plot(grid[shown], claim[shown], color=TEAL, lw=2.1, label="CLAIM    S r")
    axis.scatter([length, length], [(length / rank) ** 2, length / rank], s=28, c=[BLUE, TEAL], zorder=4)
    axis.axvline(rank, color="#3d5166", lw=0.9, ls=(0, (2, 2)))
    axis.axhline(1.0, color="#3d5166", lw=0.8, ls=(0, (2, 2)))
    axis.text(rank + 6, 1.15, "S = r", color=STAGE_MUTED, fontsize=9)
    axis.set_xlim(grid[0], grid[-1])
    axis.set_ylim(0.18, 32)
    axis.set_yscale("log")
    axis.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16])
    axis.set_yticklabels(["0.25", "0.5", "1", "2", "4", "8", "16"])
    axis.set_xlabel("sequence length  S", color=STAGE_MUTED, fontsize=10)
    axis.axvline(length, color=AMBER, lw=0.7, alpha=0.55)
    axis.set_ylabel("relative to the cost at  S = r", color=STAGE_MUTED, fontsize=10)
    axis.tick_params(colors=STAGE_MUTED, labelsize=8)
    axis.grid(True, color="#1c2836", lw=0.7)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#314556")
    legend = axis.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(STAGE_INK)


def _spark(axis, series: np.ndarray) -> None:
    axis.set_facecolor("#0e141c")
    abscissa = np.arange(len(series))
    axis.plot(abscissa, series, color=AMBER, lw=1.6, solid_capstyle="round")
    axis.set_xlim(-0.4, len(series) - 0.6)
    pad = 0.25 * (series.max() - series.min())
    axis.set_ylim(series.min() - pad, series.max() + pad)
    axis.axis("off")


def _matrix(axis, matrix: np.ndarray, cmap, vmin: float, vmax: float, active: bool) -> None:
    axis.set_facecolor("#0e141c")
    axis.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(AMBER if active else "#314556")
        spine.set_linewidth(1.7 if active else 1.0)


def _panel(figure, box: tuple[float, float, float, float]) -> None:
    aspect = figure.get_figheight() / figure.get_figwidth()
    patch = FancyBboxPatch(box[:2], box[2], box[3], boxstyle="round,pad=0.004,rounding_size=0.012", transform=figure.transFigure, facecolor="#10161d", edgecolor="#2a3848", linewidth=1.0, mutation_aspect=aspect)
    patch.set_zorder(0)
    figure.add_artist(patch)


def _caption(figure, x: float, y: float, title: str, shape: str, active: bool) -> None:
    figure.text(x, y, title, color=AMBER if active else STAGE_INK, fontsize=11, ha="left", va="center")
    figure.text(x + 0.20, y, shape, color=STAGE_MUTED, fontsize=11, ha="left", va="center")


def _banner(figure, kicker: str, sentence: str) -> None:
    figure.text(0.04, 0.962, "CLAIM", color=AMBER, fontsize=12, fontweight="medium", va="center")
    figure.text(0.115, 0.962, kicker, color=STAGE_MUTED, fontsize=12, va="center")
    figure.text(0.5, 0.915, sentence, color=STAGE_INK, fontsize=13, ha="center", va="center")


def _play(path: Path, shots: list[dict], draw, size: tuple[float, float], *, sheet: Path | None, write: bool, dpi: int = 100) -> None:
    previous = plt.rcParams["font.family"]
    mathfont = plt.rcParams["mathtext.fontset"]
    plt.rcParams["font.family"] = "Segoe UI"
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    figure = plt.figure(figsize=size)
    frames: list[Image.Image] = []
    durations: list[int] = []
    try:
        for shot in shots:
            figure.clear()
            figure.patch.set_facecolor(STAGE)
            draw(figure, shot)
            frames.append(_grab(figure, dpi))
            durations.append(int(shot["ms"]))
        if sheet is not None:
            _sheet(sheet, frames)
        if write:
            _write_gif(path, frames, durations)
    finally:
        plt.close(figure)
        plt.rcParams["font.family"] = previous
        plt.rcParams["mathtext.fontset"] = mathfont
    seconds = sum(durations) / 1000
    size_kb = path.stat().st_size / 1024 if write and path.exists() else 0
    print(f"{path.name}  {len(frames)} frames  {seconds:.1f}s  {size_kb:.0f} KB")


def _sheet(path: Path, frames: list[Image.Image]) -> None:
    picks = [frames[int(index)] for index in np.linspace(0, len(frames) - 1, 8)]
    thumb_w = 520
    thumbs = [image.resize((thumb_w, int(image.height * thumb_w / image.width)), Image.Resampling.LANCZOS) for image in picks]
    columns = 4
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumbs[0].height), (9, 13, 18))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * thumb_w, (index // columns) * thumb.height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _ease(progress: float, kind: str) -> float:
    if kind == "open":
        return progress**2
    if kind == "shut":
        return 1 - (1 - progress) ** 2
    return progress * progress * (3 - 2 * progress)


def _mix(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def _locality(path: Path) -> None:
    length, rank = 80, 8
    tokens = np.arange(length)
    units = np.arange(rank)
    centers = units * (length - 1) / (rank - 1)
    distance = (tokens[:, None] - centers[None, :]) ** 2
    curve = 1.0 - 0.85 * np.exp(-0.5 * ((np.linspace(-1, 1, length)) / 0.12) ** 2)
    content = np.exp(-0.5 * ((tokens[:, None] - length / 2) / 6.0) ** 2)
    content = content * (0.35 + 0.65 * np.linspace(1.0, 0.4, rank))

    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), dpi=120)
    axes[0].plot(curve, color="#e0a15a", lw=2.0)
    mask_image = axes[1].imshow(np.ones((rank, length)), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    weight_image = axes[2].imshow(np.ones((rank, length)), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    _dress(figure, axes, "Locality", ("value", "latent unit", "latent unit"), ("Input", "Locality mask", "Masked dynamic weights"))

    def update(frame: int) -> None:
        spread = (0.25 + 2.2 * (0.5 - 0.5 * np.cos(2 * np.pi * frame / 48))) * length
        mask = np.exp(-distance / (2.0 * spread**2))
        mask_image.set_data(mask.T)
        weight_image.set_data((mask * content).T)

    _save(figure, path, update)


def _causality(path: Path) -> None:
    length, rank = 80, 8
    tokens = np.arange(length)
    centers = np.arange(rank) * (length - 1) / (rank - 1)
    past = tokens[:, None] <= np.round(centers)[None, :]
    curve = 1.0 - 0.85 * np.exp(-0.5 * ((np.linspace(-1, 1, length)) / 0.12) ** 2)
    content = np.exp(-0.5 * ((tokens[:, None] - length / 2) / 7.0) ** 2)

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=120)
    axes[0].plot(curve, color="#e0a15a", lw=2.0)
    image = axes[1].imshow(np.ones((rank, length)), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    _dress(figure, axes, "Learnable causality", ("value", "latent unit"), ("Input", "Future weight 1.00"))

    def update(frame: int) -> None:
        phi = frame / 47
        mask = np.where(past, 1.0, 1.0 - phi)
        image.set_data((mask * content).T)
        axes[1].set_title(f"Future weight {1 - phi:.2f}", color=INK, fontsize=11)

    _save(figure, path, update)


def _location(path: Path) -> None:
    length, rank = 64, 8
    kernel = np.exp(-0.5 * ((np.arange(rank) - 2.0) / 1.05) ** 2)
    shifts = np.arange(length)
    shared = np.stack([np.roll(kernel, int(round(shift * (rank - 1) / (length - 1))) - 2) for shift in shifts])
    rng = np.random.default_rng(0)
    independent = np.clip(rng.normal(0.35, 0.22, size=shared.shape), 0, 1)

    figure, axis = plt.subplots(figsize=(7.4, 3.4), dpi=120)
    image = axis.imshow(shared.T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    _dress(figure, [axis], "Shared-shifted location invariance", ("latent unit",), ("Independent weight 0.00",))

    def update(frame: int) -> None:
        gate = frame / 47
        image.set_data(((1 - gate) * shared + gate * independent).T)
        axis.set_title(f"Independent weight {gate:.2f}", color=INK, fontsize=11)

    _save(figure, path, update)


def _dress(figure, axes, heading, ylabels, titles) -> None:
    figure.suptitle(heading, color=INK, fontsize=13)
    figure.patch.set_facecolor(BG)
    for axis, ylabel, subtitle in zip(axes, ylabels, titles):
        axis.set_facecolor(PANEL)
        axis.set_title(subtitle, color=INK, fontsize=11)
        axis.set_xlabel("time", color=MUTED)
        axis.set_ylabel(ylabel, color=MUTED)
        axis.tick_params(colors=MUTED, labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#2c3644")
    figure.tight_layout()


def _save(figure, path: Path, update) -> None:
    writer = PillowWriter(fps=12)
    with writer.saving(figure, str(path), dpi=120):
        for frame in range(48):
            update(frame)
            writer.grab_frame()
    plt.close(figure)


if __name__ == "__main__":
    import sys

    names = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    sheet_only = "--sheet" in sys.argv
    catalog = {
        "dials": _dials,
        "scaling": _scaling,
        "fields": _fields,
        "block": _block,
        "locality": _locality,
        "causality": _causality,
        "location": _location,
    }
    if not names:
        main()
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        preview = Path(__file__).resolve().parents[2] / "docs" / "figures"
        for name in names:
            target = preview / f"{name}.gif"
            kwargs = {}
            if name in ("dials", "scaling", "fields"):
                kwargs = {"sheet": preview / f"_{name}_sheet.png", "write": not sheet_only}
            catalog[name](target, **kwargs)
