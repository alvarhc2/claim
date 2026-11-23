"""Published dataset layouts, widths and training schedules.

The numbers are the ones in the manuscript and its appendix. A Kepler
Transformer reads a global view of 1,000 bins because the quadratic attention
map did not fit the GPU at 2,001 bins. CLAIM and MLP-Mixer keep all 2,001.
"""

from __future__ import annotations

from dataclasses import dataclass

DATASETS = ("ecg200", "kepler", "tess", "icu")
MODELS = ("claim", "mixer", "transformer")
FAMILIES = ("amg", "car", "cf4", "pap", "pol", "qui")
FAMILY_FEATURES = {"amg": 36, "car": 18, "cf4": 67, "pap": 24, "pol": 55, "qui": 49}


@dataclass(frozen=True)
class Schedule:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    kind: str
    patience: int | None
    weighted: bool
    temporal_decay: float
    warmup_epochs: int = 0
    step_every: int = 30
    restart_epochs: int = 20
    restart_mult: int = 2


def columns(dataset: str, model: str) -> tuple[list[dict], str, int]:
    """Column specs, embedding kind, and number of extra scalar features."""

    if dataset == "ecg200":
        specs = [_ecg(model)]
        return specs, "linear", 0
    if dataset == "kepler":
        return [_kepler_global(model), _kepler_local(model)], "conv", 0
    if dataset == "tess":
        return [_tess_global(model), _tess_local(model), _tess_secondary(model)], "conv", 1
    if dataset == "icu":
        return [_icu(model)], "linear", 0
    raise ValueError(f"Unknown dataset {dataset!r}.")


def schedule(dataset: str, model: str) -> Schedule:
    if dataset == "ecg200":
        return Schedule(1000, 16, 5e-4, 1e-4, "cosine-restart", 150, True, 0.0, restart_epochs=100)
    if dataset == "kepler":
        batch = 8 if model == "transformer" else 64
        return Schedule(300, batch, 2e-3, 2e-4, "warmup-step", 40, False, 0.0, warmup_epochs=10, step_every=30)
    if dataset == "tess":
        return Schedule(300, 64, 1e-3, 2e-4, "cosine-restart", 50, True, 0.0, restart_epochs=20)
    if dataset == "icu":
        return Schedule(100, 32, 1e-3, 0.01, "cosine", None, True, 1e-5)
    raise ValueError(f"Unknown dataset {dataset!r}.")


def _column(length: int, layers: int, width: int, rank: int, model: str, heads: int, mlp: int) -> dict:
    return {
        "length": length,
        "layers": layers,
        "width": width if model != "transformer" else rank,
        "rank": rank,
        "token_mlp": rank,
        "channel_mlp": rank,
        "heads": heads,
        "mlp": mlp,
    }


def _ecg(model: str) -> dict:
    # The Transformer width is the rank used by CLAIM, not the Mixer width.
    return _column(96, 4, 48, 70, model, heads=5, mlp=140)


def _kepler_global(model: str) -> dict:
    length = 1000 if model == "transformer" else 2001
    return _column(length, 3, 48, 96, model, heads=8, mlp=192)


def _kepler_local(model: str) -> dict:
    return _column(201, 3, 48, 48, model, heads=6, mlp=96)


def _tess_global(model: str) -> dict:
    return _column(201, 2, 32, 32, model, heads=4, mlp=64)


def _tess_local(model: str) -> dict:
    return _column(61, 2, 32, 16, model, heads=2, mlp=32)


def _tess_secondary(model: str) -> dict:
    return _column(61, 2, 32, 16, model, heads=2, mlp=32)


def _icu(model: str) -> dict:
    return _column(30, 2, 32, 16, model, heads=4, mlp=32)
