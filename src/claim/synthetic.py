"""Synthetic series with the layouts of the four studies.

The ICU generator does not imitate hospital records. It builds 30-day
cultures for the six antibiotic families, with the published feature widths,
an expanding monthly split and a resistance signal that is stronger in the
last days of the window. The other generators are short stand-ins so a
command can run without the public files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from claim.published import FAMILY_FEATURES


@dataclass
class SyntheticSplit:
    name: str
    views: dict[str, np.ndarray]
    labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    depth: np.ndarray | None = None


@dataclass
class SyntheticICU:
    family: str
    features: int
    series: np.ndarray
    labels: np.ndarray
    ordinal: np.ndarray
    year_month: np.ndarray


SMOKE_LENGTHS = {
    "ecg200": {"series": 96},
    "kepler": {"global": 64, "local": 32},
    "tess": {"global": 48, "local": 24, "secondary": 24},
}


def smoke_split(name: str, n: int = 48, seed: int = 0) -> SyntheticSplit:
    """A small cohort with the view names of ``name`` and short sequences."""

    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n).astype(np.int64)
    lengths = SMOKE_LENGTHS.get(name, SMOKE_LENGTHS["ecg200"])
    views = {}
    for view, length in lengths.items():
        curve = rng.normal(0.0, 0.2, (n, length)).astype(np.float32)
        center = length // 2
        width = max(2, length // 16)
        grid = np.arange(length)
        bump = np.exp(-0.5 * ((grid - center) / width) ** 2).astype(np.float32)
        curve[labels == 1] -= bump
        views[view] = curve
    depth = None
    if name == "tess":
        depth = (0.2 + 0.6 * labels + rng.normal(0, 0.05, n)).astype(np.float32)
    order = rng.permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    return SyntheticSplit(
        name,
        views,
        labels,
        order[:n_train],
        order[n_train : n_train + n_val],
        order[n_train + n_val :],
        depth,
    )


def icu_cohort(
    family: str,
    *,
    per_month: int = 12,
    seed: int = 0,
    start: tuple[int, int] = (2004, 8),
    end: tuple[int, int] = (2017, 6),
) -> SyntheticICU:
    """Cultures from ``start`` through ``end``, one row per culture."""

    if family not in FAMILY_FEATURES:
        raise ValueError(f"Unknown family {family!r}.")
    features = FAMILY_FEATURES[family]
    months = _months(start, end)
    rng = np.random.default_rng(seed + FAMILY_FEATURES[family])
    series = []
    labels = []
    ordinal = []
    year_month = []
    origin = np.datetime64("2004-01-01")
    for year, month in months:
        for index in range(per_month):
            day = int(rng.integers(1, 28))
            stamp = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}")
            window = rng.normal(0.0, 1.0, (30, features)).astype(np.float32)
            pad = int(rng.integers(0, 8))
            window[:pad] = 0.0
            recent = window[-5:, :3].mean()
            drift = 0.04 * ((year - 2004) * 12 + month)
            logit = 1.4 * recent + drift - 0.2
            label = int(rng.random() < 1.0 / (1.0 + np.exp(-logit)))
            series.append(window)
            labels.append(label)
            ordinal.append(int((stamp - origin) / np.timedelta64(1, "D")))
            year_month.append(year * 100 + month)
            del index
    return SyntheticICU(
        family,
        features,
        np.stack(series),
        np.asarray(labels, dtype=np.int64),
        np.asarray(ordinal, dtype=np.int64),
        np.asarray(year_month, dtype=np.int64),
    )


def _months(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    year, month = start
    last = end[0] * 12 + end[1]
    found = []
    while year * 12 + month <= last:
        found.append((year, month))
        month += 1
        if month == 13:
            month = 1
            year += 1
    return found
