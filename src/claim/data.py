"""Public ECG200, Kepler and TESS files, and the ICU rolling windows.

Kepler is stored as shards because GitHub rejects a file over 100 MB. The
loader concatenates ``part-*.npz`` in name order. The frozen indices were
drawn with the manuscript's shuffle: ``random.seed(42)`` and Fisher-Yates.
TESS keeps the split that was already stored with the views.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from claim.synthetic import SyntheticICU


@dataclass
class SplitCohort:
    name: str
    views: dict[str, np.ndarray]
    labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    depth: np.ndarray | None = None

    def take(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        chosen = {name: values[indices] for name, values in self.views.items()}
        if self.depth is not None:
            chosen["depth"] = self.depth[indices]
        return chosen


@dataclass
class Window:
    x_train: np.ndarray
    y_train: np.ndarray
    weights: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_period: str
    test_period: str


def load_ecg200(directory: str | Path) -> SplitCohort:
    directory = Path(directory)
    train = _arff(directory / "ECG200_TRAIN.arff")
    test = _arff(directory / "ECG200_TEST.arff")
    views = {"series": np.concatenate([train[0], test[0]], axis=0)}
    labels = np.concatenate([train[1], test[1]])
    n_train = len(train[1])
    # The official test file is held out. Validation is a fold of the train file.
    return SplitCohort(
        "ecg200",
        views,
        labels,
        np.arange(n_train),
        np.array([], dtype=np.int64),
        np.arange(n_train, n_train + len(test[1])),
    )


def load_kepler(directory: str | Path) -> SplitCohort:
    directory = Path(directory)
    shards = sorted(directory.glob("part-*.npz"))
    if not shards:
        raise FileNotFoundError(f"No Kepler shards in {directory}.")
    global_view, local_view, labels = [], [], []
    train_idx = val_idx = test_idx = None
    for shard in shards:
        with np.load(shard) as data:
            global_view.append(np.array(data["global_view"]))
            local_view.append(np.array(data["local_view"]))
            labels.append(np.array(data["label"]))
            if "train_idx" in data:
                train_idx = np.array(data["train_idx"])
                val_idx = np.array(data["val_idx"])
                test_idx = np.array(data["test_idx"])
    if train_idx is None:
        raise FileNotFoundError(f"{directory} has views but no frozen split.")
    return SplitCohort(
        "kepler",
        {"global": np.concatenate(global_view), "local": np.concatenate(local_view)},
        np.concatenate(labels).astype(np.int64),
        train_idx,
        val_idx,
        test_idx,
    )


def load_tess(path: str | Path) -> SplitCohort:
    path = Path(path)
    if path.is_dir():
        path = path / "views.npz"
    with np.load(path) as data:
        split = np.array(data["split"])
        return SplitCohort(
            "tess",
            {
                "global": np.array(data["global_view"]),
                "local": np.array(data["local_view"]),
                "secondary": np.array(data["secondary_view"]),
            },
            np.array(data["label"]).astype(np.int64),
            np.flatnonzero(split == 0),
            np.flatnonzero(split == 1),
            np.flatnonzero(split == 2),
            depth=np.array(data["depth_change"]).astype(np.float32).reshape(-1),
        )


def rolling_windows(cohort: SyntheticICU, decay: float, first_test: int = 201601) -> list[Window]:
    """Expanding train, next-month test, starting at ``first_test``."""

    months = [int(value) for value in np.unique(cohort.year_month)]
    windows = []
    for month in months:
        if month < first_test:
            continue
        train = cohort.year_month < month
        test = cohort.year_month == month
        if not np.any(train) or not np.any(test):
            continue
        last = int(cohort.ordinal[train].max())
        age = (last - cohort.ordinal[train]).astype(np.float64)
        weights = np.exp(-decay * age).astype(np.float32)
        windows.append(
            Window(
                cohort.series[train],
                cohort.labels[train],
                weights,
                cohort.series[test],
                cohort.labels[test],
                _span(cohort.year_month[train]),
                _stamp(month),
            )
        )
    if not windows:
        raise ValueError("The cohort has no month at or after the first test month.")
    return windows


def downsample(view: np.ndarray, length: int) -> np.ndarray:
    """Linear resampling used for the Kepler Transformer global view."""

    if view.shape[1] == length:
        return view
    source = np.arange(view.shape[1])
    grid = np.linspace(0, view.shape[1] - 1, length)
    return np.stack([np.interp(grid, source, row) for row in view]).astype(np.float32)


def _arff(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    data = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if line.lower().startswith("@data"):
            data = True
            continue
        if data:
            rows.append([float(part) for part in line.split(",")])
    table = np.asarray(rows, dtype=np.float32)
    labels = table[:, -1]
    # The file uses -1 and +1. Map them onto 0 and 1.
    labels = ((labels + 1) // 2).astype(np.int64)
    return table[:, :-1], labels


def _stamp(year_month: int) -> str:
    return f"{year_month // 100:04d}-{year_month % 100:02d}"


def _span(year_month: np.ndarray) -> str:
    return f"{_stamp(int(year_month.min()))} to {_stamp(int(year_month.max()))}"
