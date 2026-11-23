"""Pack the public arrays.

Kepler is read from the three study pickles (``global.pkl``, ``local.pkl``,
``label.pkl``) and written as compressed shards with the manuscript split.
TESS is copied from an existing ``views.npz``. ECG200 is the two ARFF files.

Example::

    python scripts/export_public_data.py --dest data --ecg path/to/ECG200 --tess path/to/views.npz --kepler path/to/kepler_data --features path/to/feature_lists
"""

from __future__ import annotations

import argparse
import pickle
import random
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack the public CLAIM arrays.")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--ecg", type=Path, required=True, help="Directory with the two ARFF files")
    parser.add_argument("--tess", type=Path, required=True, help="views.npz")
    parser.add_argument("--kepler", type=Path, required=True, help="Directory with global.pkl, local.pkl and label.pkl")
    parser.add_argument("--features", type=Path, required=True, help="Directory with amg_selected_features.txt and the other five lists")
    args = parser.parse_args()
    _ecg(args.ecg, args.dest / "ecg200")
    _features(args.features, args.dest / "icu" / "features")
    _tess(args.tess, args.dest / "tess")
    _kepler(args.kepler, args.dest / "kepler")


def _ecg(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("ECG200_TRAIN.arff", "ECG200_TEST.arff"):
        shutil.copyfile(source / name, target / name)
    print(f"ecg200 -> {target}")


def _features(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("amg", "car", "cf4", "pap", "pol", "qui"):
        shutil.copyfile(source / f"{name}_selected_features.txt", target / f"{name}.txt")
    print(f"feature names -> {target}")


def _tess(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target / "views.npz")
    with np.load(target / "views.npz") as data:
        split = data["split"]
        labels = data["label"]
        print(
            "tess",
            {int(part): int(np.sum(split == part)) for part in (0, 1, 2)},
            "positives",
            {int(part): int(np.sum(labels[split == part] == 1)) for part in (0, 1, 2)},
        )


def _kepler(source: Path, target: Path) -> None:
    with open(source / "global.pkl", "rb") as handle:
        global_raw = pickle.load(handle)
    with open(source / "local.pkl", "rb") as handle:
        local_raw = pickle.load(handle)
    with open(source / "label.pkl", "rb") as handle:
        labels = np.asarray(pickle.load(handle), dtype=np.int8)
    global_view = np.stack([_series(item) for item in global_raw]).astype(np.float32)
    local_view = np.stack([_series(item) for item in local_raw]).astype(np.float32)
    train, val, test = _split(len(labels), 42)
    target.mkdir(parents=True, exist_ok=True)
    cut = len(labels) // 2
    for index, (start, stop) in enumerate(((0, cut), (cut, len(labels)))):
        payload = {
            "global_view": global_view[start:stop],
            "local_view": local_view[start:stop],
            "label": labels[start:stop],
        }
        if index == 0:
            payload["train_idx"] = np.asarray(train, dtype=np.int64)
            payload["val_idx"] = np.asarray(val, dtype=np.int64)
            payload["test_idx"] = np.asarray(test, dtype=np.int64)
        path = target / f"part-{index:02d}.npz"
        np.savez_compressed(path, **payload)
        print(f"{path.name} {path.stat().st_size / 1e6:.1f} MB")
    print(
        "kepler",
        len(labels),
        tuple(global_view.shape),
        "positives",
        int(np.asarray(labels)[train].sum()),
        int(np.asarray(labels)[val].sum()),
        int(np.asarray(labels)[test].sum()),
    )


def _series(item) -> np.ndarray:
    array = np.asarray(item, dtype=np.float32)
    if array.ndim == 2:
        return array[:, 0]
    return array


def _split(n: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    order = list(range(n))
    for index in reversed(range(1, n)):
        swap = rng._randbelow(index + 1)
        order[index], order[swap] = order[swap], order[index]
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


if __name__ == "__main__":
    main()
