"""Command line for the published CLAIM experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from claim.console import Status, enable_color, line, title
from claim.data import downsample, load_ecg200, load_kepler, load_tess, rolling_windows
from claim.metrics import classification, mean_std
from claim.models import build
from claim.published import DATASETS, FAMILIES, MODELS, Schedule, columns, schedule
from claim.synthetic import SMOKE_LENGTHS, icu_cohort, smoke_split
from claim.training import fit

VARIANTS = ("full", "location-invariance", "base")
METRICS = ("accuracy", "sensitivity", "specificity", "precision", "f1", "auc")


def main() -> None:
    args = _parser().parse_args()
    enable_color()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synthetic = args.synthetic or (not args.paper and args.data is None)
    repetitions = args.repetitions if args.repetitions is not None else (10 if args.paper else 1)
    smoke = synthetic and not args.paper
    spec, embed, scalar_features = _architecture(args, smoke)
    plan = _plan(args)
    root = Path(args.data) if args.data else Path(__file__).resolve().parents[2] / "data"

    title("CLAIM")
    line(f"{args.dataset}  {args.model}  {args.variant if args.model == 'claim' else 'baseline'}")
    line(f"device {device.type}  repetitions {repetitions}  epochs {plan.epochs}  batch {plan.batch_size}")
    if args.model == "claim":
        line(f"locality {args.locality}  bins {args.lookup_bins}")
    if smoke:
        line("Smoke run on a generated cohort. Pass --paper to use the published schedule.")
    elif args.dataset == "icu":
        line("ICU uses a synthetic cohort with the published widths and the monthly split.")

    status = Status()
    if args.dataset == "icu":
        rows = _icu(args, spec, embed, scalar_features, plan, repetitions, device, status)
    elif args.dataset == "ecg200" and not smoke:
        rows = _ecg(args, spec, embed, scalar_features, plan, repetitions, device, status, root)
    else:
        rows = _views(args, spec, embed, scalar_features, plan, repetitions, smoke, device, status, root)
    status.close()
    _report(args, rows, Path(args.output))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m claim",
        description="Train CLAIM and the published baselines for limited-data time series classification.",
    )
    parser.add_argument("--paper", action="store_true", help="Published architecture, schedule and 10 repetitions")
    parser.add_argument("--dataset", default="ecg200", choices=DATASETS)
    parser.add_argument("--model", default="claim", choices=MODELS)
    parser.add_argument("--variant", default="full", choices=VARIANTS, help="CLAIM ablation")
    parser.add_argument("--locality", default="exact", choices=("exact", "lookup"))
    parser.add_argument("--lookup-bins", type=int, default=10, help="Masks in the shared lookup, spreads from 0 to 3S")
    parser.add_argument("--synthetic", action="store_true", help="Train on a generated cohort")
    parser.add_argument("--data", default=None, help="Directory with ecg200/, kepler/ and tess/")
    parser.add_argument("--family", default="all", choices=[*FAMILIES, "all"], help="ICU antibiotic family")
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs")
    return parser


def _architecture(args, smoke: bool):
    if not smoke:
        return columns(args.dataset, args.model)
    if args.dataset == "icu":
        lengths = {"series": 30}
    else:
        lengths = SMOKE_LENGTHS[args.dataset]
    specs = []
    for length in lengths.values():
        specs.append({
            "length": length,
            "layers": 1,
            "width": 16,
            "rank": 8,
            "token_mlp": 8,
            "channel_mlp": 8,
            "heads": 4,
            "mlp": 16,
        })
    scalar = 1 if args.dataset == "tess" else 0
    return specs, "linear", scalar


def _plan(args) -> Schedule:
    if args.paper:
        plan = schedule(args.dataset, args.model)
    else:
        plan = Schedule(1, 16, 1e-3, 0.0, "cosine", None, False, 0.0)
    epochs = args.epochs if args.epochs is not None else plan.epochs
    batch = args.batch_size if args.batch_size is not None else plan.batch_size
    learning_rate = args.learning_rate if args.learning_rate is not None else plan.learning_rate
    return replace(plan, epochs=epochs, batch_size=batch, learning_rate=learning_rate)


def _views(args, spec, embed, scalar_features, plan, repetitions, smoke, device, status, root: Path):
    if smoke:
        cohort = smoke_split(args.dataset, seed=args.seed)
        views, labels, depth = cohort.views, cohort.labels, cohort.depth
        splits = [(cohort.train_idx, cohort.val_idx, cohort.test_idx)]
    else:
        cohort = _load_public(args.dataset, root)
        views = {name: values for name, values in cohort.views.items()}
        if args.dataset == "kepler" and args.model == "transformer":
            views["global"] = downsample(views["global"], spec[0]["length"])
        labels, depth = cohort.labels, cohort.depth
        splits = [(cohort.train_idx, cohort.val_idx, cohort.test_idx)]
    rows = []
    in_features = 1
    for repetition, (train_idx, val_idx, test_idx) in enumerate(splits * repetitions):
        _seed(args.seed + repetition)
        model = build(args.model, in_features, spec, embed, scalar_features, args.variant, args.locality, args.lookup_bins)
        result = fit(
            model, views, labels, plan,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, depth=depth, weights=None,
            device=device, status=status, section=args.dataset, short=f"run {repetition + 1}",
        )
        rows.append(_row(result))
    return rows


def _ecg(args, spec, embed, scalar_features, plan, repetitions, device, status, root: Path):
    cohort = load_ecg200(root / "ecg200")
    train = cohort.train_idx
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(train))
    rows = []
    for repetition in range(repetitions):
        fold_rows = []
        for fold, (train_pos, val_pos) in enumerate(folds):
            _seed(args.seed + repetition)
            model = build(
                args.model, 1, spec, embed, scalar_features, args.variant, args.locality, args.lookup_bins
            )
            result = fit(
                model, cohort.views, cohort.labels, plan,
                train_idx=train[train_pos], val_idx=train[val_pos], test_idx=cohort.test_idx,
                depth=None, weights=None, device=device, status=status,
                section=f"{args.dataset}  repetition {repetition + 1}", short=f"fold {fold + 1}",
            )
            fold_rows.append(_row(result))
        rows.append(_average_rows(fold_rows))
    return rows


def _icu(args, spec, embed, scalar_features, plan, repetitions, device, status):
    families = FAMILIES if args.family == "all" else (args.family,)
    rows = []
    for family in families:
        per_month = 12 if args.paper else 4
        end = (2017, 6) if args.paper else (2016, 2)
        for repetition in range(repetitions):
            cohort = icu_cohort(family, per_month=per_month, seed=args.seed + repetition, end=end)
            windows = rolling_windows(cohort, plan.temporal_decay)
            labels, predicted, probabilities = [], [], []
            seconds, inference, knobs = [], [], []
            for index, window in enumerate(windows):
                _seed(args.seed + repetition)
                model = build(
                    args.model, cohort.features, spec, embed, scalar_features,
                    args.variant, args.locality, args.lookup_bins,
                )
                series = np.concatenate([window.x_train, window.x_test])
                target = np.concatenate([window.y_train, window.y_test])
                sample_weights = np.concatenate([window.weights, np.ones(len(window.y_test), dtype=np.float32)])
                n_train = len(window.y_train)
                result = fit(
                    model, {"series": series}, target, plan,
                    train_idx=np.arange(n_train), val_idx=np.array([], dtype=np.int64),
                    test_idx=np.arange(n_train, len(target)), depth=None, weights=sample_weights,
                    device=device, status=status,
                    section=f"{family}  repetition {repetition + 1}", short=f"month {index + 1}",
                )
                labels.append(result.labels)
                predicted.append(result.predicted)
                probabilities.append(result.probabilities)
                seconds.append(result.epoch_seconds)
                inference.append(result.inference_ms)
                knobs.append(result.knobs)
            pooled = classification(
                np.concatenate(labels), np.concatenate(predicted), np.concatenate(probabilities)
            )
            rows.append({
                "family": family,
                **pooled,
                "epochs": float(plan.epochs),
                "epoch_seconds": float(np.mean(seconds)),
                "inference_ms": float(np.mean(inference)),
                "knobs": knobs[-1],
            })
    return rows


def _load_public(dataset: str, root: Path):
    if dataset == "kepler":
        return load_kepler(root / "kepler")
    if dataset == "tess":
        return load_tess(root / "tess")
    if dataset == "ecg200":
        return load_ecg200(root / "ecg200")
    raise ValueError(dataset)


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _row(result) -> dict:
    return {
        **result.metrics,
        "epochs": float(result.epochs),
        "epoch_seconds": result.epoch_seconds,
        "inference_ms": result.inference_ms,
        "knobs": result.knobs,
    }


def _average_rows(rows: list[dict]) -> dict:
    averaged = {}
    for key in (*METRICS, "epochs", "epoch_seconds", "inference_ms"):
        averaged[key] = float(np.nanmean([row[key] for row in rows]))
    averaged["knobs"] = rows[-1]["knobs"]
    return averaged


def _report(args, rows: list[dict], output: Path) -> None:
    title("Result")
    families = [row.get("family") for row in rows]
    grouped = {}
    for row in rows:
        grouped.setdefault(row.get("family", args.dataset), []).append(row)
    lines = []
    for name, group in grouped.items():
        summary = mean_std(group)
        text = "  ".join(f"{key} {_pair(summary[key])}" for key in METRICS)
        label = name if families[0] else args.dataset
        line(f"{label}  {text}")
        lines.append(f"{label}  {text}")
        if group[0].get("knobs"):
            rendered = _render_knobs(group)
            if rendered:
                line(rendered)
                lines.append(rendered)
        line(
            f"epoch {_pair(_mean(group, 'epoch_seconds'))} s   "
            f"inference {_pair(_mean(group, 'inference_ms'))} ms"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    line(f"wrote {output / 'metrics.txt'}")


def _render_knobs(group: list[dict]) -> str:
    knobs = group[-1].get("knobs") or []
    if not any(knobs):
        return ""
    parts = []
    for index, knob in enumerate(knobs):
        values = " ".join(f"{name} {value:.2f}" for name, value in knob.items())
        parts.append(f"block {index} {values}")
    return "  ".join(parts)


def _mean(group, key):
    values = np.array([row[key] for row in group], dtype=np.float64)
    return float(np.nanmean(values)), float(np.nanstd(values))


def _pair(stats: tuple[float, float]) -> str:
    mean, deviation = stats
    if not np.isfinite(mean):
        return "undefined"
    return f"{mean:.4f} ± {deviation:.4f}"
