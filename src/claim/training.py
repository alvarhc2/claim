"""Optimization loop for one initialization.

Early stopping keeps the weights with the highest validation accuracy and
restores them before the test pass. The ICU schedule has no validation, so
the weights after the last epoch are the ones that are scored.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from claim.metrics import classification
from claim.published import Schedule


@dataclass
class FitResult:
    metrics: dict[str, float]
    labels: np.ndarray
    predicted: np.ndarray
    probabilities: np.ndarray
    epochs: int
    epoch_seconds: float
    inference_ms: float
    knobs: list[dict[str, float]]


def fit(
    model: nn.Module,
    views: dict[str, np.ndarray],
    labels: np.ndarray,
    schedule: Schedule,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    test_idx: np.ndarray,
    depth: np.ndarray | None,
    weights: np.ndarray | None,
    device: torch.device,
    status,
    section: str,
    short: str,
) -> FitResult:
    model.to(device)
    train_loader = _loader(views, labels, train_idx, depth, weights, schedule.batch_size, shuffle=weights is None)
    val_loader = None
    if val_idx is not None and len(val_idx):
        val_loader = _loader(views, labels, val_idx, depth, None, schedule.batch_size, shuffle=False)
    test_loader = _loader(views, labels, test_idx, depth, None, schedule.batch_size, shuffle=False)

    class_weights = None
    if schedule.weighted:
        class_weights = _class_weights(labels[train_idx]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=schedule.learning_rate,
        weight_decay=schedule.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = _scheduler(optimizer, schedule)
    best_accuracy = -1.0
    best_state = None
    waited = 0
    epoch_times = []

    for epoch in range(schedule.epochs):
        started = time.perf_counter()
        _epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        epoch_times.append(time.perf_counter() - started)
        detail = f"{epoch_times[-1]:.2f}s"
        if val_loader is not None:
            accuracy = _accuracy(model, val_loader, device)
            detail = f"val {accuracy:.3f}"
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                waited = 0
            else:
                waited += 1
            if schedule.patience is not None and waited >= schedule.patience:
                status.update(section, short, epoch + 1, schedule.epochs, "stop")
                break
        status.update(section, short, epoch + 1, schedule.epochs, detail)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    scores, labels, predicted, probabilities, inference_ms = _evaluate(model, test_loader, device)
    return FitResult(
        scores, labels, predicted, probabilities, len(epoch_times), float(np.mean(epoch_times)), inference_ms, _knobs(model)
    )


def _loader(views, labels, indices, depth, weights, batch_size, shuffle):
    tensors = []
    for name in views:
        array = np.asarray(views[name][indices], dtype=np.float32)
        if array.ndim == 2:
            array = array[:, :, None]
        tensors.append(torch.as_tensor(array))
    tensors.append(torch.as_tensor(labels[indices].astype(np.int64)))
    if depth is not None:
        tensors.insert(-1, torch.as_tensor(depth[indices], dtype=torch.float32).view(-1, 1))
    sample_weights = None
    if weights is not None:
        chosen = np.asarray(weights)[indices]
        sample_weights = torch.as_tensor(chosen, dtype=torch.float64)
        tensors.append(torch.as_tensor(chosen, dtype=torch.float32))
    dataset = TensorDataset(*tensors)
    if sample_weights is not None:
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _epoch(model, loader, criterion, optimizer, device) -> None:
    model.train()
    for batch in loader:
        moved = [tensor.to(device) for tensor in batch]
        views, scalar, labels, sample_weights = _unpack(moved)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(views, scalar), labels)
        if sample_weights is not None:
            loss = (loss * sample_weights).mean()
        else:
            loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def sample_weight_batch(batch) -> bool:
    # Weights are a vector and sit after the class labels, which are also a vector.
    return len(batch) >= 3 and batch[-1].ndim == 1 and batch[-2].ndim == 1 and batch[-1].dtype.is_floating_point


def _accuracy(model, loader, device) -> float:
    labels, predicted, _ = _predict(model, loader, device)
    if len(labels) == 0:
        return 0.0
    return float(np.mean(predicted == labels))


def _evaluate(model, loader, device):
    if hasattr(model, "fold_inference"):
        model.fold_inference()
    started = time.perf_counter()
    labels, predicted, probabilities = _predict(model, loader, device)
    elapsed = time.perf_counter() - started
    per_example = 0.0 if len(labels) == 0 else 1000.0 * elapsed / len(labels)
    return classification(labels, predicted, probabilities), labels, predicted, probabilities, per_example


def _predict(model, loader, device):
    model.eval()
    labels, predicted, probabilities = [], [], []
    with torch.no_grad():
        for batch in loader:
            views, scalar, target, _weights = _unpack([tensor.to(device) for tensor in batch])
            logits = model(views, scalar)
            labels.append(target.cpu().numpy())
            predicted.append(torch.argmax(logits, dim=-1).cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    if not labels:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float64)
    return np.concatenate(labels), np.concatenate(predicted), np.concatenate(probabilities)


def _unpack(batch):
    """Split a collated batch into views, an optional scalar, labels and weights."""

    sample_weights = None
    if sample_weight_batch(batch):
        sample_weights = batch[-1]
        labels = batch[-2]
        views = list(batch[:-2])
    else:
        labels = batch[-1]
        views = list(batch[:-1])
    scalar = None
    if views and views[-1].ndim == 2:
        scalar = views.pop()
    return views, scalar, labels, sample_weights


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=2).astype(np.float64)
    weights = len(labels) / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def _scheduler(optimizer, schedule: Schedule):
    if schedule.kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=schedule.epochs, eta_min=1e-6
        )
    if schedule.kind == "cosine-restart":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=schedule.restart_epochs,
            T_mult=schedule.restart_mult,
            eta_min=1e-6,
        )
    if schedule.kind == "warmup-step":
        warmup = max(schedule.warmup_epochs, 1)
        step = schedule.step_every

        def scale(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / warmup
            return 0.5 ** ((epoch - warmup) // step)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    raise ValueError(f"Unknown schedule {schedule.kind!r}.")


def _knobs(model) -> list[dict[str, float]]:
    found = []
    for module in model.modules():
        if hasattr(module, "knobs") and module.__class__.__name__ == "ClaimBlock":
            found.append(module.knobs())
    return found
