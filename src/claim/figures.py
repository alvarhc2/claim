"""Draw the animations used in the README.

The frames are schematic. They show the three masks on a short series with a
dip in the middle, which is the shape of a transit or a heartbeat peak once
the time axis is centered.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

ROOT = Path(__file__).resolve().parents[2] / "docs" / "figures"
INK = "#e8edf2"
MUTED = "#93a0b0"
BG = "#10141b"
PANEL = "#181e28"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    _block(ROOT / "block.gif")
    _causality(ROOT / "causality.gif")
    _location(ROOT / "location.gif")
    print(f"wrote {ROOT}")


def _block(path: Path) -> None:
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
    _dress(figure, axes, "Content-modulated mixing", ("value", "latent unit", "latent unit"), ("Input", "Locality mask", "Masked dynamic weights"))

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
    main()
