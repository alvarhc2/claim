"""The CLAIM block.

A block reads a sequence ``X`` of shape ``(batch, length, channels)`` and
returns a sequence of the same shape. Token mixing is a low-rank product

    Y = X + GELU( (X~^T (M * (X~ P1)) P2 + b)^T )

where ``X~`` is ``X`` after layer normalization, ``P1`` is ``channels x rank``
and ``P2`` is ``rank x length``. The mask ``M`` and the projection ``P2`` are
where the three adaptive scalars act.

``alpha`` sets the width of a Gaussian locality mask, ``beta`` sets how much
future tokens are suppressed, and ``gamma`` interpolates ``P2`` between a
shared circular kernel and independent weights. ``locality="lookup"`` replaces
the exact Gaussian with the same family of masks stored at ``lookup_bins``
spreads and interpolated, which is the training-time optimization. The spreads
always run from ``0`` to ``k * length`` with ``k = 3``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("full", "location-invariance", "base")
LOCALITY = ("exact", "lookup")
LOCALITY_SCALE = 3.0


class ClaimBlock(nn.Module):
    """One isotropic CLAIM block.

    Parameters
    ----------
    length, channels, rank:
        Sequence length ``S``, channel width ``C`` and mixing rank ``r``.
    variant:
        ``full`` uses locality, causality and location invariance.
        ``location-invariance`` keeps only the shared-kernel interpolation.
        ``base`` is the low-rank mixer with independent ``P2``.
    locality:
        ``exact`` evaluates the Gaussian. ``lookup`` interpolates
        ``lookup_bins`` masks whose spreads are ``linspace(0, k S, bins)``.
    """

    def __init__(
        self,
        length: int,
        channels: int,
        rank: int,
        variant: str = "full",
        locality: str = "exact",
        lookup_bins: int = 10,
    ) -> None:
        super().__init__()
        if length < 1 or channels < 1 or rank < 1:
            raise ValueError("length, channels and rank must be positive.")
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}.")
        if locality not in LOCALITY:
            raise ValueError(f"locality must be one of {LOCALITY}.")
        if lookup_bins < 2:
            raise ValueError("lookup_bins must be at least 2.")

        self.length = length
        self.channels = channels
        self.rank = rank
        self.variant = variant
        self.locality = locality
        self.lookup_bins = lookup_bins

        self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()

        # F.linear stores a map as (out, in). Paper P1 is (C, r), so the
        # stored matrix is (r, C). Paper P2 is (r, S), so the stored
        # independent weights are (S, r).
        self.p1 = nn.Parameter(torch.randn(rank, channels) * (1.0 / math.sqrt(channels)))
        self.independent = nn.Parameter(torch.randn(length, rank) * (1.0 / math.sqrt(rank)))
        self.bias = nn.Parameter(torch.zeros(length))

        if variant == "full":
            self.alpha = nn.Parameter(torch.tensor(0.0))
            self.beta = nn.Parameter(torch.tensor(0.0))
        if variant != "base":
            self.gamma = nn.Parameter(torch.tensor(0.0))
            self.shared = nn.Parameter(torch.randn(rank) * (1.0 / math.sqrt(rank)))

        distance, past, shift = _static_geometry(length, rank)
        self.register_buffer("distance", distance)
        self.register_buffer("past", past)
        self.register_buffer("shift", shift)
        if variant == "full" and locality == "lookup":
            spreads = torch.linspace(0.0, LOCALITY_SCALE * length, lookup_bins)
            masks = torch.stack([_gaussian(distance, spread) for spread in spreads])
            self.register_buffer("lookup", masks)

        self.record_weights = False
        self.dynamic_weights: torch.Tensor | None = None
        self._folded_mask: torch.Tensor | None = None
        self._folded_projection: torch.Tensor | None = None

    def train(self, mode: bool = True) -> ClaimBlock:
        super().train(mode)
        if mode:
            self._folded_mask = None
            self._folded_projection = None
        return self

    def fold_inference(self) -> None:
        """Freeze the masks and the projection for evaluation.

        ``alpha``, ``beta`` and ``gamma`` no longer change the forward pass
        until ``train()`` is called. The matrix multiplications stay.
        """

        self.eval()
        with torch.no_grad():
            self._folded_projection = self.projection().detach()
            self._folded_mask = None if self.variant != "full" else self.mask().detach()

    def locality_mask(self) -> torch.Tensor:
        """Gaussian receptive field, shape ``(length, rank)``."""

        if self.locality == "lookup":
            # sigmoid(alpha) selects a position in [0, k S]. Masks are
            # interpolated, which avoids a fresh exponential at every step.
            position = torch.sigmoid(self.alpha) * (self.lookup_bins - 1)
            low = torch.floor(position).long().clamp(0, self.lookup_bins - 1)
            high = torch.ceil(position).long().clamp(0, self.lookup_bins - 1)
            weight = position - low.float()
            return torch.lerp(self.lookup[low], self.lookup[high], weight)
        spread = LOCALITY_SCALE * self.length * torch.sigmoid(self.alpha)
        return _gaussian(self.distance, spread)

    def causality_mask(self) -> torch.Tensor:
        """Past tokens stay at 1. Future tokens are scaled by ``1 - sigmoid(beta)``."""

        future = 1.0 - torch.sigmoid(self.beta)
        return torch.where(self.past, torch.ones_like(self.distance), future.expand_as(self.distance))

    def mask(self) -> torch.Tensor:
        combined = self.locality_mask() * self.causality_mask()
        return combined.clamp_min(1e-12)

    def projection(self) -> torch.Tensor:
        """Return ``P2`` stored as ``(length, rank)``, the transpose of the paper's ``P2``."""

        if self.variant == "base":
            return self.independent
        gate = torch.sigmoid(self.gamma)
        shared = self.shared[self.shift]
        return torch.lerp(shared, self.independent, gate)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[1:] != (self.length, self.channels):
            raise ValueError(
                f"Expected (batch, {self.length}, {self.channels}), got {tuple(sequence.shape)}."
            )
        normalized = self.norm(sequence)
        weights = F.linear(normalized, self.p1)
        if self.record_weights:
            self.dynamic_weights = weights.detach()
        if self._folded_projection is not None:
            if self._folded_mask is not None:
                weights = weights * self._folded_mask
            projection = self._folded_projection
        else:
            if self.variant == "full":
                weights = weights * self.mask()
            projection = self.projection()
        mixed = torch.matmul(normalized.transpose(-2, -1), weights)
        mixed = F.linear(mixed, projection, self.bias).transpose(-2, -1)
        return sequence + self.activation(mixed)

    def knobs(self) -> dict[str, float]:
        """Sigmoid-scaled scalars that are present for this variant."""

        values: dict[str, float] = {}
        if self.variant == "full":
            values["locality"] = float(torch.sigmoid(self.alpha).detach())
            values["causality"] = float(torch.sigmoid(self.beta).detach())
        if self.variant != "base":
            values["independent"] = float(torch.sigmoid(self.gamma).detach())
        return values


def _gaussian(distance: torch.Tensor, spread: torch.Tensor) -> torch.Tensor:
    return torch.exp(-distance / (2.0 * spread**2 + 1e-12))


def _static_geometry(length: int, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Centers, past/future flags and circular shifts. None of these are trained."""

    tokens = torch.arange(length, dtype=torch.float32)
    units = torch.arange(rank, dtype=torch.float32)
    if rank == 1:
        centers = torch.tensor([(length - 1) / 2.0])
    else:
        centers = units * (length - 1) / (rank - 1)
    distance = (tokens.unsqueeze(1) - centers.unsqueeze(0)) ** 2
    past = tokens.unsqueeze(1) <= torch.round(centers).unsqueeze(0)

    if length == 1:
        output_centers = torch.zeros(1)
    else:
        output_centers = tokens * (rank - 1) / (length - 1)
    shift = (units.unsqueeze(0) - torch.round(output_centers).unsqueeze(1).long()) % rank
    return distance, past, shift.long()
