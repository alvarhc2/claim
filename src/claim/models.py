"""CLAIM, the MLP-Mixer baseline and the Temporal Transformer baseline.

Every network is a stack of isotropic columns. A medical series uses one
column and a linear embedding. Kepler uses a global column and a local
column. TESS adds a secondary column and a depth-change scalar. Astronomical
columns embed with two causal-length-preserving convolutions. Columns of
CLAIM and MLP-Mixer share a channel width, so their outputs are concatenated
on the time axis and pooled. Transformer columns can have different widths,
so each column is pooled before they are combined.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from claim.block import ClaimBlock


class _ConvEmbed(nn.Module):
    def __init__(self, in_features: int, width: int) -> None:
        super().__init__()
        mid = width // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_features, mid, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Conv1d(mid, width, kernel_size=3, padding=1),
            nn.BatchNorm1d(width),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.net(sequence.transpose(1, 2)).transpose(1, 2)


class _LinearEmbed(nn.Module):
    def __init__(self, in_features: int, width: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_features, width)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.net(sequence)


def _embed(kind: str, in_features: int, width: int) -> nn.Module:
    if kind == "conv":
        return _ConvEmbed(in_features, width)
    if kind == "linear":
        return _LinearEmbed(in_features, width)
    raise ValueError(f"Unknown embedding {kind!r}.")


class _MixerBlock(nn.Module):
    """Token MLP, then channel MLP, each with a residual connection."""

    def __init__(self, length: int, channels: int, token_mlp: int, channel_mlp: int) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(channels)
        self.token = nn.Sequential(
            nn.Linear(length, token_mlp),
            nn.GELU(),
            nn.Linear(token_mlp, length),
        )
        self.channel_norm = nn.LayerNorm(channels)
        self.channel = nn.Sequential(
            nn.Linear(channels, channel_mlp),
            nn.GELU(),
            nn.Linear(channel_mlp, channels),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        mixed = self.token_norm(sequence).transpose(1, 2)
        sequence = sequence + self.token(mixed).transpose(1, 2)
        return sequence + self.channel(self.channel_norm(sequence))


class _Attention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"Width {width} is not divisible by {heads} heads.")
        self.heads = heads
        self.head_dim = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)
        for layer in (self.query, self.key, self.value, self.out):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, length, width = sequence.shape
        def split(layer: nn.Linear) -> torch.Tensor:
            projected = layer(sequence)
            return projected.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(split(self.query), split(self.key).transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        weights = self.drop(torch.softmax(scores, dim=-1))
        mixed = torch.matmul(weights, split(self.value))
        mixed = mixed.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out(mixed)


class _EncoderLayer(nn.Module):
    """Post-norm Transformer layer: residual, then layer norm."""

    def __init__(self, width: int, heads: int, mlp: int, dropout: float) -> None:
        super().__init__()
        self.attention = _Attention(width, heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(width, mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp, width),
            nn.Dropout(dropout),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[3].weight)
        nn.init.zeros_(self.mlp[3].bias)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence = self.norm1(sequence + self.attention(sequence))
        return self.norm2(sequence + self.mlp(sequence))


class _Positions(nn.Module):
    def __init__(self, width: int, length: int) -> None:
        super().__init__()
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
        encoding = torch.zeros(1, length, width)
        encoding[0, :, 0::2] = torch.sin(position * div)
        encoding[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("encoding", encoding)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence + self.encoding


class ClaimNet(nn.Module):
    """One or more CLAIM columns, global average pooling and a linear head."""

    def __init__(
        self,
        in_features: int,
        columns: list[dict],
        num_classes: int = 2,
        embed: str = "linear",
        scalar_features: int = 0,
        variant: str = "full",
        locality: str = "exact",
        lookup_bins: int = 10,
    ) -> None:
        super().__init__()
        width = columns[0]["width"]
        if any(column["width"] != width for column in columns):
            raise ValueError("CLAIM columns share one channel width.")
        self.columns = nn.ModuleList()
        for column in columns:
            blocks = [
                ClaimBlock(
                    column["length"],
                    width,
                    column["rank"],
                    variant=variant,
                    locality=locality,
                    lookup_bins=lookup_bins,
                )
                for _ in range(column["layers"])
            ]
            self.columns.append(nn.ModuleDict({
                "embed": _embed(embed, in_features, width),
                "blocks": nn.ModuleList(blocks),
            }))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width + scalar_features, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.scalar_features = scalar_features

    def fold_inference(self) -> None:
        for column in self.columns:
            for block in column["blocks"]:
                block.fold_inference()

    def record(self, enabled: bool) -> None:
        for column in self.columns:
            for block in column["blocks"]:
                block.record_weights = enabled

    def forward(self, views: list[torch.Tensor], scalar: torch.Tensor | None = None) -> torch.Tensor:
        encoded = []
        for column, view in zip(self.columns, views):
            hidden = column["embed"](view)
            for block in column["blocks"]:
                hidden = block(hidden)
            encoded.append(self.norm(hidden))
        pooled = torch.cat(encoded, dim=1).mean(dim=1)
        if self.scalar_features:
            pooled = torch.cat([pooled, scalar], dim=-1)
        return self.head(pooled)


class MixerNet(nn.Module):
    """MLP-Mixer with the same columns, embedding and pooling as CLAIM."""

    def __init__(
        self,
        in_features: int,
        columns: list[dict],
        num_classes: int = 2,
        embed: str = "linear",
        scalar_features: int = 0,
    ) -> None:
        super().__init__()
        width = columns[0]["width"]
        self.columns = nn.ModuleList()
        for column in columns:
            blocks = [
                _MixerBlock(column["length"], width, column["token_mlp"], column["channel_mlp"])
                for _ in range(column["layers"])
            ]
            self.columns.append(nn.ModuleDict({
                "embed": _embed(embed, in_features, width),
                "blocks": nn.ModuleList(blocks),
            }))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width + scalar_features, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.scalar_features = scalar_features

    def forward(self, views: list[torch.Tensor], scalar: torch.Tensor | None = None) -> torch.Tensor:
        encoded = []
        for column, view in zip(self.columns, views):
            hidden = column["embed"](view)
            for block in column["blocks"]:
                hidden = block(hidden)
            encoded.append(self.norm(hidden))
        pooled = torch.cat(encoded, dim=1).mean(dim=1)
        if self.scalar_features:
            pooled = torch.cat([pooled, scalar], dim=-1)
        return self.head(pooled)


class TransformerNet(nn.Module):
    """Temporal Transformer with fixed sinusoids and post-norm encoder layers.

    A single column uses the two-layer head of the ECG baseline. Several
    columns are pooled separately, projected back to the first column's width,
    and, on TESS, concatenated with the depth change before a linear head.
    """

    def __init__(
        self,
        in_features: int,
        columns: list[dict],
        num_classes: int = 2,
        embed: str = "linear",
        scalar_features: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.columns = nn.ModuleList()
        for column in columns:
            layers = [
                _EncoderLayer(column["width"], column["heads"], column["mlp"], dropout)
                for _ in range(column["layers"])
            ]
            embed_layer = _embed(embed, in_features, column["width"])
            if isinstance(embed_layer, _LinearEmbed):
                nn.init.xavier_uniform_(embed_layer.net.weight)
                nn.init.zeros_(embed_layer.net.bias)
            self.columns.append(nn.ModuleDict({
                "embed": embed_layer,
                "positions": _Positions(column["width"], column["length"]),
                "layers": nn.ModuleList(layers),
                "norm": nn.LayerNorm(column["width"]),
            }))
        widths = [column["width"] for column in columns]
        self.single = len(columns) == 1
        self.scalar_features = scalar_features
        if self.single:
            width = widths[0]
            self.combine = None
            self.head = nn.Sequential(
                nn.Linear(width, width // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width // 2, num_classes),
            )
            nn.init.xavier_uniform_(self.head[0].weight)
            nn.init.zeros_(self.head[0].bias)
            nn.init.zeros_(self.head[3].weight)
            nn.init.zeros_(self.head[3].bias)
        else:
            self.combine = nn.Linear(sum(widths), widths[0])
            nn.init.xavier_uniform_(self.combine.weight)
            nn.init.zeros_(self.combine.bias)
            self.head = nn.Linear(widths[0] + scalar_features, num_classes)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, views: list[torch.Tensor], scalar: torch.Tensor | None = None) -> torch.Tensor:
        pooled = []
        for column, view in zip(self.columns, views):
            hidden = column["positions"](column["embed"](view))
            for layer in column["layers"]:
                hidden = layer(hidden)
            pooled.append(column["norm"](hidden).mean(dim=1))
        if self.single:
            return self.head(pooled[0])
        hidden = self.combine(torch.cat(pooled, dim=-1))
        if self.scalar_features:
            hidden = torch.cat([hidden, scalar], dim=-1)
        return self.head(hidden)


def build(
    model: str,
    in_features: int,
    columns: list[dict],
    embed: str,
    scalar_features: int,
    variant: str = "full",
    locality: str = "exact",
    lookup_bins: int = 10,
) -> nn.Module:
    """Construct one of the three published networks."""

    if model == "claim":
        return ClaimNet(
            in_features,
            columns,
            embed=embed,
            scalar_features=scalar_features,
            variant=variant,
            locality=locality,
            lookup_bins=lookup_bins,
        )
    if model == "mixer":
        return MixerNet(in_features, columns, embed=embed, scalar_features=scalar_features)
    if model == "transformer":
        return TransformerNet(in_features, columns, embed=embed, scalar_features=scalar_features)
    raise ValueError(f"Unknown model {model!r}.")
