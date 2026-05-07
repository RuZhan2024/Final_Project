#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-adapted CTR-GCN for pose-based fall detection.

This module intentionally defines a separate graph-model family instead of
mutating the existing custom-GCN line.

Project contract:
- external input: ``[B, T, V, F]``
- internal layout: ``[B, F, T, V]``
- single-stream and joint/bone two-stream variants
- fixed MediaPipe skeleton adjacency as the shared prior
- topology refinement through:
  1) a shared input-conditioned relation term
  2) a channel-wise low-rank refinement term

This is a project-adapted CTR-GCN line, not a faithful reproduction of the full
official CTR-GCN training stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn


def build_mediapipe_adjacency(num_joints: int = 33) -> np.ndarray:
    edges = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
        (15, 17), (15, 19), (15, 21),
        (16, 18), (16, 20), (16, 22),
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (0, 9), (0, 10),
    ]
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        if 0 <= i < num_joints and 0 <= j < num_joints:
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32)
    A_hat = A + np.eye(A.shape[0], dtype=np.float32)
    D = np.sum(A_hat, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-8)))
    return (D_inv_sqrt @ A_hat @ D_inv_sqrt).astype(np.float32)


class CTRGraphConv(nn.Module):
    """Graph convolution with shared and channel-wise topology refinement."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        num_joints: int,
        rel_channels: int = 8,
        ctr_rank: int = 8,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.out_ch = int(out_ch)
        self.base_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.theta = nn.Conv2d(in_ch, rel_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_ch, rel_channels, kernel_size=1)
        self.rel_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.alpha_shared = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.ctr_u = nn.Parameter(torch.zeros((out_ch, num_joints, ctr_rank), dtype=torch.float32))
        self.ctr_v = nn.Parameter(torch.zeros((out_ch, ctr_rank, num_joints), dtype=torch.float32))
        self.alpha_ctr = nn.Parameter(torch.zeros((out_ch,), dtype=torch.float32))
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        nn.init.xavier_uniform_(self.ctr_u)
        nn.init.xavier_uniform_(self.ctr_v)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,V]
        base = torch.einsum("vw,bctw->bctv", A_hat, self.base_proj(x))
        theta = self.theta(x).mean(dim=2)  # [B,R,V]
        phi = self.phi(x).mean(dim=2)      # [B,R,V]
        scale = float(max(1, theta.shape[1])) ** 0.5
        rel = torch.softmax(torch.einsum("brv,brw->bvw", theta, phi) / scale, dim=-1)
        x_rel = self.rel_proj(x)
        shared = torch.einsum("bvw,bctw->bctv", rel, x_rel)
        A_ctr = torch.matmul(self.ctr_u, self.ctr_v)  # [C,V,V]
        A_ctr = torch.softmax(A_ctr, dim=-1)
        ctr = torch.einsum("cvw,bctw->bctv", A_ctr, x_rel)
        y = base + self.alpha_shared * shared + self.alpha_ctr.view(1, self.out_ch, 1, 1) * ctr
        return self.drop(self.act(self.bn(y)))


class CTRGCNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        num_joints: int,
        rel_channels: int = 8,
        ctr_rank: int = 8,
        temporal_kernel: int = 9,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        pad = temporal_kernel // 2
        self.graph = CTRGraphConv(
            in_ch,
            out_ch,
            num_joints=num_joints,
            rel_channels=rel_channels,
            ctr_rank=ctr_rank,
            dropout=dropout,
        )
        self.temporal = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(temporal_kernel, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        )
        self.residual = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch),
            )
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        y = self.graph(x, A_hat)
        y = self.temporal(y)
        return self.act(y + self.residual(x))


class CTRGCNEncoder(nn.Module):
    """CTR-GCN encoder used by late-fusion wrappers.

    This intentionally lives beside ``CTRGCN`` instead of replacing it, so
    historical single-stream checkpoints keep their original state-dict keys.
    """

    def __init__(
        self,
        num_joints: int,
        in_feats: int,
        *,
        channels: tuple[int, ...] = (64, 64, 64, 128),
        rel_channels: int = 8,
        ctr_rank: int = 8,
        temporal_kernel: int = 9,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        A_hat = normalize_adjacency(build_mediapipe_adjacency(num_joints))
        self.register_buffer("A_hat", torch.from_numpy(A_hat.astype(np.float32)))

        widths = tuple(int(ch) for ch in channels) or (64, 64, 64, 128)
        blocks = []
        in_ch = int(in_feats)
        for out_ch in widths:
            blocks.append(
                CTRGCNBlock(
                    in_ch,
                    out_ch,
                    num_joints=num_joints,
                    rel_channels=rel_channels,
                    ctr_rank=ctr_rank,
                    temporal_kernel=temporal_kernel,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = int(in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,F,T,V]
        A_hat = self.A_hat
        for blk in self.blocks:
            x = blk(x, A_hat)
        return x.mean(dim=(2, 3))


class CTRGCN(nn.Module):
    """Single-stream CTR-GCN over canonical ``[B,T,V,F]`` skeleton features."""

    def __init__(
        self,
        num_joints: int,
        in_feats: int,
        *,
        channels: tuple[int, ...] = (64, 64, 64, 128),
        rel_channels: int = 8,
        ctr_rank: int = 8,
        temporal_kernel: int = 9,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        A_hat = normalize_adjacency(build_mediapipe_adjacency(num_joints))
        self.register_buffer("A_hat", torch.from_numpy(A_hat.astype(np.float32)))

        widths = tuple(int(ch) for ch in channels) or (64, 64, 64, 128)
        blocks = []
        in_ch = int(in_feats)
        for out_ch in widths:
            blocks.append(
                CTRGCNBlock(
                    in_ch,
                    out_ch,
                    num_joints=num_joints,
                    rel_channels=rel_channels,
                    ctr_rank=ctr_rank,
                    temporal_kernel=temporal_kernel,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,F,T,V]
        A_hat = self.A_hat
        for blk in self.blocks:
            x = blk(x, A_hat)
        x = x.mean(dim=(2, 3))
        return self.head(x).squeeze(-1)


class TwoStreamCTRGCN(nn.Module):
    """Joint/bone CTR-GCN with late fusion at the classifier head."""

    def __init__(
        self,
        num_joints: int,
        in_feats_j: int,
        in_feats_b: int,
        *,
        channels: tuple[int, ...] = (64, 64, 64, 128),
        rel_channels: int = 8,
        ctr_rank: int = 8,
        temporal_kernel: int = 9,
        dropout: float = 0.30,
        fuse: str = "concat",
        stream_drop_joint_p: float = 0.0,
        stream_drop_bone_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.j_enc = CTRGCNEncoder(
            num_joints,
            in_feats_j,
            channels=channels,
            rel_channels=rel_channels,
            ctr_rank=ctr_rank,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
        )
        self.b_enc = CTRGCNEncoder(
            num_joints,
            in_feats_b,
            channels=channels,
            rel_channels=rel_channels,
            ctr_rank=ctr_rank,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
        )
        self.fuse = str(fuse).lower()
        self.stream_drop_joint_p = float(stream_drop_joint_p)
        self.stream_drop_bone_p = float(stream_drop_bone_p)
        out_dim = (2 * self.j_enc.out_dim) if self.fuse == "concat" else self.j_enc.out_dim
        self.head = nn.Linear(out_dim, 1)

    def forward(self, xj: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        zj = self.j_enc(xj)
        zb = self.b_enc(xb)
        if self.training and (self.stream_drop_joint_p > 0.0 or self.stream_drop_bone_p > 0.0):
            bsz = int(zj.shape[0])
            keep_j = (torch.rand(bsz, device=zj.device) >= self.stream_drop_joint_p).to(zj.dtype).view(bsz, 1)
            keep_b = (torch.rand(bsz, device=zb.device) >= self.stream_drop_bone_p).to(zb.dtype).view(bsz, 1)
            both_off = (keep_j <= 0.0) & (keep_b <= 0.0)
            if both_off.any():
                keep_j[both_off] = 1.0
            zj = zj * keep_j
            zb = zb * keep_b
        if self.fuse == "sum":
            z = zj + zb
        elif self.fuse == "joint_only":
            z = zj
        elif self.fuse in {"bone_only", "motion_only"}:
            z = zb
        else:
            z = torch.cat([zj, zb], dim=-1)
        return self.head(z).squeeze(-1)


@dataclass
class CTRGCNConfig:
    num_joints: int = 33
    channels: tuple[int, ...] = (64, 64, 64, 128)
    rel_channels: int = 8
    ctr_rank: int = 8
    temporal_kernel: int = 9
    dropout: float = 0.30
    two_stream: bool = False
    stream_mode: str = "joint_bone"
    fuse: str = "concat"
    stream_drop_joint_p: float = 0.0
    stream_drop_bone_p: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CTRGCNConfig":
        if not isinstance(d, dict):
            return CTRGCNConfig()
        channels_raw = d.get("channels", d.get("channel_schedule", None))
        if channels_raw is None:
            base_channels = int(d.get("base_channels", 64))
            num_blocks = int(d.get("num_blocks", 4))
            channels = [base_channels] * max(1, num_blocks)
            channels[-1] = max(channels[-1], int(base_channels * 2))
        elif isinstance(channels_raw, str):
            channels = [int(x.strip()) for x in channels_raw.split(",") if x.strip()]
        else:
            channels = [int(x) for x in channels_raw]
        return CTRGCNConfig(
            num_joints=int(d.get("num_joints", 33)),
            channels=tuple(channels),
            rel_channels=int(d.get("rel_channels", 8)),
            ctr_rank=int(d.get("ctr_rank", 8)),
            temporal_kernel=int(d.get("temporal_kernel", 9)),
            dropout=float(d.get("dropout", 0.30)),
            two_stream=bool(d.get("two_stream", False)),
            stream_mode=str(d.get("stream_mode", "joint_bone")),
            fuse=str(d.get("fuse", "concat")),
            stream_drop_joint_p=float(d.get("stream_drop_joint_p", 0.0)),
            stream_drop_bone_p=float(d.get("stream_drop_bone_p", 0.0)),
        )
