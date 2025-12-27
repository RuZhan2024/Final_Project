#!/usr/bin/env python3
"""Score an unlabeled window directory and estimate alert rate.

Self-contained (no eval_common.py dependency).

Typical use
-----------
python eval/score_unlabeled_alert_rate.py \
  --arch tcn --win_dir data/processed/.../unsplit \
  --ckpt outputs/.../best.pt --thr 0.95

Outputs a JSON summary and prints a short text summary.

Alert-rate definition here is an *estimate* based on window timestamps. If
windows were generated via a subsampling strategy (not dense sliding), the
FA/24h estimate may be biased. For best results, generate evaluation windows
with dense sliding.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    raise SystemExit("[err] PyTorch is required for scoring") from e


# ----------------------------
# Small shared helpers (copy)
# ----------------------------

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def p_fall_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    if logits.ndim == 2 and logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.ndim == 2 and logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.sigmoid(logits.squeeze(-1))


def _list_npz_files(root: str) -> List[str]:
    root = os.path.abspath(root)
    if os.path.isdir(root):
        return sorted(glob.glob(os.path.join(root, "*.npz")))
    return []


def read_window_tcn(
    npz_path: str,
    *,
    center: str = "pelvis",
    use_motion: int = 1,
    use_conf_channel: int = 1,
    motion_scale_by_fps: int = 1,
    conf_gate: float = 0.20,
    use_precomputed_mask: int = 1,
) -> Tuple[np.ndarray, Dict[str, object]]:
    z = np.load(npz_path, allow_pickle=False)
    joints = z["joints"].astype(np.float32)  # [T, V, 2]
    motion = z["motion"].astype(np.float32) if "motion" in z.files else None
    mask = z["mask"].astype(np.float32) if "mask" in z.files else None

    fps = float(z["fps"]) if "fps" in z.files else None
    video_id = str(z["video_id"]) if "video_id" in z.files else None
    w_start = int(z["w_start"]) if "w_start" in z.files else None
    w_end = int(z["w_end"]) if "w_end" in z.files else None

    # center
    if center == "pelvis":
        # MediaPipe pelvis approx: midpoint of left/right hip (23/24)
        lhip = joints[:, 23, :]
        rhip = joints[:, 24, :]
        pelvis = 0.5 * (lhip + rhip)
        joints = joints - pelvis[:, None, :]
        if motion is not None:
            motion = motion

    feats = [joints.reshape(joints.shape[0], -1)]

    if use_motion and motion is not None:
        m = motion
        if motion_scale_by_fps and fps is not None and fps > 0:
            m = m * (fps / 30.0)
        feats.append(m.reshape(m.shape[0], -1))

    if use_conf_channel:
        conf = None
        if "conf" in z.files:
            conf = z["conf"].astype(np.float32)  # [T, V]
        elif "joints_conf" in z.files:
            conf = z["joints_conf"].astype(np.float32)
        if conf is not None:
            conf = conf[:, :, None].repeat(2, axis=2).reshape(conf.shape[0], -1)
            feats.append(conf)

    x = np.concatenate(feats, axis=1).astype(np.float32)  # [T, C]

    if use_precomputed_mask and mask is not None:
        # mask is [T, V]
        m = mask.astype(np.float32)
        m2 = np.repeat(m[:, :, None], 2, axis=2).reshape(m.shape[0], -1)
        x[:, : m2.shape[1]] *= m2

    meta = {
        "video_id": video_id,
        "w_start": w_start,
        "w_end": w_end,
        "fps": fps,
    }
    return x, meta


def read_window_gcn(
    npz_path: str,
    *,
    center: str = "pelvis",
    use_motion: int = 1,
    use_conf_channel: int = 1,
    motion_scale_by_fps: int = 1,
    conf_gate: float = 0.20,
    use_precomputed_mask: int = 1,
) -> Tuple[np.ndarray, Dict[str, object]]:
    z = np.load(npz_path, allow_pickle=False)
    joints = z["joints"].astype(np.float32)  # [T, V, 2]
    motion = z["motion"].astype(np.float32) if "motion" in z.files else None
    mask = z["mask"].astype(np.float32) if "mask" in z.files else None

    fps = float(z["fps"]) if "fps" in z.files else None
    video_id = str(z["video_id"]) if "video_id" in z.files else None
    w_start = int(z["w_start"]) if "w_start" in z.files else None
    w_end = int(z["w_end"]) if "w_end" in z.files else None

    if center == "pelvis":
        lhip = joints[:, 23, :]
        rhip = joints[:, 24, :]
        pelvis = 0.5 * (lhip + rhip)
        joints = joints - pelvis[:, None, :]

    feats = [joints]
    if use_motion and motion is not None:
        m = motion
        if motion_scale_by_fps and fps is not None and fps > 0:
            m = m * (fps / 30.0)
        feats.append(m)
    if use_conf_channel:
        conf = None
        if "conf" in z.files:
            conf = z["conf"].astype(np.float32)
        elif "joints_conf" in z.files:
            conf = z["joints_conf"].astype(np.float32)
        if conf is not None:
            conf = conf[:, :, None]
            feats.append(conf)

    x = np.concatenate(feats, axis=2).astype(np.float32)  # [T, V, F]

    if use_precomputed_mask and mask is not None:
        # mask [T,V] applied to XY channels only
        m = mask.astype(np.float32)[:, :, None]
        x[:, :, :2] *= m

    meta = {
        "video_id": video_id,
        "w_start": w_start,
        "w_end": w_end,
        "fps": fps,
    }
    return x, meta


# ----------------------------
# Minimal model builders (TCN/GCN) for scoring
# ----------------------------

# For scoring-only, we can re-use the same model factory as in eval/metrics.py.
# To avoid circular imports, we implement a small checkpoint-driven loader.


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, dropout: float = 0.0):
        super().__init__()
        pad = (k - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.bn1(self.conv1(x))))
        y = self.drop(self.act(self.bn2(self.conv2(y))))
        return y + self.skip(x)


class _SimpleTCN(nn.Module):
    def __init__(self, in_ch: int, hid: int = 128, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            _TCNBlock(in_ch, hid, k=3, dropout=dropout),
            _TCNBlock(hid, hid, k=3, dropout=dropout),
            _TCNBlock(hid, hid, k=3, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        y = self.net(x)
        return self.head(y).squeeze(-1)

# ---- TCN v2 (matches train_tcn.py checkpoints: conv_in.*, blocks.*, head.*) ----

class _ResTCNv2Block(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3, dilation: int = 1, p: float = 0.30):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.relu(self.bn(self.conv(x))))
        return x + out


class _TCNv2(nn.Module):
    def __init__(self, in_ch: int, hid: int = 128, depth: int = 4, dropout: float = 0.30):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_ch, hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(hid),
            nn.ReLU(inplace=True),
        )
        dilations = [2 ** i for i in range(int(depth))]
        self.blocks = nn.ModuleList([_ResTCNv2Block(hid, kernel_size=3, dilation=d, p=dropout) for d in dilations])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        y = self.conv_in(x)
        for b in self.blocks:
            y = b(y)
        y = self.pool(y).squeeze(-1)  # [B, hid]
        return self.head(y).squeeze(-1)



class _MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GCNUnit(nn.Module):
    def __init__(self, d_in: int, d_out: int, drop: float, use_se: bool = False):
        super().__init__()
        self.mlp = _MLP(d_in, d_out, drop)
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(d_out, max(8, d_out // 8)),
                nn.ReLU(inplace=True),
                nn.Linear(max(8, d_out // 8), d_out),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, V, C]
        B, T, V, C = x.shape
        y = self.mlp(x)
        if self.use_se:
            # channel squeeze over (T*V)
            yy = y.permute(0, 3, 1, 2).reshape(B, y.shape[-1], -1)
            w = self.se(yy).view(B, -1, 1, 1)
            y = y * w.permute(0, 2, 3, 1)
        return y


class _GraphTCN(nn.Module):
    def __init__(self, in_feat: int, gcn_hidden: int = 96, tcn_hidden: int = 192, dropout: float = 0.35, use_se: bool = True):
        super().__init__()
        self.g1 = _GCNUnit(in_feat, gcn_hidden, dropout, use_se=use_se)
        self.g2 = _GCNUnit(gcn_hidden, gcn_hidden, dropout, use_se=use_se)
        self.tcn = nn.Sequential(
            _TCNBlock(gcn_hidden, tcn_hidden, k=3, dropout=dropout),
            _TCNBlock(tcn_hidden, tcn_hidden, k=3, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(tcn_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, V, F]
        y = self.g2(self.g1(x))  # [B, T, V, H]
        # pool over V
        y = y.mean(dim=2)  # [B, T, H]
        y = y.transpose(1, 2)  # [B, H, T]
        y = self.tcn(y)
        return self.head(y).squeeze(-1)


class _TwoStreamGCN(nn.Module):
    def __init__(
        self,
        in_feat: int,
        gcn_hidden: int = 96,
        tcn_hidden: int = 192,
        dropout: float = 0.35,
        use_se: bool = True,
        fuse: str = "concat",
    ):
        super().__init__()
        self.fuse = fuse
        self.s1 = _GraphTCN(in_feat, gcn_hidden, tcn_hidden, dropout, use_se)
        self.s2 = _GraphTCN(in_feat, gcn_hidden, tcn_hidden, dropout, use_se)
        if fuse == "concat":
            self.head = nn.Sequential(nn.Linear(2, 1))
        else:
            self.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # crude augmentation: second stream sees reversed time
        y1 = self.s1(x)
        y2 = self.s2(torch.flip(x, dims=[1]))
        if self.fuse == "mean":
            return 0.5 * (y1 + y2)
        if self.fuse == "max":
            return torch.maximum(y1, y2)
        # concat -> simple linear fusion
        y = torch.stack([y1, y2], dim=1)  # [B, 2]
        return self.head(y).squeeze(-1)


def _get_state_dict(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for k in ["model", "state_dict", "model_state_dict"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        return {k: v for k, v in ckpt_obj.items() if torch.is_tensor(v)}
    raise ValueError("Unsupported checkpoint format")


def _infer_tcn_in_ch(state: Dict[str, torch.Tensor]) -> Optional[int]:
    for k, v in state.items():
        if k.endswith("conv1.weight") and v.ndim == 3:
            return int(v.shape[1])
    for k, v in state.items():
        if v.ndim == 3:
            return int(v.shape[1])
    return None


def build_model_from_ckpt(arch: str, ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    state = _get_state_dict(ckpt)

    if arch == "tcn":
        # Detect TCN variant by checkpoint keys
        is_v2 = any(str(k).startswith("conv_in.") or str(k).startswith("blocks.") for k in state.keys())

        in_ch = ckpt.get("in_ch", None)
        if in_ch is None:
            in_ch = _infer_tcn_in_ch(state)
        hid = int(cfg.get("hid", 128))
        dropout = float(cfg.get("dropout", 0.30))

        if is_v2:
            # Infer depth from blocks.N.* keys when possible
            block_ids = []
            for k in state.keys():
                m = re.match(r"blocks\.(\d+)\.", str(k))
                if m:
                    block_ids.append(int(m.group(1)))
            depth = (max(block_ids) + 1) if block_ids else int(cfg.get("depth", 4))
            model = _TCNv2(int(in_ch or 132), hid=hid, depth=depth, dropout=dropout)
        else:
            model = _SimpleTCN(int(in_ch or 132), hid=hid, dropout=dropout)
    else:
        use_motion = int(cfg.get("use_motion", 1))
        use_conf = int(cfg.get("use_conf_channel", 1))
        in_feat = 2 + (2 if use_motion else 0) + (1 if use_conf else 0)
        gcn_hidden = int(cfg.get("gcn_hidden", 96))
        tcn_hidden = int(cfg.get("tcn_hidden", 192))
        dropout = float(cfg.get("dropout", 0.35))
        use_se = bool(int(cfg.get("use_se", 1)))
        two_stream = bool(int(cfg.get("two_stream", 0)))
        fuse = str(cfg.get("fuse", "concat"))
        if two_stream:
            model = _TwoStreamGCN(in_feat, gcn_hidden, tcn_hidden, dropout, use_se, fuse)
        else:
            model = _GraphTCN(in_feat, gcn_hidden, tcn_hidden, dropout, use_se)

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


# ----------------------------
# Main scoring
# ----------------------------

def _estimate_duration_sec(metas: List[Dict[str, object]]) -> float:
    starts = [m.get("w_start") for m in metas if m.get("w_start") is not None]
    ends = [m.get("w_end") for m in metas if m.get("w_end") is not None]
    fpss = [m.get("fps") for m in metas if m.get("fps") is not None]
    if not starts or not ends or not fpss:
        return 0.0
    fps = float(np.median(np.asarray(fpss, dtype=np.float32)))
    if fps <= 0:
        return 0.0
    return float((max(ends) - min(starts) + 1) / fps)


def count_events_from_flags(flags: np.ndarray) -> int:
    if flags.size == 0:
        return 0
    flags = flags.astype(bool)
    return int(np.sum((~flags[:-1]) & (flags[1:])) + (1 if flags[0] else 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["tcn", "gcn"])
    ap.add_argument("--win_dir", required=True, help="Directory with .npz windows")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--center", choices=["pelvis", "none"], default="pelvis")
    ap.add_argument("--use_motion", type=int, default=1)
    ap.add_argument("--use_conf_channel", type=int, default=1)
    ap.add_argument("--motion_scale_by_fps", type=int, default=1)
    ap.add_argument("--conf_gate", type=float, default=0.20)
    ap.add_argument("--use_precomputed_mask", type=int, default=1)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    files = _list_npz_files(args.win_dir)
    if not files:
        raise SystemExit(f"[err] no .npz files under: {args.win_dir}")

    device = pick_device()
    model = build_model_from_ckpt(args.arch, args.ckpt, device)

    probs: List[float] = []
    metas: List[Dict[str, object]] = []

    # simple batching
    for i in range(0, len(files), args.batch):
        batch_files = files[i : i + args.batch]
        xs = []
        batch_metas = []
        for p in batch_files:
            if args.arch == "tcn":
                x, m = read_window_tcn(
                    p,
                    center=args.center,
                    use_motion=args.use_motion,
                    use_conf_channel=args.use_conf_channel,
                    motion_scale_by_fps=args.motion_scale_by_fps,
                    conf_gate=args.conf_gate,
                    use_precomputed_mask=args.use_precomputed_mask,
                )
                xs.append(torch.from_numpy(x))
            else:
                x, m = read_window_gcn(
                    p,
                    center=args.center,
                    use_motion=args.use_motion,
                    use_conf_channel=args.use_conf_channel,
                    motion_scale_by_fps=args.motion_scale_by_fps,
                    conf_gate=args.conf_gate,
                    use_precomputed_mask=args.use_precomputed_mask,
                )
                xs.append(torch.from_numpy(x))
            batch_metas.append(m)

        xb = torch.stack(xs, dim=0).to(device)
        with torch.no_grad():
            pb = p_fall_from_logits(model(xb)).detach().cpu().numpy().astype(np.float32)
        probs.extend(pb.tolist())
        metas.extend(batch_metas)

    probs_arr = np.asarray(probs, dtype=np.float32)
    flags = probs_arr >= float(args.thr)

    # group by video_id
    by_vid: Dict[str, List[Tuple[int, bool, Dict[str, object]]]] = {}
    for f, m in zip(flags.tolist(), metas):
        vid = str(m.get("video_id") or "")
        st = int(m.get("w_start") or 0)
        by_vid.setdefault(vid, []).append((st, bool(f), m))

    total_events = 0
    total_dur = 0.0
    for vid, rows in by_vid.items():
        rows.sort(key=lambda t: t[0])
        fl = np.asarray([r[1] for r in rows], dtype=bool)
        total_events += count_events_from_flags(fl)
        total_dur += _estimate_duration_sec([r[2] for r in rows])

    hours = total_dur / 3600.0
    alert_per_24h = (total_events / max(hours, 1e-9)) * 24.0

    out = {
        "arch": args.arch,
        "win_dir": os.path.abspath(args.win_dir),
        "ckpt": args.ckpt,
        "thr": float(args.thr),
        "n_windows": int(len(files)),
        "n_alert_events": int(total_events),
        "duration_hours_est": float(hours),
        "alert_per_24h_est": float(alert_per_24h),
        "note": "FA/24h estimate uses window timestamps. For best accuracy, evaluate on dense sliding windows.",
    }

    print(
        f"[alert] windows={len(files)}  events={total_events}  hours≈{hours:.3f}  alert/24h≈{alert_per_24h:.2f}  thr={args.thr:.3f}"
    )

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] wrote: {args.out_json}")


if __name__ == "__main__":
    main()
