# Day 1: Understand A Window Sample And Build The Feature Contract

## 1. Today's goal

Today we will teach the most important data concept in Week 2:

one training sample is one **window**, not one frame.

We will also write the feature-building module that turns a saved Week 1 window file into a model-ready tensor.

## 2. Why this part exists in the full pipeline

In the real repository, the training code does not read raw pose sequences directly. It reads saved window artifacts and reconstructs features through:

- `src/fall_detection/core/features.py`

That file is a quiet but critical contract point. It decides:

- how window files are decoded
- which channels are used
- what tensor shape the model receives

Our teaching version keeps that same role.

## 3. What you will finish by the end of today

By the end of today you will have:

- `course_project/core/__init__.py`
- `course_project/core/features.py`
- one clear understanding of the shape contracts used in Week 2

## 4. File tree snapshot for today

```text
course_project/
├── core/
│   ├── __init__.py
│   └── features.py
└── ...
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/core/__init__.py`

Why this file exists:

- it makes the `core` folder a proper Python package
- it tells students that core modules hold contracts reused by several later stages

```python
"""Core contracts for the course project."""
```

### File 2: `course_project/core/features.py`

Why this file exists:

- this file turns one saved window artifact into the exact tensor shape the model will consume

```python
"""Feature building for the Week 2 teaching pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatCfg:
    """Minimal feature configuration for the teaching model."""

    use_motion: bool = True
    use_conf_channel: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> "FeatCfg":
        payload = data or {}
        return FeatCfg(
            use_motion=bool(payload.get("use_motion", True)),
            use_conf_channel=bool(payload.get("use_conf_channel", True)),
        )


@dataclass(frozen=True)
class WindowMeta:
    """Metadata read from one saved window file."""

    path: str
    video_id: str
    w_start: int
    w_end: int
    fps: float
    y: int


def _safe_scalar(z: np.lib.npyio.NpzFile, key: str, default: float) -> float:
    if key not in z.files:
        return float(default)
    return float(np.asarray(z[key]).reshape(-1)[0])


def _safe_int(z: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    if key not in z.files:
        return int(default)
    return int(np.asarray(z[key]).reshape(-1)[0])


def _safe_str(z: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in z.files:
        return str(default)
    raw = np.asarray(z[key]).reshape(-1)[0]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def read_window_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, float, WindowMeta]:
    """Read one Week 1 window file."""
    file_path = Path(path)
    with np.load(file_path, allow_pickle=False) as z:
        joints = np.asarray(z["joints"] if "joints" in z.files else z["xy"], dtype=np.float32)
        motion = np.asarray(z["motion"], dtype=np.float32) if "motion" in z.files else None
        conf = np.asarray(z["conf"], dtype=np.float32) if "conf" in z.files else None
        mask = np.asarray(z["mask"], dtype=bool) if "mask" in z.files else None
        fps = _safe_scalar(z, "fps", 25.0)
        meta = WindowMeta(
            path=str(file_path),
            video_id=_safe_str(z, "video_id", file_path.stem),
            w_start=_safe_int(z, "w_start", 0),
            w_end=_safe_int(z, "w_end", max(0, joints.shape[0] - 1)),
            fps=fps,
            y=_safe_int(z, "y", -1),
        )
    return joints, motion, conf, mask, fps, meta


def build_canonical_input(
    joints_xy: np.ndarray,
    motion_xy: np.ndarray | None,
    conf: np.ndarray | None,
    mask: np.ndarray | None,
    fps: float,
    feat_cfg: FeatCfg,
) -> tuple[np.ndarray, np.ndarray]:
    """Build canonical input shaped [T, J, F]."""
    _ = fps
    joints = np.asarray(joints_xy, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 2:
        raise ValueError(f"Expected [T, J, 2], got {joints.shape}")

    if mask is None:
        mask_arr = np.isfinite(joints[..., 0]) & np.isfinite(joints[..., 1])
    else:
        mask_arr = np.asarray(mask, dtype=bool)

    channels: list[np.ndarray] = [joints]

    if feat_cfg.use_motion:
        if motion_xy is None:
            motion = np.zeros_like(joints, dtype=np.float32)
            motion[1:] = joints[1:] - joints[:-1]
        else:
            motion = np.asarray(motion_xy, dtype=np.float32)
        channels.append(motion)

    if feat_cfg.use_conf_channel:
        if conf is None:
            conf_arr = np.ones((joints.shape[0], joints.shape[1], 1), dtype=np.float32)
        else:
            conf_arr = np.asarray(conf, dtype=np.float32)[..., None]
        channels.append(conf_arr)

    x = np.concatenate(channels, axis=2).astype(np.float32, copy=False)
    return x, mask_arr


def build_tcn_input(x_can: np.ndarray) -> np.ndarray:
    """Flatten canonical [T, J, F] input into [T, C]."""
    arr = np.asarray(x_can, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [T, J, F], got {arr.shape}")
    t, j, f = arr.shape
    return arr.reshape(t, j * f)
```

## 6. Detailed teaching explanation

### Tiny concrete example first

Suppose one window file contains:

- `T = 16` frames
- `J = 5` joints
- raw `xy` coordinates with shape `[16, 5, 2]`

If we add:

- motion channels: `2`
- confidence channel: `1`

then the canonical feature tensor becomes:

`[T, J, F] = [16, 5, 5]`

That means each joint now carries:

- `x`
- `y`
- `dx`
- `dy`
- `conf`

The teaching TCN will later flatten that into:

`[16, 25]`

That is the shape we are building toward today.

### `FeatCfg` line by line

`@dataclass(frozen=True)`

- make the configuration immutable after creation
- this is useful because feature settings should behave like a contract, not a moving target

`use_motion: bool = True`

- motion is on by default because frame-to-frame movement is helpful in fall detection

`use_conf_channel: bool = True`

- confidence is also on by default so the model can learn whether certain joints are more trustworthy

`def to_dict(...)`

- later the checkpoint will need to store this configuration

`def from_dict(...)`

- and later evaluation/inference code will need to rebuild it

### `WindowMeta` line by line

This dataclass stores:

- file path
- `video_id`
- `w_start`
- `w_end`
- `fps`
- `y`

Those fields matter because the model cares about the tensor, but Week 3 will also care about timing and event reconstruction.

### `_safe_scalar`, `_safe_int`, `_safe_str`

These helpers exist because metadata in `.npz` files is often stored as tiny NumPy arrays.

For example, `fps` may not come back as plain `25.0`. It may come back as a one-element array. These helper functions normalize that quietly.

### `read_window_npz(...)` section by section

Inputs:

- path to one saved window `.npz`

Outputs:

- `joints`
- optional `motion`
- optional `conf`
- optional `mask`
- `fps`
- `WindowMeta`

Side effects:

- reads one file from disk

Why it exists:

- every later training and evaluation stage needs one reliable window reader

Important non-trivial lines:

`joints = np.asarray(z["joints"] if "joints" in z.files else z["xy"], dtype=np.float32)`

- accept either `joints` or `xy`
- this makes the teaching code a little more robust and closer to real-life compatibility handling

`motion = ... if "motion" in z.files else None`

- motion is optional because Week 1 windows do not store it yet
- if it is absent, we will compute it later

### `build_canonical_input(...)` section by section

Inputs:

- raw joint coordinates
- optional motion
- optional confidence
- optional mask
- fps
- feature config

Outputs:

- canonical tensor shaped `[T, J, F]`
- boolean mask shaped `[T, J]`

Side effects:

- none

Why it exists:

- the model should not know about raw file keys; it should receive one stable feature layout

Important non-trivial lines:

`if joints.ndim != 3 or joints.shape[-1] != 2:`

- fail fast if the input does not really look like joint coordinates

`mask_arr = np.isfinite(joints[..., 0]) & np.isfinite(joints[..., 1])`

- if no mask was saved, infer a simple validity mask from finite coordinates

`motion[1:] = joints[1:] - joints[:-1]`

- compute frame-to-frame motion
- the first frame stays zero because it has no previous frame

`conf_arr = np.asarray(conf, dtype=np.float32)[..., None]`

- add a final singleton dimension so confidence can concatenate as one feature channel

`x = np.concatenate(channels, axis=2)`

- axis `2` is the feature axis, so this is where raw position, motion, and confidence get fused together

### `build_tcn_input(...)` line by line

Inputs:

- canonical tensor shaped `[T, J, F]`

Outputs:

- flattened tensor shaped `[T, C]`

Side effects:

- none

Why it exists:

- the simple TCN we will build later expects per-frame feature vectors, not a separate joint axis

`return arr.reshape(t, j * f)`

- flatten joints and feature channels into one last dimension

## 7. Exact run commands

Create the folder:

```bash
mkdir -p course_project/core
```

Copy today's files into:

- `course_project/core/__init__.py`
- `course_project/core/features.py`

Run a quick inspection:

```bash
PYTHONPATH="$(pwd)" python3 - <<'PY'
from pathlib import Path
from course_project.core.features import FeatCfg, build_canonical_input, build_tcn_input, read_window_npz

fp = sorted(Path("data/course_demo/windows/train").glob("*.npz"))[0]
joints, motion, conf, mask, fps, meta = read_window_npz(fp)
x_can, mask_arr = build_canonical_input(joints, motion, conf, mask, fps, FeatCfg())
x_tcn = build_tcn_input(x_can)
print(joints.shape)
print(x_can.shape)
print(x_tcn.shape)
print(meta)
PY
```

## 8. Expected outputs

Output pattern:

```text
(16, 5, 2)
(16, 5, 5)
(16, 25)
WindowMeta(...)
```

## 9. Sanity checks

Check that:

1. `joints.shape` is `(16, 5, 2)`
2. `x_can.shape` is `(16, 5, 5)` when both motion and confidence are enabled
3. `x_tcn.shape` is `(16, 25)`
4. `meta.y` is either `0` or `1`

## 10. Common bugs and fixes

### Bug: `Expected [T, J, 2]`

Fix:

- confirm you loaded `joints` or `xy`, not the entire `.npz` object

### Bug: flattened shape is wrong

Fix:

- confirm `build_tcn_input(...)` reshapes to `t, j * f`

### Bug: confidence channel missing

Fix:

- check that `FeatCfg(use_conf_channel=True)` is being used

## 11. Mapping to the original repository

Today's teaching file maps most directly to:

- `src/fall_detection/core/features.py`

Teaching simplification:

- one compact feature set

Full repository version:

- more feature flags
- more normalization choices
- more compatibility logic

## 12. Tomorrow's preview

Tomorrow we will wrap these feature functions inside a dataset and a dataloader so we can build real training batches.
