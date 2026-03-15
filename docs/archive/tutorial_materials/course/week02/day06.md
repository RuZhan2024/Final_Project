# Day 6: Add Week 2 Tests And Debugging Checks

## 1. Today's goal

Today we will add a small but important validation layer:

- a feature-shape test
- a checkpoint roundtrip test

## 2. Why this part exists in the full pipeline

The real repository depends heavily on shape contracts and artifact portability.

If feature layout drifts or checkpoints lose their config, evaluation and deployment break later. So we test those contracts now, while the code is still small.

## 3. What you will finish by the end of today

By the end of today you will have:

- `course_project/tests/test_features.py`
- `course_project/tests/test_checkpoint.py`

## 4. File tree snapshot for today

```text
course_project/
└── tests/
    ├── test_checkpoint.py
    ├── test_features.py
    ├── test_splits.py
    └── test_windows.py
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/tests/test_features.py`

```python
from __future__ import annotations

import numpy as np

from course_project.core.features import FeatCfg, build_canonical_input, build_tcn_input


def test_build_canonical_input_shape() -> None:
    joints = np.zeros((4, 3, 2), dtype=np.float32)
    conf = np.ones((4, 3), dtype=np.float32)
    x, mask = build_canonical_input(joints, None, conf, None, 25.0, FeatCfg(use_motion=True, use_conf_channel=True))
    assert x.shape == (4, 3, 5)
    assert mask.shape == (4, 3)


def test_build_tcn_input_flattens_joint_and_feature_axes() -> None:
    x = np.zeros((4, 3, 5), dtype=np.float32)
    out = build_tcn_input(x)
    assert out.shape == (4, 15)
```

### File 2: `course_project/tests/test_checkpoint.py`

```python
from __future__ import annotations

from pathlib import Path

from course_project.core.ckpt import load_ckpt, save_ckpt


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "test.pt"
    save_ckpt(
        ckpt_path,
        {
            "arch": "tcn",
            "state_dict": {},
            "model_cfg": {"hidden": 32},
            "feat_cfg": {"use_motion": True},
            "data_cfg": {"window_size": 16},
        },
    )
    bundle = load_ckpt(ckpt_path)
    assert bundle["arch"] == "tcn"
    assert bundle["model_cfg"]["hidden"] == 32
    assert bundle["feat_cfg"]["use_motion"] is True
```

## 6. Detailed teaching explanation

### Tiny concrete example first

If `build_canonical_input(...)` receives:

- 4 frames
- 3 joints
- motion enabled
- confidence enabled

then the output should be:

`(4, 3, 5)`

That is exactly the first test.

### `test_features.py` line by line

`joints = np.zeros((4, 3, 2), dtype=np.float32)`

- a tiny synthetic joint tensor

`conf = np.ones((4, 3), dtype=np.float32)`

- confidence for each frame and joint

`assert x.shape == (4, 3, 5)`

- two coordinate channels
- two motion channels
- one confidence channel

`assert out.shape == (4, 15)`

- flatten `3 joints * 5 features = 15`

### `test_checkpoint.py` line by line

`tmp_path / "test.pt"`

- pytest provides a temporary writable folder just for this test

`save_ckpt(...)`

- save a synthetic checkpoint bundle

`bundle = load_ckpt(ckpt_path)`

- read it back immediately

The three assertions check:

- architecture name survived
- model config survived
- feature config survived

That is the portability contract we need.

## 7. Exact run commands

Copy today's files into:

- `course_project/tests/test_features.py`
- `course_project/tests/test_checkpoint.py`

Run:

```bash
PYTHONPATH="$(pwd)" pytest \
  course_project/tests/test_features.py \
  course_project/tests/test_checkpoint.py \
  -q
```

## 8. Expected outputs

```text
...                                                                   [100%]
3 passed
```

## 9. Sanity checks

Check that:

1. the feature tests pass
2. the checkpoint test passes
3. if you intentionally remove `state_dict` from the saved bundle, the checkpoint path fails meaningfully

## 10. Common bugs and fixes

### Bug: feature test says shape is wrong

Fix:

- re-check whether confidence was added with `[..., None]`

### Bug: checkpoint test fails on missing keys

Fix:

- keep the exact payload keys inside `save_ckpt(...)`

## 11. Mapping to the original repository

Today's tests map most directly to:

- `tests/test_windows_contract.py`
- the broader checkpoint compatibility expectations in the training/evaluation code

## 12. Tomorrow's preview

Tomorrow we will inspect the trained output artifacts and explain exactly how Week 3 will use them for probability interpretation, thresholding, and event-level evaluation.
