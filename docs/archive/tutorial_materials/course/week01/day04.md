# Day 4: Export Fixed Windows From Sequence Files

## 1. Today's goal

Today we will convert full pose sequences into fixed windows and save them as `.npz` files.

This is the most important artifact-building day of Week 1.

## 2. Why this part exists in the full pipeline

The real repository trains on window artifacts, not whole raw sequences.

The active windowing stage lives in:

- `scripts/make_windows.py`
- `src/fall_detection/data/windowing/make_windows_impl.py`

That stage decides:

- how large each sample window is
- how far windows move
- whether a window is positive or negative
- which metadata is saved for later training and evaluation

Today we will build the teaching version of that logic.

## 3. What you will finish by the end of today

By the end of today you will have:

- a window exporter module
- `train/`, `val/`, and `test/` window directories
- saved `.npz` window samples ready for Week 2

## 4. File tree snapshot for today

```text
course_project/
├── windowing/
│   ├── __init__.py
│   └── make_windows.py
└── ...

data/
└── course_demo/
    ├── labels.json
    ├── spans.json
    ├── splits/
    ├── pose_npz/
    └── windows/
        ├── train/
        ├── val/
        └── test/
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/windowing/__init__.py`

```python
"""Window exporters for the course project."""
```

### File 2: `course_project/windowing/make_windows.py`

```python
"""Convert pose sequences into fixed-size window files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from course_project.common.io import list_npz_files, load_json, read_lines


def span_overlap(window_start: int, window_end: int, span_start: int, span_end: int) -> int:
    left = max(window_start, span_start)
    right = min(window_end, span_end)
    return max(0, right - left)


def window_label_for(video_id: str, w_start: int, w_end: int, spans: dict[str, list[list[int]]]) -> int:
    for span_start, span_end in spans.get(video_id, []):
        if span_overlap(w_start, w_end, int(span_start), int(span_end)) > 0:
            return 1
    return 0


def iter_windows(xy: np.ndarray, conf: np.ndarray, window_size: int, stride: int):
    total_frames = int(xy.shape[0])
    for w_start in range(0, total_frames - window_size + 1, stride):
        w_end = w_start + window_size
        yield w_start, w_end, xy[w_start:w_end], conf[w_start:w_end]


def export_split_windows(
    split_name: str,
    video_ids: list[str],
    pose_dir: str | Path,
    out_root: str | Path,
    spans: dict[str, list[list[int]]],
    window_size: int,
    stride: int,
) -> int:
    pose_root = Path(pose_dir)
    out_dir = Path(out_root) / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for video_id in video_ids:
        z = np.load(pose_root / f"{video_id}.npz")
        xy = np.asarray(z["xy"], dtype=np.float32)
        conf = np.asarray(z["conf"], dtype=np.float32)
        fps = float(np.asarray(z["fps"]).reshape(-1)[0])

        for index, (w_start, w_end, win_xy, win_conf) in enumerate(iter_windows(xy, conf, window_size, stride)):
            y = window_label_for(video_id, w_start, w_end, spans)
            np.savez_compressed(
                out_dir / f"{video_id}__w{index:03d}.npz",
                joints=win_xy,
                xy=win_xy,
                conf=win_conf,
                fps=np.float32(fps),
                video_id=video_id,
                w_start=np.int32(w_start),
                w_end=np.int32(w_end),
                y=np.int32(y),
                label=np.int32(y),
            )
            written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed windows from pose sequences.")
    parser.add_argument("--pose_dir", default="data/course_demo/pose_npz")
    parser.add_argument("--spans_json", default="data/course_demo/spans.json")
    parser.add_argument("--splits_dir", default="data/course_demo/splits")
    parser.add_argument("--out_dir", default="data/course_demo/windows")
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spans = load_json(args.spans_json)
    total = 0
    for split_name in ("train", "val", "test"):
        video_ids = read_lines(Path(args.splits_dir) / f"{split_name}.txt")
        written = export_split_windows(
            split_name=split_name,
            video_ids=video_ids,
            pose_dir=args.pose_dir,
            out_root=args.out_dir,
            spans=spans,
            window_size=args.window_size,
            stride=args.stride,
        )
        total += written
        print(f"[ok] wrote {written} windows for split={split_name}")
    print(f"[ok] total windows written: {total}")


if __name__ == "__main__":
    main()
```

## 6. Detailed teaching explanation

### Tiny concrete example first

Suppose one sequence has 48 frames.

If we choose:

- `window_size = 16`
- `stride = 8`

then the windows start at:

- 0
- 8
- 16
- 24
- 32

That produces 5 windows:

- `[0, 16)`
- `[8, 24)`
- `[16, 32)`
- `[24, 40)`
- `[32, 48)`

If the fall span is `[20, 34)`, then:

- `[0, 16)` is negative
- `[8, 24)` overlaps the fall, so it is positive
- `[16, 32)` overlaps the fall, so it is positive
- `[24, 40)` overlaps the fall, so it is positive
- `[32, 48)` overlaps the fall, so it is positive

That overlap logic is the heart of today’s code.

### `span_overlap(...)` line by line

Inputs:

- window start and end
- span start and end

Outputs:

- number of overlapping frames

Side effects:

- none

Why it exists:

- the label of a window depends on whether it overlaps a fall span

`left = max(window_start, span_start)`

- overlap cannot begin before both intervals have started

`right = min(window_end, span_end)`

- overlap cannot continue past the earlier ending point

### `window_label_for(...)`

Inputs:

- video ID
- window start and end
- spans dictionary

Outputs:

- `1` for positive window
- `0` for negative window

Side effects:

- none

Why it exists:

- this is the core supervision rule for window samples

Important non-trivial line:

`for span_start, span_end in spans.get(video_id, []):`

- pull all spans for the current video
- if the video has no spans, treat it as an empty list

### `iter_windows(...)`

Inputs:

- `xy`
- `conf`
- window size
- stride

Outputs:

- one window at a time: `(w_start, w_end, win_xy, win_conf)`

Side effects:

- none

Why it exists:

- this keeps the sliding-window logic separate from file writing

Important line:

`for w_start in range(0, total_frames - window_size + 1, stride):`

- this is the actual sliding-window loop

### `export_split_windows(...)`

This is the main artifact writer.

Inputs:

- split name
- list of video IDs
- pose directory
- output directory
- spans dictionary
- window size
- stride

Outputs:

- number of windows written

Side effects:

- writes one `.npz` file per window

Why it exists:

- this function creates the training contract that Week 2 will read

Important lines:

`z = np.load(pose_root / f"{video_id}.npz")`

- load one full pose sequence

`np.savez_compressed(...)`

- write the window artifact

Saved keys:

- `joints`
- `xy`
- `conf`
- `fps`
- `video_id`
- `w_start`
- `w_end`
- `y`
- `label`

Why save both `joints` and `xy`?

- teaching simplification: it makes the artifact easier to consume from multiple paths
- full repository reality: backward compatibility like this is common when a schema evolves

### `main()`

This loops over:

- train
- val
- test

and writes separate window folders for each split.

That is exactly how the real repository organizes downstream training input.

## 7. Exact run commands

Create the folder:

```bash
mkdir -p course_project/windowing
```

Copy today's files into:

- `course_project/windowing/__init__.py`
- `course_project/windowing/make_windows.py`

Then run:

```bash
PYTHONPATH="$(pwd)" python3 course_project/windowing/make_windows.py
```

Inspect the output tree:

```bash
find data/course_demo/windows -maxdepth 2 -type f | sort
```

Inspect one saved window:

```bash
python3 - <<'PY'
import numpy as np
from pathlib import Path
fp = sorted(Path("data/course_demo/windows/train").glob("*.npz"))[0]
z = np.load(fp)
print(fp.name)
print(z["joints"].shape)
print(int(z["y"]))
print(int(z["w_start"]), int(z["w_end"]))
PY
```

## 8. Expected outputs

Script output pattern:

```text
[ok] wrote ... windows for split=train
[ok] wrote ... windows for split=val
[ok] wrote ... windows for split=test
[ok] total windows written: ...
```

Inspection output pattern:

```text
subject...__w000.npz
(16, 5, 2)
0
0 16
```

## 9. Sanity checks

Check that:

1. each split folder exists
2. every saved window has shape `(16, 5, 2)` for `joints`
3. `y` is always `0` or `1`
4. positive videos produce at least some positive windows

## 10. Common bugs and fixes

### Bug: zero windows written

Fix:

- confirm `window_size` is not larger than sequence length

### Bug: every window is negative

Fix:

- inspect `spans.json`
- check the overlap logic in `window_label_for(...)`

### Bug: saved files are missing `video_id`

Fix:

- keep the metadata fields inside `np.savez_compressed(...)`

## 11. Mapping to the original repository

Today's teaching file maps most directly to:

- `scripts/make_windows.py`
- `src/fall_detection/data/windowing/make_windows_impl.py`

Teaching simplification:

- one sliding-window rule
- one positive-overlap rule

Full repository version:

- more balancing controls
- more negative sampling policy
- richer metadata
- dataset-sensitive fallback behavior

## 12. Tomorrow's preview

Tomorrow we will stop trusting ourselves and start testing the pipeline.

We will write automated checks for:

- split leakage
- window labeling behavior
