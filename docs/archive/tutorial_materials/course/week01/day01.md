# Day 1: Set Up The Week 1 Pipeline And Generate Demo Pose Data

## 1. Today's goal

Today we will do two things:

1. set up the Week 1 student workspace
2. generate a tiny synthetic pose dataset that looks like the kind of sequence data the real repository works with

By the end of today, you will have actual `.npz` pose-sequence files on disk that we can label, split, and window during the rest of the week.

## 2. Why this part exists in the full pipeline

Before you can talk about labels, windows, or training, you need to know what one pose sequence looks like.

In the real repository, that sequence contract comes from:

- `scripts/extract_pose_videos.py`
- `scripts/preprocess_pose.py`
- `src/fall_detection/pose/preprocess_pose_npz.py`

Those files are responsible for taking raw inputs and writing cleaned pose sequences. Our teaching version starts one step later: we will generate a tiny synthetic dataset that already looks like cleaned pose data.

This lets beginners focus on the machine-learning data pipeline before worrying about computer vision extraction.

## 3. What you will finish by the end of today

By the end of today you will have:

- a `course_project/` workspace
- a reusable I/O helper module
- a synthetic pose-data generator
- demo `.npz` files saved in `data/course_demo/pose_npz/`
- a CSV annotation file saved in `data/course_demo/annotations.csv`

## 4. File tree snapshot for today

By the end of today's lesson, your project tree should look like this:

```text
course_project/
├── __init__.py
├── common/
│   └── io.py
└── scripts/
    └── make_demo_pose_data.py

data/
└── course_demo/
    ├── annotations.csv
    └── pose_npz/
        ├── subject01_fall_001.npz
        ├── subject01_walk_001.npz
        ├── subject02_fall_001.npz
        ├── subject02_walk_001.npz
        ├── subject03_fall_001.npz
        └── subject04_walk_001.npz
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/__init__.py`

Why this file exists:

- it turns `course_project/` into a normal Python package
- it makes imports like `from course_project.common.io import save_json` predictable for beginners

```python
"""Course project package for the Week 1 teaching build."""
```

### File 2: `course_project/common/io.py`

Why this file exists:

- every later script will need deterministic file discovery and small JSON/text helpers
- beginners should not repeat file I/O logic in every script

```python
"""Small I/O helpers for the course project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def read_lines(path: str | Path) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    out: list[str] = []
    for raw in file_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def write_lines(lines: list[str], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    file_path.write_text(payload, encoding="utf-8")


def list_npz_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(p for p in root_path.rglob("*.npz") if p.is_file())
```

### File 3: `course_project/scripts/make_demo_pose_data.py`

Why this file exists:

- we need a tiny teaching dataset that students can generate immediately
- the whole Week 1 pipeline should be runnable without depending on a real external dataset download

```python
"""Generate a tiny synthetic pose dataset for Week 1."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


DEMO_ROWS = [
    ("subject01_fall_001", 1, 20, 34),
    ("subject01_walk_001", 0, -1, -1),
    ("subject02_fall_001", 1, 22, 36),
    ("subject02_walk_001", 0, -1, -1),
    ("subject03_fall_001", 1, 18, 30),
    ("subject04_walk_001", 0, -1, -1),
]


def stable_seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def make_sequence(video_id: str, label: int, span_start: int, span_end: int, frames: int = 48, joints: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed_from_text(video_id))
    xy = np.zeros((frames, joints, 2), dtype=np.float32)
    conf = np.ones((frames, joints), dtype=np.float32)

    base_x = np.linspace(0.2, 0.8, joints, dtype=np.float32)
    for t in range(frames):
        drift = 0.01 * t
        xy[t, :, 0] = base_x + drift
        xy[t, :, 1] = 0.3 + 0.02 * np.arange(joints, dtype=np.float32)

    xy += rng.normal(loc=0.0, scale=0.005, size=xy.shape).astype(np.float32)

    if label == 1 and span_start >= 0:
        drop = np.linspace(0.0, 0.35, span_end - span_start, dtype=np.float32)
        for idx, t in enumerate(range(span_start, span_end)):
            xy[t, :, 1] += drop[idx]

    return xy, conf


def write_demo_dataset(out_dir: str | Path, ann_csv: str | Path) -> None:
    pose_dir = Path(out_dir)
    pose_dir.mkdir(parents=True, exist_ok=True)

    ann_path = Path(ann_csv)
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    with ann_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "label", "span_start", "span_end"])
        for video_id, label, span_start, span_end in DEMO_ROWS:
            xy, conf = make_sequence(video_id, label, span_start, span_end)
            np.savez_compressed(
                pose_dir / f"{video_id}.npz",
                xy=xy,
                conf=conf,
                fps=np.float32(25.0),
                seq_id=video_id,
                src=video_id,
            )
            writer.writerow([video_id, label, span_start, span_end])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic pose dataset.")
    parser.add_argument("--out_dir", default="data/course_demo/pose_npz")
    parser.add_argument("--ann_csv", default="data/course_demo/annotations.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_demo_dataset(args.out_dir, args.ann_csv)
    print(f"[ok] wrote demo pose data to {args.out_dir}")
    print(f"[ok] wrote annotations to {args.ann_csv}")


if __name__ == "__main__":
    main()
```

## 6. Detailed teaching explanation

### Tiny concrete example first

Before we explain the code, let us agree on what one pose sequence looks like.

If a sequence has:

- `48` frames
- `5` joints
- `2` coordinates per joint

then its `xy` array has shape:

`[48, 5, 2]`

Frame `0` might look like:

```python
[
  [0.20, 0.30],
  [0.35, 0.32],
  [0.50, 0.34],
  [0.65, 0.36],
  [0.80, 0.38],
]
```

That means:

- 5 joints
- each joint has `(x, y)`

That is the representation our whole teaching Week 1 pipeline will use.

### `course_project/common/io.py` line by line

`from pathlib import Path`

- we use `Path` because it makes file and directory handling much clearer than raw strings

`from typing import Any`

- this tells the reader that our JSON helpers are intentionally generic
- later in the course, these functions will load dictionaries, lists, and mixed structures

`def load_json(path: str | Path) -> Any:`

- input: a path to a JSON file
- output: the parsed Python object
- side effects: reads one file from disk
- why it exists: later scripts should not each rewrite their own JSON loader

`file_path = Path(path)`

- normalize the incoming value once
- after this line, the function works the same whether the caller passed a string or a `Path`

`with file_path.open("r", encoding="utf-8") as f:`

- open the file safely
- choose UTF-8 explicitly so text decoding stays predictable

`return json.load(f)`

- turn the JSON file into a Python object in one step

`def save_json(obj: Any, path: str | Path) -> None:`

- input: a Python object and an output path
- output: none
- side effects: creates parent folders and writes JSON
- why it exists: labels, spans, and summaries all become JSON artifacts

`file_path.parent.mkdir(parents=True, exist_ok=True)`

- create missing parent folders automatically
- this is one of those lines beginners often skip and later regret

`json.dump(obj, f, indent=2, sort_keys=True)`

- `indent=2` makes the file readable
- `sort_keys=True` makes the output deterministic
- deterministic outputs are much easier to compare during debugging

`def read_lines(...)` and `def write_lines(...)`

- these will become useful when we start writing split lists like `train.txt`

Important lines inside `read_lines(...)`:

`line = raw.strip()`

- trim whitespace so empty space does not accidentally become part of the split list

`if not line or line.startswith("#"):`

- ignore blank lines
- ignore comment lines
- this makes the text format more forgiving and more human-friendly

Important lines inside `write_lines(...)`:

`payload = "\n".join(lines)`

- convert a list of strings into one text block

`if payload: payload += "\n"`

- add a trailing newline only when there is content
- this makes command-line tools display the file more cleanly

`def list_npz_files(...)`

- input: a directory
- output: a sorted list of `.npz` files
- side effects: none
- why it exists: determinism matters; if file order changes randomly, debugging becomes harder

`return sorted(p for p in root_path.rglob("*.npz") if p.is_file())`

- `rglob("*.npz")` searches recursively
- `if p.is_file()` filters out anything unexpected
- `sorted(...)` forces a stable order

### `course_project/scripts/make_demo_pose_data.py` section by section

Before we go line by line, I want to correct an important reproducibility mistake that beginners often make.

An earlier version of this lesson used:

```python
abs(hash(video_id)) % (2**32)
```

That looks deterministic, but Python intentionally randomizes string hashing between interpreter sessions. So the same `video_id` can produce a different hash value across runs. That means the generated synthetic noise can drift from session to session.

For a teaching pipeline, that is bad.

So we replace it with a stable SHA-256 based seed.

#### `DEMO_ROWS`

This table defines the tiny teaching dataset.

Each row stores:

- `video_id`
- binary label
- fall start frame
- fall end frame

If the label is `0`, we use `-1, -1` to mean "no fall span".

We keep this table small on purpose. Students should be able to read the entire dataset definition in one glance.

#### `stable_seed_from_text(...)` line by line

Inputs:

- one text string, here the `video_id`

Outputs:

- one deterministic integer seed

Side effects:

- none

Why it exists:

- reproducibility matters even for synthetic teaching data

`digest = hashlib.sha256(text.encode("utf-8")).hexdigest()`

- convert the text to bytes
- hash it
- get the hexadecimal string representation

`return int(digest[:16], 16) % (2**32)`

- take the first 16 hex characters
- convert that substring from base 16 into an integer
- reduce it into a 32-bit range so NumPy accepts it comfortably

#### `make_sequence(...)`

Inputs:

- `video_id`
- `label`
- `span_start`
- `span_end`
- optional frame count and joint count

Outputs:

- `xy` array shaped `[T, J, 2]`
- `conf` array shaped `[T, J]`

Side effects:

- none, it just returns arrays

Why it exists:

- this is the teaching stand-in for the real pose extraction and pose preprocessing pipeline

Important lines:

`rng = np.random.default_rng(stable_seed_from_text(video_id))`

- this is the exact line that makes the sequence reproducible across repeated runs
- if the `video_id` stays the same, the random noise stays the same

`xy = np.zeros((frames, joints, 2), dtype=np.float32)`

- this creates the main pose tensor

`conf = np.ones((frames, joints), dtype=np.float32)`

- all joints start with perfect confidence in the teaching dataset
- that is a simplification
- we keep the array anyway because the real repository frequently carries confidence through later stages

`base_x = np.linspace(0.2, 0.8, joints, dtype=np.float32)`

- spread the joints across the horizontal axis
- otherwise every joint would start at the same x-position, which would make the synthetic skeleton harder to read

`xy[t, :, 0] = base_x + drift`

- x coordinates drift gently over time so the sequence is not perfectly static

`xy[t, :, 1] = 0.3 + 0.02 * np.arange(joints, dtype=np.float32)`

- give the joints a simple vertical ordering
- this is not meant to be a realistic human skeleton
- it is meant to be a stable, inspectable teaching representation

`if label == 1 and span_start >= 0:`

- for positive videos, we simulate a downward motion during the fall span by increasing y values

That is a teaching simplification of a real fall event.

`drop = np.linspace(0.0, 0.35, span_end - span_start, dtype=np.float32)`

- build a gradually increasing vertical displacement

`xy[t, :, 1] += drop[idx]`

- apply that displacement during the annotated event interval
- this gives positive sequences a visible temporal pattern that later window labels can align with

#### `write_demo_dataset(...)`

Inputs:

- output directory for `.npz` files
- path for the annotations CSV

Outputs:

- none

Side effects:

- writes `.npz` files
- writes `annotations.csv`

Why it exists:

- the rest of Week 1 needs real files to work with

Important non-trivial lines:

`pose_dir.mkdir(parents=True, exist_ok=True)`

- create the output folder if it does not exist yet

`writer.writerow(["video_id", "label", "span_start", "span_end"])`

- write an explicit CSV header
- that makes the annotation file self-describing

#### `np.savez_compressed(...)`

This writes one pose sequence with keys:

- `xy`
- `conf`
- `fps`
- `seq_id`
- `src`

That matches the spirit of the real repository, where sequence NPZs carry both arrays and metadata.

`writer.writerow([video_id, label, span_start, span_end])`

- write the annotation row that tomorrow's label builder will read

## 7. Exact run commands

Create the folders:

```bash
mkdir -p course_project/common course_project/scripts data/course_demo
```

Copy the files from today's lesson into:

- `course_project/__init__.py`
- `course_project/common/io.py`
- `course_project/scripts/make_demo_pose_data.py`

Then run:

```bash
PYTHONPATH="$(pwd)" python3 course_project/scripts/make_demo_pose_data.py
```

Inspect one output file:

```bash
PYTHONPATH="$(pwd)" python3 - <<'PY'
import numpy as np
z = np.load("data/course_demo/pose_npz/subject01_fall_001.npz")
print(z["xy"].shape)
print(z["conf"].shape)
print(float(z["fps"]))
PY
```

## 8. Expected outputs

Generator command:

```text
[ok] wrote demo pose data to data/course_demo/pose_npz
[ok] wrote annotations to data/course_demo/annotations.csv
```

Inspection command:

```text
(48, 5, 2)
(48, 5)
25.0
```

## 9. Sanity checks

Check that:

1. `data/course_demo/pose_npz/` contains six `.npz` files
2. `data/course_demo/annotations.csv` exists
3. one `.npz` file contains `xy`, `conf`, and `fps`
4. `xy.shape[-1]` is `2`

## 10. Common bugs and fixes

### Bug: `ModuleNotFoundError: No module named 'course_project'`

Fix:

- run with `PYTHONPATH="$(pwd)"`

### Bug: output folder is empty

Fix:

- confirm the script name is exactly `course_project/scripts/make_demo_pose_data.py`
- confirm the `main()` block was copied

### Bug: `xy` has the wrong shape

Fix:

- check that `xy = np.zeros((frames, joints, 2), dtype=np.float32)` was copied exactly

### Bug: the dataset changes across repeated Python runs

Fix:

- make sure you copied `stable_seed_from_text(...)`
- make sure the random generator uses `stable_seed_from_text(video_id)`

Why the older version was wrong:

- Python's built-in `hash()` is not guaranteed to stay stable across interpreter sessions
- a cryptographic hash like SHA-256 gives us the repeatability we actually want

## 11. Mapping to the original repository

Today's teaching files correspond most closely to:

- `src/fall_detection/pose/preprocess_pose_npz.py`
- `Makefile` pose and preprocessing stages

Teaching simplification:

- we generate synthetic pose data instead of extracting it from video

Full repository version:

- raw videos or images are processed into pose sequences through real extraction and cleanup steps

## 12. Tomorrow's preview

Tomorrow we will take `annotations.csv` and turn it into the two core artifacts that the rest of the repository relies on:

- `labels.json`
- `spans.json`

That is the first place where the project begins to look like a real supervised learning pipeline.
