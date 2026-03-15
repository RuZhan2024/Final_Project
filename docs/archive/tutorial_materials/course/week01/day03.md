# Day 3: Create Subject-Safe Train, Validation, And Test Splits

## 1. Today's goal

Today we will build deterministic, subject-safe split lists:

- `train.txt`
- `val.txt`
- `test.txt`

## 2. Why this part exists in the full pipeline

In fall detection, leakage is easy to introduce.

If the same person appears in both training and test data, the model can look much better than it really is.

That is why the real repository treats splits as a serious pipeline stage. The active implementation lives in:

- `scripts/make_splits.py`
- `src/fall_detection/data/splits/make_splits.py`

Today we will build the teaching version of that logic.

## 3. What you will finish by the end of today

By the end of today you will have:

- a split builder script
- one subject-safe train list
- one validation list
- one test list
- a tiny split summary JSON

## 4. File tree snapshot for today

```text
course_project/
├── splits/
│   ├── __init__.py
│   └── make_splits.py
└── ...

data/
└── course_demo/
    ├── labels.json
    ├── splits/
    │   ├── train.txt
    │   ├── val.txt
    │   ├── test.txt
    │   └── split_summary.json
    └── ...
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/splits/__init__.py`

```python
"""Split builders for the course project."""
```

### File 2: `course_project/splits/make_splits.py`

```python
"""Create subject-safe train/val/test split lists."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from course_project.common.io import load_json, save_json, write_lines


def subject_id_from_video_id(video_id: str) -> str:
    return video_id.split("_", 1)[0]


def build_grouped_video_lists(labels: dict[str, int]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for video_id in sorted(labels):
        subject_id = subject_id_from_video_id(video_id)
        grouped.setdefault(subject_id, []).append(video_id)
    return grouped


def split_subjects(subject_ids: list[str], seed: int, train_frac: float, val_frac: float) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    ids = list(subject_ids)
    rng.shuffle(ids)

    n_total = len(ids)
    n_train = max(1, int(round(train_frac * n_total)))
    n_val = max(1, int(round(val_frac * n_total)))
    if n_train + n_val >= n_total:
        n_val = 1
        n_train = max(1, n_total - 2)
    n_test = n_total - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_ids = sorted(ids[:n_train])
    val_ids = sorted(ids[n_train:n_train + n_val])
    test_ids = sorted(ids[n_train + n_val:])
    return train_ids, val_ids, test_ids


def expand_subject_split(grouped: dict[str, list[str]], subject_ids: list[str]) -> list[str]:
    out: list[str] = []
    for subject_id in subject_ids:
        out.extend(sorted(grouped.get(subject_id, [])))
    return out


def build_splits(labels: dict[str, int], seed: int = 123, train_frac: float = 0.5, val_frac: float = 0.25) -> dict[str, list[str]]:
    grouped = build_grouped_video_lists(labels)
    subject_ids = sorted(grouped)
    train_ids, val_ids, test_ids = split_subjects(subject_ids, seed, train_frac, val_frac)
    return {
        "train": expand_subject_split(grouped, train_ids),
        "val": expand_subject_split(grouped, val_ids),
        "test": expand_subject_split(grouped, test_ids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subject-safe split files.")
    parser.add_argument("--labels_json", default="data/course_demo/labels.json")
    parser.add_argument("--out_dir", default="data/course_demo/splits")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--val_frac", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_json(args.labels_json)
    splits = build_splits(labels, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)

    out_dir = Path(args.out_dir)
    write_lines(splits["train"], out_dir / "train.txt")
    write_lines(splits["val"], out_dir / "val.txt")
    write_lines(splits["test"], out_dir / "test.txt")
    save_json({name: len(items) for name, items in splits.items()}, out_dir / "split_summary.json")
    print(f"[ok] wrote splits to {out_dir}")


if __name__ == "__main__":
    main()
```

## 6. Detailed teaching explanation

### Tiny concrete example first

Suppose we have these video IDs:

```text
subject01_fall_001
subject01_walk_001
subject02_fall_001
subject02_walk_001
subject03_fall_001
subject04_walk_001
```

If we split by video directly, `subject01` could appear in both train and test.

That would be leakage.

Instead, we split by **subject ID**:

- `subject01`
- `subject02`
- `subject03`
- `subject04`

Then we expand those subject groups back into video lists.

That is the central idea of today's script.

### `subject_id_from_video_id(...)` line by line

Inputs:

- one video ID string

Outputs:

- the subject prefix

Side effects:

- none

Why it exists:

- grouping by person is the simplest reliable leakage control in this teaching dataset

`return video_id.split("_", 1)[0]`

- this keeps only the part before the first underscore
- for `subject02_fall_001`, it returns `subject02`
- the `1` means we split only once, which is exactly what we want here

### `build_grouped_video_lists(...)`

Inputs:

- `labels` dictionary

Outputs:

- a dictionary like `{"subject01": ["subject01_fall_001", "subject01_walk_001"], ...}`

Side effects:

- none

Why it exists:

- we need to split subjects first, then expand back to videos

Important line:

`grouped.setdefault(subject_id, []).append(video_id)`

- create the subject bucket if it does not exist yet
- then append the current video to that bucket

### `split_subjects(...)`

Inputs:

- subject IDs
- random seed
- train fraction
- validation fraction

Outputs:

- subject IDs for train
- subject IDs for validation
- subject IDs for test

Side effects:

- none

Why it exists:

- this is the actual split policy

Important lines:

`rng = random.Random(seed)`

- deterministic shuffling

`if n_train + n_val >= n_total:`

- protects against bad rounding on tiny datasets

That matters in teaching because we are using only four subjects.

### `expand_subject_split(...)`

Inputs:

- grouped subject-to-video mapping
- a list of subject IDs for one split

Outputs:

- the expanded video list

Side effects:

- none

Why it exists:

- later pipeline stages expect split lists of videos, not subject IDs

### `build_splits(...)`

This is the main reusable function of the file.

It:

1. groups videos by subject
2. splits the subjects
3. expands each subject split back to video IDs

This function is what our tests will target later.

### `main()`

This function turns the logic into real artifacts on disk:

- `train.txt`
- `val.txt`
- `test.txt`
- `split_summary.json`

## 7. Exact run commands

Create the folder:

```bash
mkdir -p course_project/splits
```

Copy today's files into:

- `course_project/splits/__init__.py`
- `course_project/splits/make_splits.py`

Then run:

```bash
PYTHONPATH="$(pwd)" python3 course_project/splits/make_splits.py
```

Inspect the outputs:

```bash
for f in data/course_demo/splits/train.txt data/course_demo/splits/val.txt data/course_demo/splits/test.txt; do
  echo "== $f =="
  cat "$f"
done
```

## 8. Expected outputs

Script output:

```text
[ok] wrote splits to data/course_demo/splits
```

Expected split structure:

- each file contains whole-video IDs
- no subject should appear in more than one split

Because the split is seeded, your exact files should remain stable for the same code and seed.

## 9. Sanity checks

Check that:

1. all six videos appear exactly once across the three split files
2. no split file is empty
3. `subject01` does not appear in more than one split
4. `split_summary.json` matches the actual line counts

## 10. Common bugs and fixes

### Bug: one split file is empty

Fix:

- keep the small-dataset guards inside `split_subjects(...)`

### Bug: the same subject appears in train and test

Fix:

- confirm you split subject IDs, not raw video IDs

### Bug: split files have duplicate lines

Fix:

- use `sorted(labels)` when building grouped video lists

## 11. Mapping to the original repository

Today's teaching file maps most directly to:

- `scripts/make_splits.py`
- `src/fall_detection/data/splits/make_splits.py`

Teaching simplification:

- one split rule based on subject prefix

Full repository version:

- several grouping modes
- more balancing controls
- dataset-specific grouping options such as CAUCAFall subject mode

## 12. Tomorrow's preview

Tomorrow we will build the most important Week 1 artifact:

fixed windows.

That is the point where raw sequences become training-ready machine-learning samples.
