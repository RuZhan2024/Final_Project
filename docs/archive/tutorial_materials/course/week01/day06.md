# Day 6: Run The Full Week 1 Preprocessing Pipeline With One Command

## 1. Today's goal

Today we will create one orchestration script that runs the whole Week 1 pipeline:

1. generate demo pose sequences
2. build labels and spans
3. build splits
4. export windows

## 2. Why this part exists in the full pipeline

The real repository is orchestrated through:

- `Makefile`
- thin script wrappers in `scripts/`

That means the real project is not designed as a sequence of manual one-off steps. It is designed as a repeatable pipeline.

Today we will build that same habit in a tiny teaching form.

## 3. What you will finish by the end of today

By the end of today you will have:

- one script that runs the whole Week 1 preprocessing flow
- a single command students can use to reproduce the full Week 1 artifact set

## 4. File tree snapshot for today

```text
course_project/
└── scripts/
    ├── make_demo_pose_data.py
    └── run_week1_pipeline.py
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/scripts/run_week1_pipeline.py`

Why this file exists:

- good pipelines should be reproducible
- beginners should experience the workflow as one coherent system, not four disconnected scripts

```python
"""Run the full Week 1 preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

from course_project.labels.make_labels import build_labels_and_spans, load_annotation_rows
from course_project.common.io import save_json
from course_project.scripts.make_demo_pose_data import write_demo_dataset
from course_project.splits.make_splits import build_splits
from course_project.windowing.make_windows import export_split_windows


def run_week1_pipeline(root: str | Path = "data/course_demo") -> None:
    root_path = Path(root)
    pose_dir = root_path / "pose_npz"
    ann_csv = root_path / "annotations.csv"
    labels_json = root_path / "labels.json"
    spans_json = root_path / "spans.json"
    splits_dir = root_path / "splits"
    windows_dir = root_path / "windows"

    write_demo_dataset(pose_dir, ann_csv)

    rows = load_annotation_rows(ann_csv)
    labels, spans = build_labels_and_spans(rows)
    save_json(labels, labels_json)
    save_json(spans, spans_json)

    splits = build_splits(labels, seed=123, train_frac=0.5, val_frac=0.25)
    for split_name, video_ids in splits.items():
        from course_project.common.io import write_lines
        write_lines(video_ids, splits_dir / f"{split_name}.txt")

    total = 0
    for split_name, video_ids in splits.items():
        total += export_split_windows(
            split_name=split_name,
            video_ids=video_ids,
            pose_dir=pose_dir,
            out_root=windows_dir,
            spans=spans,
            window_size=16,
            stride=8,
        )

    print("[ok] Week 1 pipeline finished")
    print(f"[ok] root: {root_path}")
    print(f"[ok] total windows: {total}")


def main() -> None:
    run_week1_pipeline()


if __name__ == "__main__":
    main()
```

## 6. Detailed teaching explanation

### Tiny concrete example first

This script is doing the Week 1 equivalent of what `make pipeline-data-<dataset>` does in the real repository.

It is a tiny orchestrator.

Instead of telling students to manually remember:

- first run generator
- then labels
- then splits
- then windows

we give them one reproducible entry point.

### `run_week1_pipeline(...)` section by section

Inputs:

- optional root path for the teaching dataset

Outputs:

- none directly

Side effects:

- writes all Week 1 artifacts to disk

Why it exists:

- makes the Week 1 pipeline reproducible and easy to rerun

#### Path setup

```python
root_path = Path(root)
pose_dir = root_path / "pose_npz"
...
```

- all Week 1 outputs live under one root
- this keeps the project tidy and makes cleanup easy
- `Path(root)` also normalizes the incoming path immediately so every later join is explicit and readable

#### Generate synthetic data

```python
write_demo_dataset(pose_dir, ann_csv)
```

- Day 1 stage
- this line is the teaching stand-in for a much larger real extraction and preprocessing stage

#### Build labels and spans

```python
rows = load_annotation_rows(ann_csv)
labels, spans = build_labels_and_spans(rows)
```

- Day 2 stage

#### Build splits

```python
splits = build_splits(labels, seed=123, train_frac=0.5, val_frac=0.25)
```

- Day 3 stage

#### Write split files

We use `write_lines(...)` to turn the in-memory split dictionary into the standard `train.txt`, `val.txt`, `test.txt` artifacts.

#### Export windows

```python
total += export_split_windows(...)
```

- Day 4 stage

By now the structure of the full Week 1 pipeline should feel clear:

`raw sequence files -> annotations -> labels/spans -> split lists -> windows`

## 7. Exact run commands

Copy today's file into:

- `course_project/scripts/run_week1_pipeline.py`

Run:

```bash
PYTHONPATH="$(pwd)" python3 course_project/scripts/run_week1_pipeline.py
```

Inspect the top-level outputs:

```bash
find data/course_demo -maxdepth 2 -type f | sort
```

## 8. Expected outputs

```text
[ok] Week 1 pipeline finished
[ok] root: data/course_demo
[ok] total windows: ...
```

## 9. Sanity checks

Check that these now all exist:

1. `data/course_demo/annotations.csv`
2. `data/course_demo/labels.json`
3. `data/course_demo/spans.json`
4. `data/course_demo/splits/train.txt`
5. `data/course_demo/splits/val.txt`
6. `data/course_demo/splits/test.txt`
7. `data/course_demo/windows/train/`
8. `data/course_demo/windows/val/`
9. `data/course_demo/windows/test/`

## 10. Common bugs and fixes

### Bug: pipeline runs but no windows are created

Fix:

- confirm `export_split_windows(...)` is being called for every split

### Bug: import errors inside the runner

Fix:

- run with `PYTHONPATH="$(pwd)"`
- confirm all previous lesson files exist

### Bug: split files are missing

Fix:

- confirm `write_lines(...)` is being called inside the split loop

## 11. Mapping to the original repository

Today's orchestration lesson maps most directly to:

- `Makefile`
- the sequence of `scripts/make_labels_*.py`, `scripts/make_splits.py`, and `scripts/make_windows.py`

Teaching simplification:

- one Python script instead of a full Makefile DAG

Full repository version:

- more stages
- dataset-specific branches
- stamp files and reproducible dependency tracking

## 12. Tomorrow's preview

Tomorrow we will finish Week 1 properly by inspecting the outputs, checking for leakage again, and explaining exactly what Week 2 will consume.
