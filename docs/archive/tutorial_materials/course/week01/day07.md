# Day 7: Inspect Week 1 Outputs And Prepare For Week 2

## 1. Today's goal

Today we will inspect the artifacts produced by the Week 1 pipeline and make sure we understand exactly what Week 2 will read.

## 2. Why this part exists in the full pipeline

The real repository includes sanity-check tools because preprocessing errors are expensive.

Examples include:

- `scripts/check_spans.py`
- `scripts/check_windows.py`
- `tests/test_windows_contract.py`

Today’s lesson is the teaching version of that habit.

## 3. What you will finish by the end of today

By the end of today you will have:

- one output-checking script
- a clear understanding of the Week 1 artifact contract
- a clean handoff into Week 2

## 4. File tree snapshot for today

```text
course_project/
└── scripts/
    ├── run_week1_pipeline.py
    └── check_week1_outputs.py
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/scripts/check_week1_outputs.py`

Why this file exists:

- Week 1 should end with evidence, not assumptions
- students should confirm that the artifacts are structurally correct before training on them

```python
"""Inspect and validate the Week 1 pipeline outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from course_project.common.io import load_json, list_npz_files, read_lines
from course_project.splits.make_splits import subject_id_from_video_id


def check_week1_outputs(root: str | Path = "data/course_demo") -> None:
    root_path = Path(root)
    labels = load_json(root_path / "labels.json")
    spans = load_json(root_path / "spans.json")

    split_subjects: dict[str, set[str]] = {}
    for split_name in ("train", "val", "test"):
        video_ids = read_lines(root_path / "splits" / f"{split_name}.txt")
        split_subjects[split_name] = {subject_id_from_video_id(video_id) for video_id in video_ids}

    assert split_subjects["train"].isdisjoint(split_subjects["val"])
    assert split_subjects["train"].isdisjoint(split_subjects["test"])
    assert split_subjects["val"].isdisjoint(split_subjects["test"])

    all_windows = list_npz_files(root_path / "windows")
    positive_windows = 0
    for fp in all_windows:
        z = np.load(fp)
        y = int(np.asarray(z["y"]).reshape(-1)[0])
        positive_windows += int(y == 1)
        assert z["joints"].shape == (16, 5, 2)

    print(f"[ok] labels: {len(labels)} videos")
    print(f"[ok] spans: {sum(len(v) for v in spans.values())} annotated spans")
    print(f"[ok] windows: {len(all_windows)} total")
    print(f"[ok] positive windows: {positive_windows}")
    print("[ok] no subject leakage detected")


def main() -> None:
    check_week1_outputs()


if __name__ == "__main__":
    main()
```

## 6. Detailed teaching explanation

### Tiny concrete example first

Week 2 is going to read window files, not sequence files.

So before we move on, we want to prove things like:

- every saved window is shaped correctly
- positive windows really exist
- subject leakage is still absent

That is what today’s script does.

### `check_week1_outputs(...)` section by section

Inputs:

- optional root path for the teaching dataset

Outputs:

- none directly

Side effects:

- reads Week 1 artifacts
- raises assertions if something is wrong
- prints a human-readable summary

Why it exists:

- this is the final confidence check before model training starts in Week 2

#### Load labels and spans

```python
labels = load_json(root_path / "labels.json")
spans = load_json(root_path / "spans.json")
```

- we start from the two core annotation artifacts
- by Day 7, students should treat these as stable contracts, not temporary helper files

#### Re-check split leakage

We rebuild the subject sets from:

- `train.txt`
- `val.txt`
- `test.txt`

and assert again that they do not overlap.

This is not redundant. It is a deliberate end-of-week integrity check.

Important non-trivial line:

`split_subjects[split_name] = {subject_id_from_video_id(video_id) for video_id in video_ids}`

- rebuild the subject sets from the saved split files themselves
- this is a good engineering habit: validate artifacts, not hidden in-memory assumptions

#### Inspect every window file

```python
all_windows = list_npz_files(root_path / "windows")
```

- collect all saved windows

Then for each one:

- read the label
- count positives
- assert that `joints.shape == (16, 5, 2)`

That is the exact contract Week 2 will rely on.

#### Print summary

The final printed lines answer the most useful Week 1 questions:

- how many videos do we have?
- how many annotated spans?
- how many windows?
- how many positive windows?
- did leakage appear?

## 7. Exact run commands

Copy today's file into:

- `course_project/scripts/check_week1_outputs.py`

Run:

```bash
PYTHONPATH="$(pwd)" python3 course_project/scripts/check_week1_outputs.py
```

## 8. Expected outputs

Output pattern:

```text
[ok] labels: 6 videos
[ok] spans: 3 annotated spans
[ok] windows: ...
[ok] positive windows: ...
[ok] no subject leakage detected
```

## 9. Sanity checks

Check that:

1. the script finishes without assertion errors
2. at least one positive window exists
3. all windows still have the expected shape
4. the printed video count matches `labels.json`

## 10. Common bugs and fixes

### Bug: assertion fails on window shape

Fix:

- inspect `course_project/windowing/make_windows.py`
- confirm the saved windows really use `window_size=16`

### Bug: positive window count is zero

Fix:

- inspect `spans.json`
- inspect the overlap rule in `window_label_for(...)`

### Bug: subject leakage assertion fails

Fix:

- regenerate splits using the Day 3 script
- confirm you grouped by subject ID, not raw video ID

## 11. Mapping to the original repository

Today's teaching check maps most directly to:

- `scripts/check_spans.py`
- `scripts/check_windows.py`
- `tests/test_windows_contract.py`
- `tests/test_split_group_leakage.py`

Teaching simplification:

- one compact inspection script

Full repository version:

- richer audits over many datasets and artifact folders

## 12. Tomorrow's preview

Week 1 is now complete.

In Week 2, we will build directly on the window artifacts you created this week. That means:

- dataset loading from window folders
- tensor shapes for model input
- a first temporal model
- training and validation loops

The most important thing to remember is this:

Week 2 does not start from raw sequences.

It starts from the windows you now understand and can trust.
