# Day 5: Write Tests For Splits And Windows

## 1. Today's goal

Today we will write tests that verify the two most fragile Week 1 contracts:

- no subject leakage across splits
- window labels match fall-span overlap

## 2. Why this part exists in the full pipeline

The real repository already treats these contracts seriously. For example:

- `tests/test_split_group_leakage.py`
- `tests/test_windows_contract.py`

That is a strong engineering choice, and students should learn from it early.

## 3. What you will finish by the end of today

By the end of today you will have:

- a split leakage test
- a window semantics test
- a repeatable Week 1 validation command

## 4. File tree snapshot for today

```text
course_project/
└── tests/
    ├── test_splits.py
    └── test_windows.py
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/tests/test_splits.py`

Why this file exists:

- it checks the most important safety rule in Week 1: no subject should appear in more than one split

```python
from __future__ import annotations

from course_project.splits.make_splits import build_splits, subject_id_from_video_id


def test_subjects_do_not_leak_across_splits() -> None:
    labels = {
        "subject01_fall_001": 1,
        "subject01_walk_001": 0,
        "subject02_fall_001": 1,
        "subject02_walk_001": 0,
        "subject03_fall_001": 1,
        "subject04_walk_001": 0,
    }

    splits = build_splits(labels, seed=123, train_frac=0.5, val_frac=0.25)
    subject_sets = {
        split_name: {subject_id_from_video_id(video_id) for video_id in video_ids}
        for split_name, video_ids in splits.items()
    }

    assert subject_sets["train"].isdisjoint(subject_sets["val"])
    assert subject_sets["train"].isdisjoint(subject_sets["test"])
    assert subject_sets["val"].isdisjoint(subject_sets["test"])
```

### File 2: `course_project/tests/test_windows.py`

Why this file exists:

- it checks the most important labeling rule in Week 1: windows become positive only when they overlap a fall span

```python
from __future__ import annotations

import numpy as np

from course_project.windowing.make_windows import iter_windows, window_label_for


def test_window_label_for_marks_overlap_as_positive() -> None:
    spans = {"subject01_fall_001": [[20, 34]]}
    assert window_label_for("subject01_fall_001", 0, 16, spans) == 0
    assert window_label_for("subject01_fall_001", 16, 32, spans) == 1


def test_iter_windows_produces_expected_count() -> None:
    xy = np.zeros((48, 5, 2), dtype=np.float32)
    conf = np.ones((48, 5), dtype=np.float32)
    windows = list(iter_windows(xy, conf, window_size=16, stride=8))
    assert len(windows) == 5
```

## 6. Detailed teaching explanation

### Tiny concrete example first

If our split builder puts:

- `subject01_fall_001` in train
- `subject01_walk_001` in test

then our evaluation is already compromised, because the same person appears in both splits.

That is exactly what the first test is trying to prevent.

For windows, if span `[20, 34)` exists, then:

- window `[0, 16)` should be negative
- window `[16, 32)` should be positive

That is exactly what the second test checks.

### `test_splits.py` line by line

`from course_project.splits.make_splits import build_splits, subject_id_from_video_id`

- import the two exact functions this test is meant to trust
- good tests point directly at the behavior they are validating

`labels = {...}`

- we build a tiny synthetic label set directly inside the test

`splits = build_splits(...)`

- call the reusable split logic from Day 3

`subject_sets = {...}`

- convert each split from video IDs to subject IDs

This comprehension is worth reading carefully:

- loop over each split
- loop over each video inside that split
- convert every video ID into its subject ID
- store the unique subject IDs in a set

`assert subject_sets["train"].isdisjoint(subject_sets["val"])`

- this means train and validation share no subjects

The other two assertions check the same thing for train/test and val/test.

That repetition is not wasteful. It is explicit. In a course, explicitness is a feature.

### `test_windows.py` line by line

`from course_project.windowing.make_windows import iter_windows, window_label_for`

- import the two most important windowing behaviors directly

`spans = {"subject01_fall_001": [[20, 34]]}`

- one simple span

`assert window_label_for("subject01_fall_001", 0, 16, spans) == 0`

- no overlap, so negative

`assert window_label_for("subject01_fall_001", 16, 32, spans) == 1`

- overlap exists, so positive

`windows = list(iter_windows(...))`

- for 48 frames, window size 16, stride 8, there should be 5 windows

That is a very useful contract to lock down early.

## Break it on purpose, then fix it

I want students to feel why these tests matter, not just watch them pass.

### Exercise A: break the split logic

Open `course_project/splits/make_splits.py` and temporarily change:

```python
subject_id = subject_id_from_video_id(video_id)
```

to:

```python
subject_id = video_id
```

Now rerun:

```bash
PYTHONPATH="$(pwd)" pytest course_project/tests/test_splits.py -q
```

What should happen:

- the split test should fail

Why it failed:

- you stopped grouping by subject and started grouping by video
- that destroys the leakage protection

Now put the original line back.

### Exercise B: break the overlap rule

Open `course_project/windowing/make_windows.py` and temporarily change:

```python
if span_overlap(w_start, w_end, int(span_start), int(span_end)) > 0:
```

to:

```python
if span_overlap(w_start, w_end, int(span_start), int(span_end)) > 100:
```

Now rerun:

```bash
PYTHONPATH="$(pwd)" pytest course_project/tests/test_windows.py -q
```

What should happen:

- the overlap test should fail

Why it failed:

- you made the positive-label rule impossible to satisfy
- the test is proving that the window-label contract is real, not decorative

Now restore the original line and rerun the full Week 1 test command.

## 7. Exact run commands

Create the folder:

```bash
mkdir -p course_project/tests
```

Copy today's files into:

- `course_project/tests/test_splits.py`
- `course_project/tests/test_windows.py`

Run the tests:

```bash
PYTHONPATH="$(pwd)" pytest course_project/tests/test_splits.py course_project/tests/test_windows.py -q
```

## 8. Expected outputs

```text
...                                                                   [100%]
3 passed
```

## 9. Sanity checks

Check that:

1. all tests pass
2. if you intentionally break the overlap logic, the window test fails
3. if you split by video instead of subject, the split test fails
4. after you restore the correct lines, the full test suite passes again

## 10. Common bugs and fixes

### Bug: `pytest` cannot import `course_project`

Fix:

- run with `PYTHONPATH="$(pwd)"`

### Bug: expected 5 windows but got a different number

Fix:

- re-check the range inside `iter_windows(...)`

### Bug: the overlap test fails

Fix:

- verify that `span_overlap(...)` returns a positive value only when the intervals truly overlap

## 11. Mapping to the original repository

Today's tests map most directly to:

- `tests/test_split_group_leakage.py`
- `tests/test_windows_contract.py`

Teaching simplification:

- tiny synthetic cases

Full repository version:

- larger integration-style contracts over real artifact folders

## 12. Tomorrow's preview

Tomorrow we will stop running each script manually and build one small orchestrator that runs the whole Week 1 preprocessing pipeline end to end.
