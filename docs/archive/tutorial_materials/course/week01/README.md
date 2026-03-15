# Week 1: From Pose Sequences To Training Windows

## Why this week matters

Week 1 teaches the first real stage of the fall detection system:

`pose sequences -> labels and spans -> subject-safe splits -> fixed windows`

That is the real beginning of the repository's machine-learning pipeline. In the original project, the model never trains directly on raw videos. It trains on window artifacts produced only after labeling, split control, and window export are finished.

If students misunderstand this stage, everything later becomes unstable:

- training metrics become misleading
- evaluation leaks across subjects
- deployment thresholds are fitted on the wrong data contract

So this week is intentionally slow, explicit, and implementation-first.

Instructor note:

I want students to feel that this week is building a real system, not filling in background paperwork. We are not "doing preprocessing before the interesting part." We are building the exact artifact contract that the model, the evaluator, and the deployment path will rely on later.

## What students will build this week

By the end of Week 1, students will have written a small teaching reconstruction under `course_project/` with these core pieces:

- `course_project/common/io.py`
- `course_project/scripts/make_demo_pose_data.py`
- `course_project/labels/make_labels.py`
- `course_project/splits/make_splits.py`
- `course_project/windowing/make_windows.py`
- `course_project/tests/test_splits.py`
- `course_project/tests/test_windows.py`
- `course_project/scripts/run_week1_pipeline.py`
- `course_project/scripts/check_week1_outputs.py`

Important teaching note:

- the lesson markdown files are the source of truth
- students copy code from the notes into those paths
- the repository itself does not provide these course files as permanent standalone code
- older tutorial files have been removed so they do not compete with this course path

## How Week 1 maps to the real repository

This teaching week is based on the real active pipeline traced through:

- `Makefile`
- `scripts/make_labels_*.py`
- `scripts/make_splits.py`
- `scripts/make_windows.py`
- `src/fall_detection/data/labels/*`
- `src/fall_detection/data/splits/make_splits.py`
- `src/fall_detection/data/windowing/make_windows_impl.py`

Teaching simplification:

- one tiny synthetic pose dataset
- one generic label parser
- one subject-safe split rule
- one compact window exporter

Full project reality:

- multiple datasets with custom label parsers
- dataset-specific hacks and annotation conventions
- more balancing rules
- richer window schema and export settings

## Daily lesson map

### Day 1

- course setup
- pose sequence representation
- tiny demo dataset generator
- first runnable milestone: generate and inspect pose `.npz` files

### Day 2

- labels and fall-event spans
- converting annotations into `labels.json` and `spans.json`
- milestone: produce deterministic label artifacts

### Day 3

- subject-safe splits
- why split leakage is dangerous
- milestone: produce `train.txt`, `val.txt`, `test.txt`

### Day 4

- window generation
- overlap logic between windows and fall spans
- milestone: export training-ready window `.npz` files

### Day 5

- contract tests
- verifying split safety and window semantics
- milestone: pass Week 1 tests

### Day 6

- orchestration
- running the full Week 1 preprocessing pipeline with one command
- milestone: complete mini preprocessing pipeline

### Day 7

- output inspection
- sanity checks
- mapping the teaching build back to the real repository
- milestone: understand exactly what Week 2 will consume

## Week 1 final outcome

At the end of this week, students will be able to run a small but real preprocessing pipeline that produces:

- labeled videos
- fall spans
- leak-safe split lists
- fixed training windows

That becomes the direct input to Week 2, where students will build the dataset loader and train the first model.

## Teaching arc for the week

The sequence of days is deliberate.

Day 1 teaches data representation before supervision, because beginners need to know what one pose sequence actually looks like. Day 2 introduces labels and spans, which turns raw sequence files into supervised examples. Day 3 adds subject-safe split logic so evaluation later will mean something. Day 4 converts sequences into windows, which is the real reusable training contract in this repository. Day 5 adds tests so students see that pipeline guarantees should be checked, not assumed. Day 6 builds one orchestration script so the whole preprocessing stage can be rerun cleanly. Day 7 inspects the outputs and makes the Week 2 handoff explicit.

That is the Week 1 promise: by the end of the week, students do not just have files on disk. They understand why those files exist and why the architecture is organized that way.
