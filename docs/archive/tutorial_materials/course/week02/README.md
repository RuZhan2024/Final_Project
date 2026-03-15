# Week 2: From Windows To A Trainable Fall Detection Model

## Why this week matters

Week 1 produced the artifact contract that the rest of the repository relies on:

- fixed-size window files
- per-window labels
- train/val/test separation
- metadata such as `video_id`, `w_start`, `w_end`, and `fps`

Week 2 starts exactly there.

This week teaches the next real execution path of the repository:

`window files -> tensors -> dataloaders -> model -> training loop -> validation loop -> checkpoint`

That is the first point where the project becomes a genuine learning system instead of a preprocessing pipeline.

## What students will build this week

By the end of Week 2, students will have written these `course_project/` files by copying code from the daily lessons:

- `course_project/core/__init__.py`
- `course_project/core/features.py`
- `course_project/core/ckpt.py`
- `course_project/models/__init__.py`
- `course_project/models/tcn.py`
- `course_project/training/__init__.py`
- `course_project/training/data.py`
- `course_project/training/metrics.py`
- `course_project/training/train_tcn.py`
- `course_project/tests/test_features.py`
- `course_project/tests/test_checkpoint.py`
- `course_project/scripts/inspect_week2_outputs.py`

Important teaching note:

- lesson markdown remains the source of truth
- students still copy code from the notes into `course_project/`
- there is no parallel packaged teaching codebase

## How Week 2 maps to the real repository

This week is aligned to the active repository path through:

- `src/fall_detection/core/features.py`
- `src/fall_detection/core/ckpt.py`
- `src/fall_detection/core/models.py`
- `src/fall_detection/training/train_tcn.py`
- `scripts/train_tcn.py`

Teaching simplification:

- one model family
- one clean feature layout
- one small checkpoint schema
- one beginner-readable training loop

Full project reality:

- richer feature flags
- more training controls
- more losses and monitoring modes
- multiple architectures

## Daily lesson map

### Day 1

- what one window sample means
- feature tensors
- canonical shapes
- milestone: load one window and build one model-ready tensor

### Day 2

- dataset and dataloader construction
- batching windows
- metadata flow
- milestone: build one batch and inspect its shapes

### Day 3

- the first temporal model
- a small teaching TCN
- milestone: run one forward pass

### Day 4

- metrics and checkpointing
- why checkpoints must store configuration, not just weights
- milestone: save and reload a checkpoint bundle

### Day 5

- full training loop
- validation loop
- first real training run
- milestone: save `best.pt` and `metrics.json`

### Day 6

- tests and debugging
- shape checks
- checkpoint roundtrip checks
- milestone: pass Week 2 tests

### Day 7

- inspect the trained artifacts
- explain what Week 3 will consume
- milestone: understand model outputs and the handoff to thresholding

## Week 2 final outcome

At the end of this week, students will have:

- a working dataset loader
- a simple TCN model
- a complete training script
- a saved checkpoint bundle
- training metrics written to disk

That becomes the direct input to Week 3, where the course will teach thresholds, operating points, and event-level evaluation.
