# 33-Joint Migration Checklist

## Goal
- Unify offline pipeline and deployment defaults to MediaPipe 33 joints.
- Keep legacy 17-joint checkpoints runnable in deployment (compatibility only).

## Why
- Current mismatch (`ADAPTER_USE=1` producing 17-joint windows vs frontend/runtime 33-joint input) causes train/eval/deploy distribution drift.
- One canonical joint layout reduces false alarm debugging complexity.

## Scope
- Data window generation adapter defaults and flags.
- Makefile adapter flags passed into windows generation targets.
- Adapter contract tests.
- No broad model refactor; no retraining in this task.

## Step-by-step
1. Add explicit adapter joint-layout flag
- File: `src/fall_detection/data/adapters/base.py`
- Add `joint_layout` option to adapter factory and adapter instances:
  - `mp33` (default): pass through 33 joints
  - `internal17`: map to legacy 17

2. Wire layout flag into window generation
- File: `src/fall_detection/data/windowing/make_windows_impl.py`
- Add CLI arg: `--adapter_joint_layout {mp33,internal17}` (default `mp33`)
- Pass through to `build_adapter(...)`.

3. Wire layout flag into Makefile adapter flags
- File: `Makefile`
- Add:
  - `ADAPTER_JOINT_LAYOUT ?= mp33`
  - `ADAPTER_FLAGS += --adapter_joint_layout "$(ADAPTER_JOINT_LAYOUT)"`

4. Update adapter tests
- File: `tests/test_adapter_contract.py`
- Default adapter call should assert 33-joint output.
- Add explicit `internal17` case to verify legacy compatibility.

5. Validation
- `python -m py_compile src/fall_detection/data/adapters/base.py src/fall_detection/data/windowing/make_windows_impl.py`
- `python -m py_compile scripts/make_windows.py`
- `pytest -q tests/test_adapter_contract.py`

## Operational notes
- Existing 17-joint checkpoints can still run online via runtime alignment logic (33->17 compatibility path).
- New training/eval windows generated with adapter mode should now be 33-joint by default.
- If you need legacy reproduction, run:
  - `make windows-<ds> ADAPTER_USE=1 ADAPTER_JOINT_LAYOUT=internal17`

## Rollback
- Set `ADAPTER_JOINT_LAYOUT=internal17` in command line or Makefile override to restore previous behavior.
