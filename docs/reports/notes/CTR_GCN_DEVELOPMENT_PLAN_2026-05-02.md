# CTR-GCN Development Plan

Date: 2026-05-02  
Branch: `feat/ctr-gcn-upgrade`

## 1. Purpose

This document defines the development plan for adding a new `CTR-GCN` model line to the project.

The goal is **not** to modify or replace the current `TCN` or `custom GCN` pipelines.  
The goal is to add a **separate, modern graph-model path** that can be trained, evaluated, and compared under the same project protocol.

## 2. Project Constraints

The implementation must satisfy all of the following:

1. The existing `TCN` path must remain unchanged.
2. The existing `custom GCN` path must remain unchanged.
3. `CTR-GCN` must be added as a separate model family, not as a silent mutation of the current GCN code.
4. The new path must work with the existing feature contract:
   - canonical input shape `[T, V, F]`
   - current window NPZ pipeline
   - current checkpoint bundle format
5. The new path must support the existing downstream workflow:
   - training
   - checkpoint loading
   - `fit_ops`
   - replay/runtime evaluation
6. Initial development should prioritise:
   - correctness
   - compatibility
   - controlled comparison
   over exploratory optimisation.

## 3. Recommended Solution

The recommended solution is:

- add a **standalone `CTR-GCN` model module**
- add a **standalone training script**
- add **minimal shared-builder support** so evaluation and deployment tooling can load the new model family
- keep the current `TCN` and `custom GCN` training scripts intact

This is better than modifying the current GCN implementation because it:

- preserves backward compatibility
- avoids contaminating the defended `custom GCN` baseline
- makes the new graph line easier to compare and reason about
- allows the project to report `custom GCN` and `CTR-GCN` as distinct model families

## 4. Current Status

A minimal implementation scaffold has already been started on this branch:

- `ml/src/fall_detection/core/ctr_gcn.py`
- `ml/src/fall_detection/training/train_ctr_gcn.py`
- minimal shared support in:
  - `ml/src/fall_detection/core/models.py`
  - `ml/src/fall_detection/evaluation/fit_ops.py`
  - `ml/src/fall_detection/deploy/common.py`
  - `ml/src/fall_detection/deploy/run_alert_policy.py`

This scaffold currently means:

- the new model family can be constructed by the shared builder
- the new training script exists
- the new path is syntax-valid

This does **not** yet mean:

- the architecture is final
- the training pipeline is fully validated
- the evaluation flow has been fully exercised on real data
- the new line is ready for report/paper use

Further work should follow the task sequence below.

Current milestone status:

- Task A: completed
- Task B0: completed
- Task B: completed
- Task C: completed
- Task D: completed
- Task E: completed
- Task F: completed
- Task G and beyond: not yet started

## 5. Development Tasks

### Task A. Lock the CTR-GCN architecture contract

Goal:
- define the exact project-level meaning of `CTR-GCN`
- avoid creating a model that only carries the CTR-GCN name without a clear topology-refinement mechanism

Required decisions:

1. Implementation status
   - The first version will be a **project-adapted, single-stream CTR-GCN line**.
   - It must not be described as a full reproduction of the original CTR-GCN paper unless the final architecture is shown to be closely matched and validated.

2. Input contract
   - External input shape: `[B, T, V, F]`
   - Internal layout: `[B, F, T, V]`
   - No multi-person `M` dimension in the first pass
   - Feature dimension `F` must be inferred from the existing `feat_cfg`

3. Graph / topology contract
   - Define the base skeleton adjacency used by the model
   - Define whether the graph begins from physical skeleton edges, identity-plus-neighbour structure, or another fixed prior
   - Define how channel-wise topology refinement is represented
   - Define whether refinement is static learned, input-dependent, or hybrid
   - Ensure the implementation is visibly distinct from the current custom GCN

4. Block contract
   - Define the number of CTR-GCN blocks
   - Define base channel width
   - Define temporal kernel size
   - Define residual connection behaviour
   - Define dropout default
   - Define normalisation layers
   - Define classifier head

5. Checkpoint contract
   - `arch` must be saved as `"ctr_gcn"`
   - `model_cfg` must contain all architecture parameters needed to rebuild the model
   - `feat_cfg` and `data_cfg` must be preserved
   - Loading must not require manual arguments outside the checkpoint bundle

Output:
- one stable `CTRGCNConfig`
- one short architecture summary in code comments or development notes

The architecture contract must explicitly specify:

- public model description
- external and internal tensor layout
- graph adjacency prior
- channel-wise topology refinement design
- block count and channel schedule
- temporal kernel size
- residual behaviour
- dropout and normalisation
- classifier head
- checkpoint rebuild fields

Done when:
- `CTRGCNConfig` is explicit enough that another developer can rebuild the same model without reading the training script
- the model is clearly separate from the existing custom GCN
- the project can honestly describe the model as a **project-adapted CTR-GCN line**

### Task B0. Protect existing model families

Goal:
- prove that adding `CTR-GCN` has not changed the existing `TCN` or `custom GCN` paths

Checks:
- existing `TCN` checkpoint can still be loaded
- existing `custom GCN` checkpoint can still be loaded
- `build_model("tcn", ...)` behaviour is unchanged
- `build_model("gcn", ...)` behaviour is unchanged
- no existing training or deployment entry point changes semantics

Output:
- one smoke-check note confirming legacy paths still work

Done when:
- the current defended `TCN` and `custom GCN` lines are shown to load and build successfully after the `CTR-GCN` changes

### Task B. Validate model I/O contract

Goal:
- prove that `CTR-GCN` consumes the same canonical input contract as the current graph models

Checks:
- input shape `[B, T, V, F]`
- feature dimension inferred correctly from `feat_cfg`
- checkpoint serialization stores enough information to rebuild the model

Output:
- one successful build-and-forward validation

Done when:
- the model can be created from `build_model("ctr_gcn", ...)`
- a forward pass works on sample canonical tensors

### Task C. Validate the standalone training script

Goal:
- prove that `train_ctr_gcn.py` can train on the existing window dataset format

Checks:
- dataset loading
- batch collation
- optimizer step
- validation pass
- checkpoint saving

Output:
- one short smoke-test training run

Done when:
- training reaches at least one validation cycle
- `best.pt` and `last.pt` are written correctly

### Task D. Validate checkpoint compatibility

Goal:
- ensure the new checkpoint line is usable by downstream tooling

Checks:
- `arch="ctr_gcn"` saved in bundle
- `model_cfg` includes explicit input dimension
- `feat_cfg` and `data_cfg` are persisted

Output:
- one checkpoint bundle that reloads cleanly

Done when:
- `load_ckpt()` + `build_model()` + `load_state_dict()` work without manual patching

### Task E. Validate `fit_ops` compatibility

Goal:
- ensure `CTR-GCN` can enter the existing alert-policy pipeline

Checks:
- `fit_ops.py --arch ctr_gcn`
- window inference
- threshold sweep
- YAML output

Output:
- one successful `fit_ops` run on validation data

Done when:
- ops YAML is emitted successfully
- no architecture-specific mismatch blocks the sweep

### Task F. Validate replay/runtime evaluation compatibility

Goal:
- ensure `CTR-GCN` can be measured on the same bounded runtime surfaces as the current lines

Checks:
- metrics evaluation
- replay evaluation
- deploy-time probability path
- alert-policy runner

Output:
- one successful bounded replay evaluation

Done when:
- `CTR-GCN` produces event-level output under the current evaluation stack

### Task G. Establish the first official baseline

Goal:
- create the first real `CTR-GCN` baseline on `CAUCAFall`

Scope:
- single dataset only
- controlled configuration
- no large sweep yet

Output:
- one trained baseline checkpoint
- one metrics JSON
- one `fit_ops` output

Current execution note:
- baseline preparation may be completed on the current Intel/macOS development machine
- full baseline training should be run on a stronger NVIDIA machine
- see [CTR_GCN_BASELINE_PREP_2026-05-02.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/CTR_GCN_BASELINE_PREP_2026-05-02.md:1)

Done when:
- the project has a stable first `CTR-GCN` line that can be compared with `TCN` and `custom GCN`

### Task H. Run the first controlled comparison

Goal:
- compare:
  - `TCN`
  - `custom GCN`
  - `CTR-GCN`

Scope:
- same dataset
- same window contract
- same evaluation protocol

Output:
- one comparison note or table

Done when:
- we can state whether `CTR-GCN` is:
  - worse than `TCN`
  - better than the current `custom GCN`
  - worth promoting to the main graph line

### Task I. Small high-value tuning round

Goal:
- improve `CTR-GCN` only after the baseline is stable

Priority tuning dimensions:
1. batch size
2. learning rate
3. dropout
4. weight decay
5. base channels
6. number of blocks

Output:
- 1 to 2 stronger candidate configs

Done when:
- at least one tuned candidate is defensibly better than the baseline

### Task J. Promotion decision

Goal:
- decide whether `CTR-GCN` should become the new graph-model line used in future comparisons and writing

Decision rule:
- promote only if it is clearly stronger than the current `custom GCN`
- ideally it should also be stable enough to support runtime evaluation

Output:
- one explicit decision:
  - promote
  - keep exploratory
  - abandon

Done when:
- the model line’s status is clear enough for report/paper integration

## 6. Recommended Execution Order

The work should proceed in this order:

1. Task A
2. Task B0
3. Task B
4. Task C
5. Task D
6. Task E
7. Task F
8. Task G
9. Task H
10. Task I
11. Task J

## 7. Out of Scope for the First Pass

The following should **not** be done in the first pass:

- rewriting the current `custom GCN`
- changing the current `TCN`
- adding multiple new graph families at once
- doing large multi-dataset sweeps before the baseline is stable
- report/paper claims before the model is actually validated
- directly importing or vendoring the full official CTR-GCN training stack into this repo

## 8. Immediate Next Step

The immediate next step should be:

**Task A: Lock the architecture contract**

Reason:
- the rest of the work depends on a stable definition of what `CTR-GCN` means in this project

## 9. Required Evidence Log

Each completed task must leave evidence in one or more of the following forms:

- command used
- output path
- checkpoint path
- metrics JSON path
- ops YAML path
- short result note
- known limitation note

No task should be marked done only because the code runs once interactively.

## 10. Review Gate

This plan is accepted as the roadmap for the `CTR-GCN` branch.

However, implementation beyond Task A must not proceed until Task A produces an explicit architecture contract.

Development may proceed to Task B0 and later tasks only when:

1. the `CTR-GCN` variant is clearly named
2. the input/output contract is fixed
3. the graph topology/refinement mechanism is defined
4. legacy `TCN` and `custom GCN` paths are protected by smoke checks
5. the first-pass scope remains limited to one controlled baseline
