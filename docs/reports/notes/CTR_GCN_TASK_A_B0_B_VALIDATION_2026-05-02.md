# CTR-GCN Task A/B0/B Validation Note

Date: 2026-05-02  
Branch: `feat/ctr-gcn-upgrade`

## 1. Scope

This note records the first four accepted milestones for the `CTR-GCN` branch:

- Task A: lock the architecture contract
- Task B0: protect existing model families
- Task B: validate the `CTR-GCN` model I/O contract
- Task C: validate the standalone training script

It does **not** claim that `CTR-GCN` has been trained or evaluated as a project result.

## 2. Task A Outcome

Task A was completed by fixing the branch-wide definition of:

- public model description: `project-adapted single-stream CTR-GCN`
- external tensor layout: `[B, T, V, F]`
- internal tensor layout: `[B, F, T, V]`
- graph prior: MediaPipe 33-joint physical skeleton, self-loop augmented and symmetrically normalised
- topology refinement: hybrid
  - shared input-conditioned relation term
  - channel-wise low-rank refinement term
- first-pass baseline values:
  - `channels = (64, 64, 64, 128)`
  - `rel_channels = 8`
  - `ctr_rank = 8`
  - `temporal_kernel = 9`
  - `dropout = 0.30`

Reference:

- [CTR_GCN_ARCHITECTURE_CONTRACT_2026-05-02.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/CTR_GCN_ARCHITECTURE_CONTRACT_2026-05-02.md:1)

## 3. Task B0 Outcome: Legacy Protection

The branch was checked to ensure that the new `ctr_gcn` path did not break defended `TCN` and `custom GCN` build/load/forward behaviour.

### 3.1 Successful build/load smoke checks

- `outputs/caucafall_gcn_W48S12/best.pt` -> `arch = gcn`, model rebuilt and loaded successfully
- `outputs/_smoke/gcn/best.pt` -> `arch = gcn`, model rebuilt and loaded successfully
- `outputs/_smoke/tcn/best.pt` -> `arch = tcn`, model rebuilt and loaded successfully
- `outputs/caucafall_tcn_W48S12_r2_train_hneg/best.pt` -> `arch = tcn`, model rebuilt and loaded successfully

### 3.2 Successful forward smoke checks

- `outputs/caucafall_gcn_W48S12/best.pt` -> real window forward pass succeeded
- `outputs/caucafall_tcn_W48S12_r2_train_hneg/best.pt` -> real window forward pass succeeded

### 3.3 Interpretation

The shared-builder changes required for `ctr_gcn` support did not break the currently defended `TCN` and `custom GCN` paths.

## 4. Task B Outcome: CTR-GCN I/O Contract Validation

The first-pass `CTR-GCN` line was validated against the project’s canonical graph input contract.

### 4.1 Real sample forward check

Dataset surface:

- `data/processed/caucafall/windows_eval_W48_S12/val`

Observed sample:

- sample shape: `(48, 33, 5)`
- label shape: `(1,)`

Validation result:

- `build_model("ctr_gcn", model_cfg, feat_cfg, fps_default=23.0)` succeeded
- forward pass on `x.unsqueeze(0)` succeeded
- model output was a single logit

Observed output:

- `forward_ndim = 1`
- `forward_numel = 1`

This confirms that the current `CTR-GCN` line accepts the project’s external contract and emits the expected single-logit prediction shape.

### 4.2 Minimal checkpoint round-trip

A minimal synthetic `ctr_gcn` checkpoint bundle was created and reloaded successfully with:

- `arch = "ctr_gcn"`
- explicit `model_cfg`
- `feat_cfg`
- `data_cfg`
- `state_dict`

Reload result:

- reloaded model type: `CTRGCN`
- channel schedule preserved: `(64, 64, 64, 128)`
- `in_feats` preserved: `8`

### 4.3 Interpretation

The current `CTR-GCN` path:

- can be rebuilt through the shared builder
- respects the canonical single-stream graph input contract
- can be serialized and reloaded without hidden training-script defaults

## 5. Task C Outcome: Standalone Training-Script Validation

The first-pass `CTR-GCN` trainer was exercised on the existing window dataset format with a one-epoch smoke run.

### 5.1 Smoke command scope

Dataset surface:

- `data/processed/caucafall/windows_eval_W48_S12/train`
- `data/processed/caucafall/windows_eval_W48_S12/val`

Smoke output directory:

- `outputs/_smoke/ctr_gcn_W48S12_smoke`

### 5.2 Observed training result

The standalone trainer completed:

- dataset loading
- batch collation
- optimizer step
- validation pass
- metric computation
- checkpoint writing

Observed validation line:

- `train_loss=0.5150`
- `AP=0.9881`
- `AUC=0.9908`
- `F1=0.9412`
- `P=0.933`
- `R=0.949`
- `FPR=0.047`
- `thr=0.400`

### 5.3 Checkpoint bundle validation

Written files:

- `outputs/_smoke/ctr_gcn_W48S12_smoke/best.pt`
- `outputs/_smoke/ctr_gcn_W48S12_smoke/last.pt`
- `outputs/_smoke/ctr_gcn_W48S12_smoke/train_config.json`

The written `best.pt` bundle was checked and confirmed to contain:

- `arch = "ctr_gcn"`
- `model_cfg`
- `feat_cfg`
- `data_cfg`
- `state_dict`

The preserved architecture fields included:

- `channels = (64, 64, 64, 128)`
- `in_feats = 5`

### 5.4 Interpretation

Task C is complete.

The current `train_ctr_gcn.py` path is now shown to work end-to-end for:

- existing NPZ window loading
- model construction
- one smoke training run
- validation
- checkpoint persistence

## 6. Task D Outcome: Checkpoint Compatibility Validation

The smoke `CTR-GCN` checkpoint line was validated through the shared reload path.

Validated bundle:

- `outputs/_smoke/ctr_gcn_W48S12_smoke/best.pt`

Successful path:

- `load_ckpt()`
- `build_model(arch, model_cfg, feat_cfg, fps_default=...)`
- `load_state_dict(..., strict=True)`
- forward pass on a real validation window

Observed reload result:

- `arch = "ctr_gcn"`
- rebuilt model type: `CTRGCN`
- real-window forward pass succeeded

Interpretation:

- the new checkpoint family is downstream-compatible at the shared builder / loader level

## 7. Task E Outcome: `fit_ops` Compatibility Validation

The smoke checkpoint line was validated against the existing operating-point fitting flow.

Validated command surface:

- `fit_ops.py --arch ctr_gcn`

Successful artifacts:

- `configs/ops/ctr_gcn_caucafall_smoke.yaml`
- `configs/ops/ctr_gcn_caucafall_smoke.sweep.json`

Interpretation:

- `CTR-GCN` can enter the existing threshold/policy sweep path without architecture-specific patching at run time

## 8. Task F Outcome: Replay / Runtime Validation

The smoke checkpoint and fitted ops file were validated through the event-level alert-policy runner.

Validated command surface:

- `run_alert_policy.py --arch ctr_gcn`

Successful output:

- `outputs/_smoke/ctr_gcn_W48S12_smoke/alert_policy_val.json`

Observed event-level output:

- `Subject.9/Fall backwards`
- `Subject.9/Fall sitting`

Interpretation:

- the current `CTR-GCN` line already produces event-level bounded runtime output under the existing evaluation stack

## 9. Status After Task F

The branch is now ready to enter:

- Task G: first official `CAUCAFall` baseline

The branch is **not yet** ready to claim:

- project-level model performance
- report/paper promotion
