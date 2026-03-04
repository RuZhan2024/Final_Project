# Training Upgrades Task Plan (TCN + GCN Only)

## What Changed in This Plan
- Clarified `scheduler_metric` explicitness handling: argparse default must be `None`, then auto-resolve from `monitor` only when unset.
- Added canonical checkpoint schema requirement in `ckpt.py` with explicit legacy compatibility normalization.
- Added checkpoint schema marker requirement (`ckpt_version`, e.g. `2`) for forward debugging.
- Added atomic checkpoint write requirement (`.tmp` + `os.replace`) to avoid partial/corrupt ckpts.
- Added optional enhancement: `last.pt` raw weights + `best.pt` EMA weights (when EMA enabled).
- Added explicit EMA resume ordering to avoid ambiguous initialization/restore behavior.
- Clarified resume-state policy to avoid half-implemented optimizer/scheduler/scaler resume.
- Clarified deterministic flag behavior: seeding is always applied; only cuDNN toggles change.
- Expanded smoke tests: include old checkpoint (without `ema_state`), both save styles, wrappers compile checks, and EMA-effect check.
- Expanded docs deliverable requirements for scheduler auto-default, EMA best checkpoint behavior, and backward compatibility guarantees.

## Scope Confirmation
- Target files are in this repo under:
  - `src/fall_detection/training/train_tcn.py`
  - `src/fall_detection/training/train_gcn.py`
  - `src/fall_detection/core/ckpt.py`
  - `src/fall_detection/core/ema.py`
- Thin wrappers exist at:
  - `scripts/train_tcn.py`
  - `scripts/train_gcn.py`
- No model-family additions (still TCN + GCN only).
- This plan updates training correctness/stability/reproducibility behavior only.

## Objectives
1. Fix GCN EMA checkpoint correctness + resume consistency.
2. Add TCN `--min_epochs` parity for early stopping.
3. Add GCN dataloader throughput knobs (`prefetch_factor`, `persistent_workers`) safely.
4. Add `--deterministic` switch to both trainers (without affecting seeding).
5. Auto-align scheduler metric with monitor **only when user does not explicitly pass `--scheduler_metric`**.
6. Standardize checkpoint schema behavior in `ckpt.py` with backward compatibility.
7. Add short usage doc + minimal smoke validation.

---

## Phase-by-Phase Execution Plan

### Phase 1A — GCN EMA Checkpoint Correctness + Checkpoint Schema (Must)

### Tasks
- [ ] Audit current `train_gcn.py` save/resume flow:
  - confirm how `resume_bundle` is handled.
  - confirm current call-style to `save_ckpt`.
  - confirm whether optimizer/scheduler/scaler resume is currently implemented.
- [ ] Define canonical checkpoint schema in `src/fall_detection/core/ckpt.py`:
  - canonical write path: `save_ckpt(path, **kwargs)` (kwargs style, same direction as TCN).
  - legacy accepted path: `save_ckpt(path, bundle_dict)`.
  - normalize both internally to one consistent on-disk key-set.
  - include schema marker in bundle, e.g. `ckpt_version: 2`.
- [ ] Ensure `load_ckpt()` returns a consistent bundle dict shape regardless of old/new checkpoint origin.
- [ ] Make checkpoint writes atomic in `ckpt.py`:
  - write to `path.tmp` first, then `os.replace(tmp, path)`.
- [ ] Implement EMA-aware best checkpoint save in `train_gcn.py`:
  - under `with (ema.use(model) if ema else nullcontext()):` capture `state_dict` for `best.pt`.
  - include `ema_state` in bundle when EMA enabled.
- [ ] Implement EMA resume restore in `train_gcn.py` with explicit ordering:
  1) load checkpoint bundle
  2) restore model weights
  3) create EMA object (if enabled)
  4) load `ema_state` into EMA if present (try/except warn, no crash)
  5) restore optimizer/scheduler/scaler states if implemented in trainer; otherwise explicitly keep as not implemented
- [ ] Resume-state clarity requirement:
  - if optimizer/scheduler/scaler resume currently works, preserve that behavior.
  - if it is not implemented today, explicitly document: only model (+ EMA) weights are resumed.
- [ ] Backward compatibility guard:
  - old checkpoints without `ema_state` must load/resume without exception.

### Optional enhancement (recommended)
- [ ] Save `last.pt` raw model weights (debugging/reference), while keeping `best.pt` as EMA-averaged weights when EMA is enabled.

### Acceptance Criteria
- Old checkpoints (without `ema_state`) load without exception.
- New checkpoints include `ema_state` when EMA enabled.
- `best.pt` from GCN contains EMA weights when EMA enabled.
- Resume does not depend on `ema_state` existing.
- EMA state loading happens only after model is instantiated and weights are restored.
- GCN and TCN use the same `save_ckpt` style after upgrade (preferred), OR `ckpt.py` fully supports both styles with normalization.
- Saving via `save_ckpt(path, **kwargs)` and `save_ckpt(path, bundle_dict)` produces the same normalized on-disk keys.
- Checkpoint files are written atomically (no partial file on interruption at write boundary).
- Optional path: if implemented, `last.pt` is raw weights and `best.pt` is EMA weights when EMA enabled.

### Risk
- Medium: checkpoint schema consistency and resume-state ordering.

---

### Phase 1B — TCN `min_epochs` Early-Stop Gate (Must)

### Tasks
- [ ] Add `min_epochs: int = 0` to TCN config/dataclass.
- [ ] Add argparse flag `--min_epochs` in `train_tcn.py`.
- [ ] Update early-stop condition so patience-triggered stop can only happen when `ep >= min_epochs`.

### Acceptance Criteria
- Passing `--min_epochs 20` prevents patience-based early stop before epoch 20.
- Existing behavior preserved when `--min_epochs 0`.

### Risk
- Low.

---

### Phase 2C — GCN DataLoader Throughput Knobs (Should)

### Tasks
- [ ] Add args to `train_gcn.py`:
  - `--prefetch_factor`
  - `--persistent_workers`
- [ ] Add corresponding config fields.
- [ ] Add guarded loader kwargs:
  - pass `persistent_workers/prefetch_factor` only when `num_workers > 0`.
- [ ] Keep `pin_memory` behavior CUDA-aware (`pin_memory=True` on CUDA, else False).

### Acceptance Criteria
- `num_workers=0` path does not pass invalid DataLoader args.
- `num_workers>0` path uses prefetch/persistent worker options.

### Risk
- Low.

---

### Phase 2D — Deterministic Switch for TCN + GCN (Should)

### Tasks
- [ ] Add `--deterministic` (int, default 1) to both trainers.
- [ ] Ensure seed application is unconditional:
  - always call `set_seed(seed)` regardless of deterministic flag.
- [ ] Deterministic flag only controls cuDNN mode:
  - `deterministic==1` => `cudnn.deterministic=True`, `cudnn.benchmark=False`
  - `deterministic==0` => `cudnn.deterministic=False`, `cudnn.benchmark=True`

### Acceptance Criteria
- Same seed behavior applies in both modes.
- Only determinism/performance cuDNN behavior toggles with the flag.

### Risk
- Low.

---

### Phase 3E — Scheduler/Monitor Default Alignment with Explicitness Detection (Recommended)

### Tasks
- [ ] Make `--scheduler_metric` argparse default `None` in both trainers.
- [ ] Keep user-facing choices restricted to:
  - `{val_loss, val_f1, val_ap}`
  - internal `None` is only default sentinel for auto-resolution.
- [ ] After arg parsing, resolve metric as follows:
  - if `scheduler_metric is None`:
    - `monitor=ap` => `scheduler_metric=val_ap`
    - `monitor=f1` => `scheduler_metric=val_f1`
  - else: keep user-provided `scheduler_metric` unchanged.
- [ ] Update help/docs wording to: default is **auto (resolved from monitor)**.

### Acceptance Criteria
- Passing `--scheduler_metric val_loss` keeps `val_loss` even if `monitor=ap`.
- Omitting `--scheduler_metric` yields:
  - `monitor=ap` => `val_ap`
  - `monitor=f1` => `val_f1`

### Risk
- Medium: must avoid accidentally overriding explicit user selection.

---

## Testing Plan (No Full Training)

### Required compile checks
- [ ] `python -m py_compile src/fall_detection/training/train_tcn.py`
- [ ] `python -m py_compile src/fall_detection/training/train_gcn.py`
- [ ] `python -m py_compile src/fall_detection/core/ckpt.py` (if edited)
- [ ] `python -m py_compile src/fall_detection/core/ema.py` (if edited)
- [ ] `python -m py_compile scripts/train_tcn.py`
- [ ] `python -m py_compile scripts/train_gcn.py`

### Minimal runtime smoke (no dataset)
- [ ] Add either:
  - `tests/test_ckpt_ema_schema.py` (preferred if test framework is active), or
  - `tools/smoke_ckpt_ema.py`
- [ ] Cover ALL required cases:
  - A) save+load checkpoint **with** `ema_state`
  - B) load checkpoint **without** `ema_state` (simulate old bundle) and confirm no exception
  - C) save via legacy dict style and load back (no crash; normalized keys match canonical style)
  - D) EMA-effect sanity: update model + EMA so EMA/raw differ, then confirm `best.pt` (EMA path) and `last.pt` (raw path, if optional enhancement implemented) can differ as expected

### Optional sanity
- [ ] `python scripts/train_tcn.py --help`
- [ ] `python scripts/train_gcn.py --help`

### Acceptance Criteria
- All checkpoint smoke cases run without exception.
- Both save styles result in consistent normalized key-set after load.
- All edited files and wrappers compile.

---

## Deliverables
- [ ] Git patch (clean and applies cleanly).
- [ ] `docs/training_upgrades.md` including:
  - what changed
  - new flags and defaults
  - **scheduler_metric default is auto (resolved from monitor)**
  - **best.pt uses EMA weights when EMA is enabled**
  - resume behavior for EMA
  - backward compatibility guarantees for old checkpoints (including missing `ema_state`)
- [ ] Smoke-check command list and outputs summary.
- [ ] Final modified-files list and focused diff summary.

---

## Notes on Backward Compatibility
- Old checkpoints remain loadable without `ema_state`.
- `save_ckpt` accepts canonical kwargs style and legacy dict style; both normalized internally.
- Explicit user CLI selections must always win over auto-derived defaults.
