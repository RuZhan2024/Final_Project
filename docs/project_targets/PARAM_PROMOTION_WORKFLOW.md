# Parameter Promotion Workflow

Use this when an optimization run is validated and you want to promote it into code defaults.

## 1) Prepare a promotion profile

Edit:
- `artifacts/reports/tuning/tcn_promotion_profile.json`

Fields:
- `ops_yaml`: source ops yaml to copy into canonical `configs/ops/tcn_caucafall.yaml`
- `tcn_defaults`: values to write into `Makefile` defaults

## 2) Apply promotion

```bash
python tools/promote_tcn_defaults.py \
  --profile_json artifacts/reports/tuning/tcn_promotion_profile.json
```

## 3) Verify

```bash
rg -n "^TCN_(DROPOUT|MASK_JOINT_P|MASK_FRAME_P|WEIGHT_DECAY|LABEL_SMOOTHING|HARD_NEG_LIST|HARD_NEG_MULT|RESUME)\\s*\\?=" Makefile
python - <<'PY'
import yaml
d=yaml.safe_load(open("configs/ops/tcn_caucafall.yaml"))
print(d["model"]["ckpt"])
print(d["ops"]["OP2"]["tau_high"], d["ops"]["OP2"]["tau_low"])
PY
```

## 4) Re-run lock checks

```bash
bash tools/run_deployment_lock_validation.sh
python tools/check_release_bundle.py
```

## 5) Record + commit

- Update:
  - `artifacts/reports/release_snapshot.md`
  - `docs/project_targets/DEPLOYMENT_LOCK.md`
  - `docs/project_targets/THESIS_EVIDENCE_MAP.md` (if needed)
- Commit with message like:
  - `promote(tcn-caucafall): update Makefile defaults and canonical ops from exp_xxx`
