#!/usr/bin/env bash
set -euo pipefail

# Round-1 overfitting mitigation experiments (CAUCAFall priority).
# This script does NOT auto-run all commands by default.
# Run step-by-step and record status transitions with tools/track_experiment.py.

source .venv/bin/activate
export PYTHONPATH="$(pwd)/src:$(pwd)"

echo "[info] Round-1 command pack"
echo "[info] Use: planned -> running -> done/failed records for each EXP ID."

###############################################################################
# EXP 1: TCN baseline-refresh (control)
###############################################################################
# EXP_ID: exp_tcn_caucafall_r1_ctrl_seed33724876
python tools/track_experiment.py \
  --exp_id exp_tcn_caucafall_r1_ctrl_seed33724876 \
  --phase train --dataset caucafall --arch tcn --seed 33724876 \
  --status running \
  --changed_params "control_run;resume=best;mask_joint_p=0.05;mask_frame_p=0.05;dropout=0.30" \
  --command "make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_r1_ctrl"

make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_r1_ctrl
make fit-ops-caucafall ADAPTER_USE=1 OUT_TAG=_r1_ctrl ALERT_CONFIRM=0
make eval-caucafall ADAPTER_USE=1 OUT_TAG=_r1_ctrl

python tools/track_experiment.py \
  --exp_id exp_tcn_caucafall_r1_ctrl_seed33724876 \
  --phase eval --dataset caucafall --arch tcn --seed 33724876 \
  --status done \
  --changed_params "control_run;resume=best;mask_joint_p=0.05;mask_frame_p=0.05;dropout=0.30" \
  --command "make train/fit-ops/eval tcn caucafall OUT_TAG=_r1_ctrl" \
  --artifacts "outputs/caucafall_tcn_W48S12_r1_ctrl/best.pt;outputs/metrics/tcn_caucafall_r1_ctrl.json" \
  --metrics_json "outputs/metrics/tcn_caucafall_r1_ctrl.json" \
  --ops_yaml "configs/ops/tcn_caucafall_r1_ctrl.yaml"

###############################################################################
# EXP 2: TCN stronger regularization + masking
###############################################################################
# EXP_ID: exp_tcn_caucafall_r1_augreg_seed33724876
python tools/track_experiment.py \
  --exp_id exp_tcn_caucafall_r1_augreg_seed33724876 \
  --phase train --dataset caucafall --arch tcn --seed 33724876 \
  --status running \
  --changed_params "dropout=0.40;mask_joint_p=0.12;mask_frame_p=0.08;label_smoothing=0.03;weight_decay=1e-3" \
  --command "make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_r1_augreg TCN_DROPOUT=0.40 TCN_MASK_JOINT_P=0.12 TCN_MASK_FRAME_P=0.08 TCN_LABEL_SMOOTHING=0.03 TCN_WEIGHT_DECAY=0.001"

make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_r1_augreg \
  TCN_DROPOUT=0.40 TCN_MASK_JOINT_P=0.12 TCN_MASK_FRAME_P=0.08 \
  TCN_LABEL_SMOOTHING=0.03 TCN_WEIGHT_DECAY=0.001
make fit-ops-caucafall ADAPTER_USE=1 OUT_TAG=_r1_augreg ALERT_CONFIRM=0
make eval-caucafall ADAPTER_USE=1 OUT_TAG=_r1_augreg

python tools/track_experiment.py \
  --exp_id exp_tcn_caucafall_r1_augreg_seed33724876 \
  --phase eval --dataset caucafall --arch tcn --seed 33724876 \
  --status done \
  --changed_params "dropout=0.40;mask_joint_p=0.12;mask_frame_p=0.08;label_smoothing=0.03;weight_decay=1e-3" \
  --command "make train/fit-ops/eval tcn caucafall OUT_TAG=_r1_augreg" \
  --artifacts "outputs/caucafall_tcn_W48S12_r1_augreg/best.pt;outputs/metrics/tcn_caucafall_r1_augreg.json" \
  --metrics_json "outputs/metrics/tcn_caucafall_r1_augreg.json" \
  --ops_yaml "configs/ops/tcn_caucafall_r1_augreg.yaml"

###############################################################################
# EXP 3: GCN baseline-refresh (control)
###############################################################################
# EXP_ID: exp_gcn_caucafall_r1_ctrl_seed33724876
python tools/track_experiment.py \
  --exp_id exp_gcn_caucafall_r1_ctrl_seed33724876 \
  --phase train --dataset caucafall --arch gcn --seed 33724876 \
  --status running \
  --changed_params "control_run;mask_joint_p=0.05;mask_frame_p=0.05;dropout=0.30" \
  --command "make train-gcn-caucafall ADAPTER_USE=1 OUT_TAG=_r1_ctrl"

make train-gcn-caucafall ADAPTER_USE=1 OUT_TAG=_r1_ctrl
make fit-ops-gcn-caucafall ADAPTER_USE=1 OUT_TAG=_r1_ctrl ALERT_CONFIRM=0
make eval-gcn-caucafall ADAPTER_USE=1 OUT_TAG=_r1_ctrl

python tools/track_experiment.py \
  --exp_id exp_gcn_caucafall_r1_ctrl_seed33724876 \
  --phase eval --dataset caucafall --arch gcn --seed 33724876 \
  --status done \
  --changed_params "control_run;mask_joint_p=0.05;mask_frame_p=0.05;dropout=0.30" \
  --command "make train/fit-ops/eval gcn caucafall OUT_TAG=_r1_ctrl" \
  --artifacts "outputs/caucafall_gcn_W48S12_r1_ctrl/best.pt;outputs/metrics/gcn_caucafall_r1_ctrl.json" \
  --metrics_json "outputs/metrics/gcn_caucafall_r1_ctrl.json" \
  --ops_yaml "configs/ops/gcn_caucafall_r1_ctrl.yaml"

###############################################################################
# EXP 4: GCN stronger regularization + masking
###############################################################################
# EXP_ID: exp_gcn_caucafall_r1_augreg_seed33724876
python tools/track_experiment.py \
  --exp_id exp_gcn_caucafall_r1_augreg_seed33724876 \
  --phase train --dataset caucafall --arch gcn --seed 33724876 \
  --status running \
  --changed_params "dropout=0.40;mask_joint_p=0.12;mask_frame_p=0.08;label_smoothing=0.03;weight_decay=1e-3" \
  --command "make train-gcn-caucafall ADAPTER_USE=1 OUT_TAG=_r1_augreg GCN_DROPOUT=0.40 GCN_MASK_JOINT_P=0.12 GCN_MASK_FRAME_P=0.08 GCN_LABEL_SMOOTHING=0.03 GCN_WEIGHT_DECAY=0.001"

make train-gcn-caucafall ADAPTER_USE=1 OUT_TAG=_r1_augreg \
  GCN_DROPOUT=0.40 GCN_MASK_JOINT_P=0.12 GCN_MASK_FRAME_P=0.08 \
  GCN_LABEL_SMOOTHING=0.03 GCN_WEIGHT_DECAY=0.001
make fit-ops-gcn-caucafall ADAPTER_USE=1 OUT_TAG=_r1_augreg ALERT_CONFIRM=0
make eval-gcn-caucafall ADAPTER_USE=1 OUT_TAG=_r1_augreg

python tools/track_experiment.py \
  --exp_id exp_gcn_caucafall_r1_augreg_seed33724876 \
  --phase eval --dataset caucafall --arch gcn --seed 33724876 \
  --status done \
  --changed_params "dropout=0.40;mask_joint_p=0.12;mask_frame_p=0.08;label_smoothing=0.03;weight_decay=1e-3" \
  --command "make train/fit-ops/eval gcn caucafall OUT_TAG=_r1_augreg" \
  --artifacts "outputs/caucafall_gcn_W48S12_r1_augreg/best.pt;outputs/metrics/gcn_caucafall_r1_augreg.json" \
  --metrics_json "outputs/metrics/gcn_caucafall_r1_augreg.json" \
  --ops_yaml "configs/ops/gcn_caucafall_r1_augreg.yaml"

echo "[done] round-1 template finished"
