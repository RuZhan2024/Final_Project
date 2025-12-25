# -----------------------------------------
# Makefile — Fall Detection v2 (clean)
# Datasets: LE2i, UR_Fall, CAUCAFall, MUVIM
# Flow: extract -> preprocess -> labels -> splits -> windows -> train -> fit_ops -> eval
# -----------------------------------------

SHELL := /bin/bash

.DEFAULT_GOAL := help

# -------------------------
# Python / venv
# -------------------------
PY   := python3
VENV := source ".venv/bin/activate"

# Ensure repo root is on Python import path (so `import core` works)
PYTHONPATH := $(CURDIR)
export PYTHONPATH

RUN  := $(VENV) && PYTHONPATH="$(PYTHONPATH)" $(PY)
# -------------------------
# Global knobs
# -------------------------
SPLIT_SEED ?= 33724876
TRAIN_FRAC ?= 0.80
VAL_FRAC   ?= 0.10
TEST_FRAC  ?= 0.10

# Windows (default: W32 S8)
WIN_W ?= 48
WIN_S ?= 12

# Pose preprocess knobs
CONF_THR   ?= 0.20
SMOOTH_K   ?= 5
MAX_GAP    ?= 4
NORM_MODE  ?= torso
PELVIS_FILL?= nearest

# Common train knobs
EPOCHS ?= 100
# -------------------------
# GCN training knobs
# -------------------------
OUT_TAG ?=
LR_GCN ?= $(LR)
EPOCHS_GCN ?= $(EPOCHS)
BATCH_GCN ?= $(BATCH)

GCN_PATIENCE ?= 12
GCN_GCN_HIDDEN ?= 96
GCN_TCN_HIDDEN ?= 192
GCN_DROPOUT ?= 0.35
GCN_USE_SE ?= 1
GCN_TWO_STREAM ?= 1
GCN_FUSE ?= concat

MASK_JOINT_P ?= 0.15
MASK_FRAME_P ?= 0.10
GCN_POS_WEIGHT ?= auto
GCN_BALANCED_SAMPLER ?= 0

GCN_WEIGHT_DECAY ?= 1e-4
GCN_MONITOR ?= f1
GCN_LABEL_SMOOTHING ?= 0.0
GCN_NUM_WORKERS ?= 0


# Threshold sweep (used inside train_gcn.py for val metric selection)
GCN_THR_MIN ?= 0.01
GCN_THR_MAX ?= 0.95
GCN_THR_STEP ?= 0.05
GCN_MIN_EPOCHS ?= 5

# Dataset-specific overrides (keep defaults for most, stabilise MUVIM)
LR_GCN_LE2I ?= $(LR_GCN)
LR_GCN_URFD ?= $(LR_GCN)
LR_GCN_CAUC ?= $(LR_GCN)
LR_GCN_MUVIM ?= 3e-4

GCN_MONITOR_LE2I ?= $(GCN_MONITOR)
GCN_MONITOR_URFD ?= $(GCN_MONITOR)
GCN_MONITOR_CAUC ?= $(GCN_MONITOR)
GCN_MONITOR_MUVIM ?= ap

GCN_THR_STEP_MUVIM ?= 0.01
GCN_DROPOUT_MUVIM ?= 0.20
MASK_JOINT_P_MUVIM ?= 0.05
MASK_FRAME_P_MUVIM ?= 0.02
GCN_POS_WEIGHT_MUVIM ?= none
GCN_BALANCED_SAMPLER_MUVIM ?= 1
GCN_BALANCED_SAMPLER_FLAG_MUVIM = $(if $(filter 1,$(GCN_BALANCED_SAMPLER_MUVIM)),--balanced_sampler,)

BATCH  ?= 128
LR     ?= 1e-3

# -------------------------
# Training knobs (new trainers)
# -------------------------
TCN_HIDDEN ?= 128
TCN_DROPOUT ?= 0.30
TCN_NUM_BLOCKS ?= 4
TCN_KERNEL ?= 3
TCN_PATIENCE ?= 12
TCN_GRAD_CLIP ?= 1.0
TCN_POS_WEIGHT ?= auto
TCN_BALANCED_SAMPLER ?= 0
TCN_MASK_JOINT_P ?= 0.15
TCN_MASK_FRAME_P ?= 0.10
TCN_MONITOR ?= f1

# Dataset overrides (stabilise LE2i/MUVIM)
LR_TCN_LE2I ?= $(LR)
LR_TCN_URFD ?= $(LR)
LR_TCN_CAUC ?= $(LR)
LR_TCN_MUVIM ?= 3e-4

TCN_MONITOR_LE2I ?= ap
TCN_MONITOR_URFD ?= f1
TCN_MONITOR_CAUC ?= f1
TCN_MONITOR_MUVIM ?= ap

TCN_DROPOUT_MUVIM ?= 0.20
TCN_MASK_JOINT_P_MUVIM ?= 0.05
TCN_MASK_FRAME_P_MUVIM ?= 0.03
TCN_POS_WEIGHT_MUVIM ?= none
TCN_BALANCED_SAMPLER_MUVIM ?= 1

GCN_GRAD_CLIP ?= 1.0

# Alert policy defaults (real-time)
ALERT_EMA_ALPHA ?= 0.20
ALERT_K ?= 2
ALERT_N ?= 3
ALERT_TAU_HIGH ?= 0.90
ALERT_TAU_LOW ?= 0.70
ALERT_COOLDOWN_S ?= 30

# --- fit_ops threshold sweep (tau_high is FITTED on val; tau_low = tau_low_ratio * tau_high) ---
FIT_THR_MIN ?= 0.01
FIT_THR_MAX ?= 0.95
FIT_THR_STEP ?= 0.01
FIT_TAU_LOW_RATIO ?= 0.60

# Event grouping + matching (seconds)
FIT_TIME_MODE ?= center  # start|center|end mapping from window index → time
FIT_MERGE_GAP_S ?= 1.0  # merge predicted alert events if gap <= this
FIT_OVERLAP_SLACK_S ?= 0.5  # slack when matching predicted alerts to GT falls

# OP selection targets (tune per dataset if needed)
FIT_OP1_RECALL ?= 0.95  # High Safety: pick first OP with recall >= this (if possible)
FIT_OP3_FA24H ?= 1.0  # Low Alarms: pick first OP with FA/24h <= this (if possible)

# fit_ops composed flags
FITOPS_POLICY_FLAGS = --ema_alpha "$(ALERT_EMA_ALPHA)" --k "$(ALERT_K)" --n "$(ALERT_N)" --cooldown_s "$(ALERT_COOLDOWN_S)" --tau_low_ratio "$(FIT_TAU_LOW_RATIO)"
FITOPS_SWEEP_FLAGS  = --thr_min "$(FIT_THR_MIN)" --thr_max "$(FIT_THR_MAX)" --thr_step "$(FIT_THR_STEP)" --time_mode "$(strip $(FIT_TIME_MODE))" --merge_gap_s "$(strip $(FIT_MERGE_GAP_S))" --overlap_slack_s "$(strip $(FIT_OVERLAP_SLACK_S))" --op1_recall "$(strip $(FIT_OP1_RECALL))" --op3_fa24h "$(strip $(FIT_OP3_FA24H))"


# -------------------------
# Project dirs
# -------------------------
RAW_DIR    := data/raw
INTERIM    := data/interim
PROCESSED  := data/processed
OUT_DIR    := outputs
CFG_DIR    := configs
LABELS_DIR := $(CFG_DIR)/labels
SPLITS_DIR := $(CFG_DIR)/splits
OPS_DIR    := $(CFG_DIR)/ops
REPORTS_DIR := $(OUT_DIR)/reports
FIG_DIR     := $(OUT_DIR)/figures

# -------------------------
# Raw dataset locations
# -------------------------
RAW_LE2I  := $(RAW_DIR)/LE2i
RAW_URFD  := $(RAW_DIR)/UR_Fall_clips
RAW_CAUC  := $(RAW_DIR)/CAUCAFall
RAW_MUVIM := $(RAW_DIR)/MUVIM

# -------------------------
# FPS defaults
# -------------------------
FPS_LE2I  ?= 25
FPS_URFD  ?= 30
FPS_CAUC  ?= 23
FPS_MUVIM ?= 30

# -------------------------
# Interim pose dirs
# -------------------------
POSE_LE2I_RAW  := $(INTERIM)/le2i/pose_npz_raw
POSE_LE2I      := $(INTERIM)/le2i/pose_npz

POSE_URFD_RAW  := $(INTERIM)/urfd/pose_npz_raw
POSE_URFD      := $(INTERIM)/urfd/pose_npz

POSE_CAUC_RAW  := $(INTERIM)/caucafall/pose_npz_raw
POSE_CAUC      := $(INTERIM)/caucafall/pose_npz

POSE_MUVIM_RAW := $(INTERIM)/muvim/pose_npz_raw
POSE_MUVIM     := $(INTERIM)/muvim/pose_npz

# -------------------------
# Labels / spans
# -------------------------
LABELS_LE2I := $(LABELS_DIR)/le2i.json
SPANS_LE2I  := $(LABELS_DIR)/le2i_spans.json

LABELS_URFD := $(LABELS_DIR)/urfd.json
SPANS_URFD  := $(LABELS_DIR)/urfd_spans.json

LABELS_CAUC := $(LABELS_DIR)/caucafall.json
SPANS_CAUC  := $(LABELS_DIR)/caucafall_spans.json

LABELS_MUVIM := $(LABELS_DIR)/muvim.json
SPANS_MUVIM  := $(LABELS_DIR)/muvim_spans.json

# MUVIM CSV (you showed this)
MUVIM_ZED_CSV ?= $(RAW_MUVIM)/ZED_RGB/ZED_RGB.csv

# -------------------------
# Splits files
# -------------------------
SPLIT_LE2I_TRAIN := $(SPLITS_DIR)/le2i_train.txt
SPLIT_LE2I_VAL   := $(SPLITS_DIR)/le2i_val.txt
SPLIT_LE2I_TEST  := $(SPLITS_DIR)/le2i_test.txt

SPLIT_URFD_TRAIN := $(SPLITS_DIR)/urfd_train.txt
SPLIT_URFD_VAL   := $(SPLITS_DIR)/urfd_val.txt
SPLIT_URFD_TEST  := $(SPLITS_DIR)/urfd_test.txt

SPLIT_CAUC_TRAIN := $(SPLITS_DIR)/caucafall_train.txt
SPLIT_CAUC_VAL   := $(SPLITS_DIR)/caucafall_val.txt
SPLIT_CAUC_TEST  := $(SPLITS_DIR)/caucafall_test.txt

SPLIT_MUVIM_TRAIN := $(SPLITS_DIR)/muvim_train.txt
SPLIT_MUVIM_VAL   := $(SPLITS_DIR)/muvim_val.txt
SPLIT_MUVIM_TEST  := $(SPLITS_DIR)/muvim_test.txt

# -------------------------
# Windows output dirs
# -------------------------
WIN_LE2I  := $(PROCESSED)/le2i/windows_W$(WIN_W)_S$(WIN_S)
WIN_URFD  := $(PROCESSED)/urfd/windows_W$(WIN_W)_S$(WIN_S)
WIN_CAUC  := $(PROCESSED)/caucafall/windows_W$(WIN_W)_S$(WIN_S)
WIN_MUVIM := $(PROCESSED)/muvim/windows_W$(WIN_W)_S$(WIN_S)

# Extra window knobs (override at call time if needed)
WIN_EXTRA ?= --strategy balanced --min_overlap_frames 1 --pos_per_span 20 --neg_ratio 2.0 --max_neg_per_video 250 --max_windows_per_video_no_spans 120 --min_valid_frac 0.0

# -------------------------
# Model outputs
# -------------------------
OUT_TCN_LE2I  := $(OUT_DIR)/le2i_tcn_W$(WIN_W)S$(WIN_S)
OUT_TCN_URFD  := $(OUT_DIR)/urfd_tcn_W$(WIN_W)S$(WIN_S)
OUT_TCN_CAUC  := $(OUT_DIR)/caucafall_tcn_W$(WIN_W)S$(WIN_S)
OUT_TCN_MUVIM := $(OUT_DIR)/muvim_tcn_W$(WIN_W)S$(WIN_S)

OUT_GCN_LE2I  := $(OUT_DIR)/le2i_gcn_W$(WIN_W)S$(WIN_S)
OUT_GCN_URFD  := $(OUT_DIR)/urfd_gcn_W$(WIN_W)S$(WIN_S)
OUT_GCN_CAUC  := $(OUT_DIR)/caucafall_gcn_W$(WIN_W)S$(WIN_S)
OUT_GCN_MUVIM := $(OUT_DIR)/muvim_gcn_W$(WIN_W)S$(WIN_S)
# Checkpoint paths
CKPT_TCN_LE2I  := $(OUT_TCN_LE2I)/best.pt
CKPT_TCN_URFD  := $(OUT_TCN_URFD)/best.pt
CKPT_TCN_CAUC  := $(OUT_TCN_CAUC)/best.pt
CKPT_TCN_MUVIM := $(OUT_TCN_MUVIM)/best.pt
CKPT_GCN_LE2I  := $(OUT_GCN_LE2I)/best.pt
CKPT_GCN_URFD  := $(OUT_GCN_URFD)/best.pt
CKPT_GCN_CAUC  := $(OUT_GCN_CAUC)/best.pt
CKPT_GCN_MUVIM := $(OUT_GCN_MUVIM)/best.pt

# Ops yaml outputs
OPS_TCN_LE2I  := $(OPS_DIR)/tcn_le2i.yaml
OPS_TCN_URFD  := $(OPS_DIR)/tcn_urfd.yaml
OPS_TCN_CAUC  := $(OPS_DIR)/tcn_caucafall.yaml
OPS_TCN_MUVIM := $(OPS_DIR)/tcn_muvim.yaml

OPS_GCN_LE2I  := $(OPS_DIR)/gcn_le2i.yaml
OPS_GCN_URFD  := $(OPS_DIR)/gcn_urfd.yaml
OPS_GCN_CAUC  := $(OPS_DIR)/gcn_caucafall.yaml
OPS_GCN_MUVIM := $(OPS_DIR)/gcn_muvim.yaml

CENTER ?= $(FEAT_CENTER)
USE_MOTION ?= 1
USE_CONF_CHANNEL ?= 1
MOTION_SCALE_BY_FPS ?= 1
CONF_GATE ?= 0.20
USE_PRECOMPUTED_MASK ?= 1

PREPROC_ARGS = \
  --center "$(CENTER)" \
  --use_motion "$(USE_MOTION)" \
  --use_conf_channel "$(USE_CONF_CHANNEL)" \
  --motion_scale_by_fps "$(MOTION_SCALE_BY_FPS)" \
  --conf_gate "$(CONF_GATE)" \
  --use_precomputed_mask "$(USE_PRECOMPUTED_MASK)"


# -------------------------
# Help
# -------------------------
.PHONY: help
help:
	@echo ""
	@echo "Fall Detection v2 — Make targets"
	@echo ""
	@echo "Preprocess pipelines:"
	@echo "  make pipeline-le2i | pipeline-urfd | pipeline-caucafall | pipeline-muvim"
	@echo "  make pipeline-le2i-noextract   (skip extract, reuse pose_npz_raw)"
	@echo ""
	@echo "Sanity checks:"
	@echo "  make check-windows-<ds>        (ds: le2i urfd caucafall muvim)"
	@echo "  make check-windows            (all datasets)"
	@echo ""
	@echo "Train:"
	@echo "  make train-tcn-<ds> | train-gcn-<ds>          (ds: le2i urfd caucafall muvim)"
	@echo "  make pipeline-all              (TCN: train+fit-ops+eval+plot for all datasets)"
	@echo "  make pipeline-all-gcn          (GCN: train+fit-ops+eval+plot for all datasets)"
	@echo ""
	@echo "Thresholds + evaluation:"
	@echo "  make fit-ops-<ds> | fit-ops-gcn-<ds>"
	@echo "  make eval-<ds>    | eval-<ds>-gcn"
	@echo "  make plot-<ds>    | plot-<ds>-gcn"
	@echo "  make eval-all | plot-all | eval-all-gcn | plot-all-gcn"
	@echo ""
	@echo "Common overrides:"
	@echo "  make WIN_W=48 WIN_S=12 pipeline-le2i"
	@echo "  make EPOCHS=80 BATCH=128 LR=1e-3 train-gcn-muvim"
	@echo ""
# -------------------------
# Extract pose
# -------------------------
.PHONY: extract-le2i extract-urfd extract-caucafall extract-muvim

extract-le2i:
	@mkdir -p "$(POSE_LE2I_RAW)"
	$(RUN) pose/extract_2d.py \
	  --videos_glob "$(RAW_LE2I)/**/Videos/*.avi" "$(RAW_LE2I)/**/*.avi" "$(RAW_LE2I)/**/Videos/*.mp4" "$(RAW_LE2I)/**/*.mp4" \
	  --out_dir "$(POSE_LE2I_RAW)" \
	  --fps_default "$(FPS_LE2I)" \
	  --skip_existing || true

extract-urfd:
	@mkdir -p "$(POSE_URFD_RAW)"
	$(RUN) pose/extract_2d_from_images.py \
	  --images_glob "$(RAW_URFD)/*/*.jpg" "$(RAW_URFD)/*/*.png" \
	  --sequence_id_depth 1 \
	  --out_dir "$(POSE_URFD_RAW)" \
	  --dataset urfd \
	  --fps "$(FPS_URFD)" \
	  --skip_existing

extract-caucafall:
	@mkdir -p "$(POSE_CAUC_RAW)"
	$(RUN) pose/extract_2d_from_images.py \
	  --images_glob "$(RAW_CAUC)/**/*.jpg" "$(RAW_CAUC)/**/*.png" \
	  --sequence_id_depth 2 \
	  --out_dir "$(POSE_CAUC_RAW)" \
	  --dataset caucafall \
	  --fps "$(FPS_CAUC)" \
	  --skip_existing

extract-muvim:
	@echo "[info] MUVIM extraction depends on your raw layout."
	@echo "       If you already have pose_npz_raw or pose_npz, you can run preprocess-muvim."

# -------------------------
# Preprocess (clean/gate/smooth/normalize)
# -------------------------
.PHONY: preprocess-le2i preprocess-urfd preprocess-caucafall preprocess-muvim
.PHONY: preprocess-le2i-only preprocess-urfd-only preprocess-caucafall-only preprocess-muvim-only

preprocess-le2i: extract-le2i preprocess-le2i-only
preprocess-le2i-only:
	@mkdir -p "$(POSE_LE2I)"
	$(RUN) pose/preprocess_pose_npz.py \
	  --in_dir  "$(POSE_LE2I_RAW)" \
	  --out_dir "$(POSE_LE2I)" \
	  --recursive --skip_existing \
	  --conf_thr "$(CONF_THR)" --smooth_k "$(SMOOTH_K)" --max_gap "$(MAX_GAP)" \
	  --normalize "$(NORM_MODE)" --pelvis_fill "$(PELVIS_FILL)"

preprocess-urfd: extract-urfd preprocess-urfd-only
preprocess-urfd-only:
	@mkdir -p "$(POSE_URFD)"
	$(RUN) pose/preprocess_pose_npz.py \
	  --in_dir  "$(POSE_URFD_RAW)" \
	  --out_dir "$(POSE_URFD)" \
	  --recursive --skip_existing \
	  --conf_thr "$(CONF_THR)" --smooth_k "$(SMOOTH_K)" --max_gap "$(MAX_GAP)" \
	  --normalize "$(NORM_MODE)" --pelvis_fill "$(PELVIS_FILL)"

preprocess-caucafall: extract-caucafall preprocess-caucafall-only
preprocess-caucafall-only:
	@mkdir -p "$(POSE_CAUC)"
	$(RUN) pose/preprocess_pose_npz.py \
	  --in_dir  "$(POSE_CAUC_RAW)" \
	  --out_dir "$(POSE_CAUC)" \
	  --recursive --skip_existing \
	  --conf_thr "$(CONF_THR)" --smooth_k "$(SMOOTH_K)" --max_gap "$(MAX_GAP)" \
	  --normalize "$(NORM_MODE)" --pelvis_fill "$(PELVIS_FILL)"

preprocess-muvim: extract-muvim preprocess-muvim-only
preprocess-muvim-only:
	@mkdir -p "$(POSE_MUVIM)"
	@if [ -d "$(POSE_MUVIM_RAW)" ] && [ "$$(ls -A "$(POSE_MUVIM_RAW)" 2>/dev/null | wc -l)" -gt 0 ]; then \
	  $(RUN) pose/preprocess_pose_npz.py \
	    --in_dir  "$(POSE_MUVIM_RAW)" \
	    --out_dir "$(POSE_MUVIM)" \
	    --recursive --skip_existing \
	    --conf_thr "$(CONF_THR)" --smooth_k "$(SMOOTH_K)" --max_gap "$(MAX_GAP)" \
	    --normalize "$(NORM_MODE)" --pelvis_fill "$(PELVIS_FILL)"; \
	else \
	  echo "[info] no pose_npz_raw for muvim; if you already have pose_npz, skip preprocess."; \
	fi

# -------------------------
# Labels (+ spans where supported)
# -------------------------
# CAUCAFall annotation support (optional):
# If you want to try frame-level class -> spans, set:
#   URFD_ANN_GLOB ?=
URFD_FALL_CLASS_ID ?= 1
URFD_MIN_RUN ?= 3
URFD_GAP_FILL ?= 1

#   CAUCA_FALL_CLASS_ID=0 or 1  (depends on dataset class mapping)
CAUCA_ANN_GLOB ?= data/raw/caucafall/**/*.txt
CAUCA_FALL_CLASS_ID ?= 0

# Per-frame action txt parsing (URFD/CAUCA) safety switch
USE_PER_FRAME_ACTION_TXT ?= 0

CAUCA_MIN_RUN ?= 3
CAUCA_GAP_FILL ?= 2

.PHONY: labels-le2i labels-urfd labels-caucafall labels-muvim

labels-le2i: preprocess-le2i-only
	@mkdir -p "$(LABELS_DIR)"
	$(RUN) labels/make_le2i_labels.py \
	  --npz_dir "$(POSE_LE2I)" \
	  --raw_root "$(RAW_LE2I)" \
	  --out_labels "$(LABELS_LE2I)" \
	  --out_spans  "$(SPANS_LE2I)"

labels-urfd: preprocess-urfd-only
	@mkdir -p "$(LABELS_DIR)"
	@ANN=""; \
	if [ -n "$(URFD_ANN_GLOB)" ]; then \
	  ANN="--ann_glob \"$(URFD_ANN_GLOB)\" --use_per_frame_action_txt $(USE_PER_FRAME_ACTION_TXT) --fall_class_id $(URFD_FALL_CLASS_ID) --min_run $(URFD_MIN_RUN) --gap_fill $(URFD_GAP_FILL)"; \
	fi; \
	eval "$(RUN) labels/make_urfd_labels.py --npz_dir \"$(POSE_URFD)\" --out_labels \"$(LABELS_URFD)\" --out_spans \"$(SPANS_URFD)\" $$ANN"

labels-caucafall: preprocess-caucafall-only
	@mkdir -p "$(LABELS_DIR)"
	@ANN=""; \
	if [ -n "$(CAUCA_ANN_GLOB)" ]; then \
	  ANN="--ann_glob \"$(CAUCA_ANN_GLOB)\" --use_per_frame_action_txt $(USE_PER_FRAME_ACTION_TXT) --fall_class_id $(CAUCA_FALL_CLASS_ID) --min_run $(CAUCA_MIN_RUN) --gap_fill $(CAUCA_GAP_FILL)"; \
	fi; \
	eval "$(RUN) labels/make_caucafall_labels.py --npz_dir \"$(POSE_CAUC)\" --out_labels \"$(LABELS_CAUC)\" --out_spans \"$(SPANS_CAUC)\" $$ANN"

labels-muvim: preprocess-muvim-only
	@mkdir -p "$(LABELS_DIR)"
	$(RUN) labels/make_muvim_labels.py \
	  --npz_dir "$(POSE_MUVIM)" \
	  --zed_csv "$(MUVIM_ZED_CSV)" \
	  --out_labels "$(LABELS_MUVIM)" \
	  --out_spans  "$(SPANS_MUVIM)"

# -------------------------
# Splits
# -------------------------
.PHONY: splits-le2i splits-urfd splits-caucafall splits-muvim

splits-le2i: labels-le2i
	@mkdir -p "$(SPLITS_DIR)"
	$(RUN) split/make_splits.py \
	  --labels_json "$(LABELS_LE2I)" \
	  --out_dir "$(SPLITS_DIR)" \
	  --prefix le2i \
	  --train "$(TRAIN_FRAC)" --val "$(VAL_FRAC)" --test "$(TEST_FRAC)" \
	  --seed "$(SPLIT_SEED)"

splits-urfd: labels-urfd
	@mkdir -p "$(SPLITS_DIR)"
	$(RUN) split/make_splits.py \
	  --labels_json "$(LABELS_URFD)" \
	  --out_dir "$(SPLITS_DIR)" \
	  --prefix urfd \
	  --train "$(TRAIN_FRAC)" --val "$(VAL_FRAC)" --test "$(TEST_FRAC)" \
	  --seed "$(SPLIT_SEED)"

splits-caucafall: labels-caucafall
	@mkdir -p "$(SPLITS_DIR)"
	$(RUN) split/make_splits.py \
	  --labels_json "$(LABELS_CAUC)" \
	  --out_dir "$(SPLITS_DIR)" \
	  --prefix caucafall \
	  --train "$(TRAIN_FRAC)" --val "$(VAL_FRAC)" --test "$(TEST_FRAC)" \
	  --seed "$(SPLIT_SEED)"

splits-muvim: labels-muvim
	@mkdir -p "$(SPLITS_DIR)"
	$(RUN) split/make_splits.py \
	  --labels_json "$(LABELS_MUVIM)" \
	  --out_dir "$(SPLITS_DIR)" \
	  --prefix muvim \
	  --train "$(TRAIN_FRAC)" --val "$(VAL_FRAC)" --test "$(TEST_FRAC)" \
	  --seed "$(SPLIT_SEED)"

# -------------------------
# Windows
# -------------------------
.PHONY: windows-le2i windows-urfd windows-caucafall windows-muvim

windows-le2i: splits-le2i
	@mkdir -p "$(WIN_LE2I)"
	$(RUN) windows/make_windows.py \
	  --npz_dir "$(POSE_LE2I)" \
	  --labels_json "$(LABELS_LE2I)" \
	  --spans_json  "$(SPANS_LE2I)" \
	  --out_dir "$(WIN_LE2I)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_LE2I)" \
	  --train_list "$(SPLIT_LE2I_TRAIN)" --val_list "$(SPLIT_LE2I_VAL)" --test_list "$(SPLIT_LE2I_TEST)" \
	  $(WIN_EXTRA)

windows-urfd: splits-urfd
	@mkdir -p "$(WIN_URFD)"
	$(RUN) windows/make_windows.py \
	  --npz_dir "$(POSE_URFD)" \
	  --labels_json "$(LABELS_URFD)" \
	  --out_dir "$(WIN_URFD)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_URFD)" \
	  --train_list "$(SPLIT_URFD_TRAIN)" --val_list "$(SPLIT_URFD_VAL)" --test_list "$(SPLIT_URFD_TEST)" \
	  $(WIN_EXTRA)

windows-caucafall: splits-caucafall
	@mkdir -p "$(WIN_CAUC)"
	@SP=""; if [ -f "$(SPANS_CAUC)" ] && [ "$$(wc -c < "$(SPANS_CAUC)")" -gt 5 ]; then SP="--spans_json \"$(SPANS_CAUC)\""; fi; \
	eval "$(RUN) windows/make_windows.py --npz_dir \"$(POSE_CAUC)\" --labels_json \"$(LABELS_CAUC)\" $$SP --out_dir \"$(WIN_CAUC)\" --W \"$(WIN_W)\" --stride \"$(WIN_S)\" --fps_default \"$(FPS_CAUC)\" --train_list \"$(SPLIT_CAUC_TRAIN)\" --val_list \"$(SPLIT_CAUC_VAL)\" --test_list \"$(SPLIT_CAUC_TEST)\" $(WIN_EXTRA)"

windows-muvim: splits-muvim
	@mkdir -p "$(WIN_MUVIM)"
	$(RUN) windows/make_windows.py \
	  --npz_dir "$(POSE_MUVIM)" \
	  --labels_json "$(LABELS_MUVIM)" \
	  --spans_json  "$(SPANS_MUVIM)" \
	  --out_dir "$(WIN_MUVIM)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_MUVIM)" \
	  --train_list "$(SPLIT_MUVIM_TRAIN)" --val_list "$(SPLIT_MUVIM_VAL)" --test_list "$(SPLIT_MUVIM_TEST)" \
	  $(WIN_EXTRA)



# -------------------------
# Check windows (sanity checks)
# -------------------------
.PHONY: check-windows check-windows-le2i check-windows-urfd check-windows-caucafall check-windows-muvim

check-windows: check-windows-le2i check-windows-urfd check-windows-caucafall check-windows-muvim

check-windows-le2i:
	$(RUN) windows/check_windows.py --root "$(WIN_LE2I)"

check-windows-urfd:
	$(RUN) windows/check_windows.py --root "$(WIN_URFD)"

check-windows-caucafall:
	$(RUN) windows/check_windows.py --root "$(WIN_CAUC)"

check-windows-muvim:
	$(RUN) windows/check_windows.py --root "$(WIN_MUVIM)"

# -------------------------
# Pipeline (with extract)
# -------------------------
.PHONY: pipeline-le2i pipeline-urfd pipeline-caucafall pipeline-muvim
pipeline-le2i: preprocess-le2i windows-le2i
pipeline-urfd: preprocess-urfd windows-urfd
pipeline-caucafall: preprocess-caucafall windows-caucafall
pipeline-muvim: preprocess-muvim windows-muvim

# -------------------------
# Pipeline (NO extract)
# -------------------------
.PHONY: pipeline-le2i-noextract pipeline-urfd-noextract pipeline-caucafall-noextract pipeline-muvim-noextract

pipeline-le2i-noextract: preprocess-le2i-only windows-le2i

pipeline-urfd-noextract: preprocess-urfd-only windows-urfd

pipeline-caucafall-noextract: preprocess-caucafall-only windows-caucafall

pipeline-muvim-noextract: preprocess-muvim-only windows-muvim

# -------------------------
# Train
# -------------------------
.PHONY: train-tcn-le2i train-tcn-urfd train-tcn-caucafall train-tcn-muvim
.PHONY: train-gcn-le2i train-gcn-urfd train-gcn-caucafall train-gcn-muvim

train-tcn-le2i:
	# [check] train-tcn-le2i
	if [ ! -d "$(WIN_LE2I)/train" ] || [ ! -d "$(WIN_LE2I)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_LE2I)"; \
	  echo "      Run: make windows-le2i   (or pipeline-le2i-noextract)"; \
	  exit 2; \
	fi
	$(RUN) models/train_tcn.py --train_dir "$(WIN_LE2I)/train" --val_dir "$(WIN_LE2I)/val" \
	  --epochs "$(EPOCHS)" --batch "$(BATCH)" --lr "$(LR_TCN_LE2I)" --seed "$(SPLIT_SEED)" \
	  --fps_default "$(FPS_LE2I)" \
	  $(TCN_BASE_FLAGS) \
	  --monitor "$(TCN_MONITOR_LE2I)" \
	  --dropout "$(TCN_DROPOUT)" \
	  --mask_joint_p "$(TCN_MASK_JOINT_P)" --mask_frame_p "$(TCN_MASK_FRAME_P)" \
	  --pos_weight "$(TCN_POS_WEIGHT)" $(TCN_BALANCED_SAMPLER_FLAG) \
	  --save_dir "$(OUT_TCN_LE2I)$(OUT_TAG)"

train-tcn-urfd:
	# [check] train-tcn-urfd
	if [ ! -d "$(WIN_URFD)/train" ] || [ ! -d "$(WIN_URFD)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_URFD)"; \
	  echo "      Run: make windows-urfd   (or pipeline-urfd-noextract)"; \
	  exit 2; \
	fi
	$(RUN) models/train_tcn.py --train_dir "$(WIN_URFD)/train" --val_dir "$(WIN_URFD)/val" \
	  --epochs "$(EPOCHS)" --batch "$(BATCH)" --lr "$(LR_TCN_URFD)" --seed "$(SPLIT_SEED)" \
	  --fps_default "$(FPS_URFD)" \
	  $(TCN_BASE_FLAGS) \
	  --monitor "$(TCN_MONITOR_URFD)" \
	  --dropout "$(TCN_DROPOUT)" \
	  --mask_joint_p "$(TCN_MASK_JOINT_P)" --mask_frame_p "$(TCN_MASK_FRAME_P)" \
	  --pos_weight "$(TCN_POS_WEIGHT)" $(TCN_BALANCED_SAMPLER_FLAG) \
	  --save_dir "$(OUT_TCN_URFD)$(OUT_TAG)"

train-tcn-caucafall:
	# [check] train-tcn-caucafall
	if [ ! -d "$(WIN_CAUC)/train" ] || [ ! -d "$(WIN_CAUC)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_CAUC)"; \
	  echo "      Run: make windows-caucafall   (or pipeline-caucafall-noextract)"; \
	  exit 2; \
	fi
	$(RUN) models/train_tcn.py --train_dir "$(WIN_CAUC)/train" --val_dir "$(WIN_CAUC)/val" \
	  --epochs "$(EPOCHS)" --batch "$(BATCH)" --lr "$(LR_TCN_CAUC)" --seed "$(SPLIT_SEED)" \
	  --fps_default "$(FPS_CAUC)" \
	  $(TCN_BASE_FLAGS) \
	  --monitor "$(TCN_MONITOR_CAUC)" \
	  --dropout "$(TCN_DROPOUT)" \
	  --mask_joint_p "$(TCN_MASK_JOINT_P)" --mask_frame_p "$(TCN_MASK_FRAME_P)" \
	  --pos_weight "$(TCN_POS_WEIGHT)" $(TCN_BALANCED_SAMPLER_FLAG) \
	  --save_dir "$(OUT_TCN_CAUC)$(OUT_TAG)"

train-tcn-muvim:
	# [check] train-tcn-muvim
	if [ ! -d "$(WIN_MUVIM)/train" ] || [ ! -d "$(WIN_MUVIM)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_MUVIM)"; \
	  echo "      Run: make windows-muvim   (or pipeline-muvim-noextract)"; \
	  exit 2; \
	fi
	$(RUN) models/train_tcn.py --train_dir "$(WIN_MUVIM)/train" --val_dir "$(WIN_MUVIM)/val" \
	  --epochs "$(EPOCHS)" --batch "$(BATCH)" --lr "$(LR_TCN_MUVIM)" --seed "$(SPLIT_SEED)" \
	  --fps_default "$(FPS_MUVIM)" \
	  $(TCN_BASE_FLAGS) \
	  --monitor "$(TCN_MONITOR_MUVIM)" \
	  --dropout "$(TCN_DROPOUT_MUVIM)" \
	  --mask_joint_p "$(TCN_MASK_JOINT_P_MUVIM)" --mask_frame_p "$(TCN_MASK_FRAME_P_MUVIM)" \
	  --pos_weight "$(TCN_POS_WEIGHT_MUVIM)" $(TCN_BALANCED_SAMPLER_FLAG_MUVIM) \
	  --save_dir "$(OUT_TCN_MUVIM)$(OUT_TAG)"

train-gcn-le2i:
	# [check] train-gcn-le2i
	if [ ! -d "$(WIN_LE2I)/train" ] || [ ! -d "$(WIN_LE2I)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_LE2I)"; \
	  echo "      Run: make windows-le2i   (or pipeline-le2i-noextract)"; \
	  exit 2; \
	fi
	$(RUN) models/train_gcn.py --train_dir "$(WIN_LE2I)/train" --val_dir "$(WIN_LE2I)/val" \
	  --epochs "$(EPOCHS_GCN)" --batch "$(BATCH_GCN)" --lr "$(LR_GCN_LE2I)" --seed "$(SPLIT_SEED)" \
	  --fps_default "$(FPS_LE2I)" \
	  $(GCN_MODEL_FLAGS) \
	  --monitor "$(GCN_MONITOR_LE2I)" \
	  --dropout "$(GCN_DROPOUT)" \
	  --pos_weight "$(GCN_POS_WEIGHT)" $(GCN_BALANCED_SAMPLER_FLAG) \
	  --save_dir "$(OUT_GCN_LE2I)$(OUT_TAG)"

train-gcn-urfd:
	# [check] train-gcn-urfd
	if [ ! -d "$(WIN_URFD)/train" ] || [ ! -d "$(WIN_URFD)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_URFD)"; \
	  echo "      Run: make windows-urfd   (or pipeline-urfd-noextract)"; \
	  exit 2; \
	fi
	$(RUN) models/train_gcn.py --train_dir "$(WIN_URFD)/train" --val_dir "$(WIN_URFD)/val" \
	  --epochs "$(EPOCHS_GCN)" --batch "$(BATCH_GCN)" --lr "$(LR_GCN_URFD)" --seed "$(SPLIT_SEED)" \
	  --fps_default "$(FPS_URFD)" \
	  $(GCN_MODEL_FLAGS) \
	  --monitor "$(GCN_MONITOR_URFD)" \
	  --dropout "$(GCN_DROPOUT)" \
	  --pos_weight "$(GCN_POS_WEIGHT)" $(GCN_BALANCED_SAMPLER_FLAG) \
	  --save_dir "$(OUT_GCN_URFD)$(OUT_TAG)"

train-gcn-caucafall:
	# [check] train-gcn-caucafall
	if [ ! -d "$(WIN_CAUC)/train" ] || [ ! -d "$(WIN_CAUC)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_CAUC)"; \
	  echo "      Run: make windows-caucafall   (or pipeline-caucafall-noextract)"; \
	  exit 2; \
	fi
	$(RUN) models/train_gcn.py --train_dir "$(WIN_CAUC)/train" --val_dir "$(WIN_CAUC)/val" \
	  --epochs "$(EPOCHS_GCN)" --batch "$(BATCH_GCN)" --lr "$(LR_GCN_CAUC)" --seed "$(SPLIT_SEED)" \
	  --fps_default "$(FPS_CAUC)" \
	  $(GCN_MODEL_FLAGS) \
	  --monitor "$(GCN_MONITOR_CAUC)" \
	  --dropout "$(GCN_DROPOUT)" \
	  --pos_weight "$(GCN_POS_WEIGHT)" $(GCN_BALANCED_SAMPLER_FLAG) \
	  --save_dir "$(OUT_GCN_CAUC)$(OUT_TAG)"

train-gcn-muvim:
	# [check] train-gcn-muvim
	if [ ! -d "$(WIN_MUVIM)/train" ] || [ ! -d "$(WIN_MUVIM)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_MUVIM)"; \
	  echo "      Run: make windows-muvim   (or pipeline-muvim-noextract)"; \
	  exit 2; \
	fi
	$(RUN) models/train_gcn.py --train_dir "$(WIN_MUVIM)/train" --val_dir "$(WIN_MUVIM)/val" \
	  --epochs "$(EPOCHS_GCN)" --batch "$(BATCH_GCN)" --lr "$(LR_GCN_MUVIM)" --seed "$(SPLIT_SEED)" \
	  --fps_default "$(FPS_MUVIM)" \
	  $(GCN_MODEL_FLAGS) \
	  --monitor "$(GCN_MONITOR_MUVIM)" \
	  --dropout "$(GCN_DROPOUT_MUVIM)" \
	  --pos_weight "$(GCN_POS_WEIGHT_MUVIM)" $(GCN_BALANCED_SAMPLER_FLAG_MUVIM) \
	  --save_dir "$(OUT_GCN_MUVIM)$(OUT_TAG)"


pipeline-all-gcn: train-gcn-le2i train-gcn-urfd train-gcn-caucafall train-gcn-muvim eval-all-gcn plot-all-gcn

pipeline-all-gcn-noextract:
	$(MAKE) pipeline-le2i-noextract
	$(MAKE) pipeline-urfd-noextract
	$(MAKE) pipeline-caucafall-noextract
	$(MAKE) pipeline-muvim-noextract
# -------------------------
# Feature config (shared) — training only
# (Saved into checkpoint bundle; eval scripts load from ckpt.)
# -------------------------
CENTER ?= $(FEAT_CENTER)
FEAT_USE_MOTION ?= 1
FEAT_USE_CONF_CHANNEL ?= 1
FEAT_MOTION_SCALE_BY_FPS ?= 1
FEAT_CONF_GATE ?= 0.20
FEAT_USE_PRECOMPUTED_MASK ?= 1

FEAT_FLAGS = \
  --center "$(CENTER)" \
  --use_motion "$(FEAT_USE_MOTION)" \
  --use_conf_channel "$(FEAT_USE_CONF_CHANNEL)" \
  --motion_scale_by_fps "$(FEAT_MOTION_SCALE_BY_FPS)" \
  --conf_gate "$(FEAT_CONF_GATE)" \
  --use_precomputed_mask "$(FEAT_USE_PRECOMPUTED_MASK)"

# TCN flags
TCN_BALANCED_SAMPLER_FLAG = $(if $(filter 1,$(TCN_BALANCED_SAMPLER)),--balanced_sampler,)
TCN_BALANCED_SAMPLER_FLAG_MUVIM = $(if $(filter 1,$(TCN_BALANCED_SAMPLER_MUVIM)),--balanced_sampler,)
TCN_BASE_FLAGS = \
  $(FEAT_FLAGS) \
  --hidden "$(TCN_HIDDEN)" --num_blocks "$(TCN_NUM_BLOCKS)" --kernel "$(TCN_KERNEL)" \
  --grad_clip "$(TCN_GRAD_CLIP)" --patience "$(TCN_PATIENCE)"

# GCN flags
GCN_BALANCED_SAMPLER_FLAG = $(if $(filter 1,$(GCN_BALANCED_SAMPLER)),--balanced_sampler,)
GCN_TWO_STREAM_FLAG = $(if $(filter 1,$(GCN_TWO_STREAM)),--two_stream,)
GCN_MODEL_FLAGS = \
  $(FEAT_FLAGS) \
  --gcn_hidden "$(GCN_GCN_HIDDEN)" --tcn_hidden "$(GCN_TCN_HIDDEN)" \
  --use_se "$(GCN_USE_SE)" \
  $(GCN_TWO_STREAM_FLAG) --fuse "$(GCN_FUSE)" \
  --grad_clip "$(GCN_GRAD_CLIP)" --patience "$(GCN_PATIENCE)"


# ----------------------
# Fit OPs (TCN)
# ----------------------
# ----------------------
# Runtime feature flags (must match training)
# ----------------------
CENTER ?= $(FEAT_CENTER)
# If your TCN was trained without motion/conf, keep these at 0.
TCN_USE_MOTION ?= $(FEAT_USE_MOTION)
TCN_USE_CONF_CHANNEL ?= $(FEAT_USE_CONF_CHANNEL)
TCN_MOTION_SCALE_BY_FPS ?= $(FEAT_MOTION_SCALE_BY_FPS)
TCN_CONF_GATE ?= $(FEAT_CONF_GATE)
TCN_USE_PRECOMPUTED_MASK ?= $(FEAT_USE_PRECOMPUTED_MASK)
# Your current GCN training command uses these as 1.
GCN_USE_MOTION ?= $(FEAT_USE_MOTION)
GCN_USE_CONF_CHANNEL ?= $(FEAT_USE_CONF_CHANNEL)
GCN_MOTION_SCALE_BY_FPS ?= $(FEAT_MOTION_SCALE_BY_FPS)
GCN_CONF_GATE ?= $(FEAT_CONF_GATE)
GCN_USE_PRECOMPUTED_MASK ?= $(FEAT_USE_PRECOMPUTED_MASK)
FIT_FLAGS_TCN = --center "$(CENTER)" --use_motion "$(TCN_USE_MOTION)" --use_conf_channel "$(TCN_USE_CONF_CHANNEL)" --motion_scale_by_fps "$(TCN_MOTION_SCALE_BY_FPS)" --conf_gate "$(TCN_CONF_GATE)" --use_precomputed_mask "$(TCN_USE_PRECOMPUTED_MASK)"
FIT_FLAGS_GCN = --center "$(CENTER)" --use_motion "$(GCN_USE_MOTION)" --use_conf_channel "$(GCN_USE_CONF_CHANNEL)" --motion_scale_by_fps "$(GCN_MOTION_SCALE_BY_FPS)" --conf_gate "$(GCN_CONF_GATE)" --use_precomputed_mask "$(GCN_USE_PRECOMPUTED_MASK)"

# Training flags for GCN (kept consistent with fit/eval feature flags)
GCN_BALANCED_SAMPLER_FLAG = $(if $(filter 1,$(GCN_BALANCED_SAMPLER)),--balanced_sampler,)
TRAIN_FLAGS_GCN = $(FIT_FLAGS_GCN) \
  --patience "$(GCN_PATIENCE)" \
  --gcn_hidden "$(GCN_GCN_HIDDEN)" --tcn_hidden "$(GCN_TCN_HIDDEN)" \
  --dropout "$(GCN_DROPOUT)" --use_se "$(GCN_USE_SE)" \
  --two_stream "$(GCN_TWO_STREAM)" --fuse "$(GCN_FUSE)" \
  --mask_joint_p "$(MASK_JOINT_P)" --mask_frame_p "$(MASK_FRAME_P)" \
  --pos_weight "$(GCN_POS_WEIGHT)" $(GCN_BALANCED_SAMPLER_FLAG) \
  --weight_decay "$(GCN_WEIGHT_DECAY)" \
  --label_smoothing "$(GCN_LABEL_SMOOTHING)" --num_workers "$(GCN_NUM_WORKERS)"

EVAL_FLAGS_TCN = $(FIT_FLAGS_TCN)
EVAL_FLAGS_GCN = $(FIT_FLAGS_GCN)

# ----------------------
# Fit OPs (TCN)
# ----------------------
.PHONY: fit-ops-le2i fit-ops-urfd fit-ops-caucafall fit-ops-muvim
.PHONY: fit-ops-tcn-le2i fit-ops-tcn-urfd fit-ops-tcn-caucafall fit-ops-tcn-muvim
fit-ops-tcn-le2i:      fit-ops-le2i
fit-ops-tcn-urfd:      fit-ops-urfd
fit-ops-tcn-caucafall: fit-ops-caucafall
fit-ops-tcn-muvim:     fit-ops-muvim

fit-ops-le2i:
	# [check] fit-ops-le2i
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn --val_dir "$(WIN_LE2I)/val" --ckpt "$(CKPT_TCN_LE2I)" --out "$(OPS_TCN_LE2I)" \
	  --pose_npz_dir "$(POSE_LE2I)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

fit-ops-urfd:
	# [check] fit-ops-urfd
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn --val_dir "$(WIN_URFD)/val" --ckpt "$(CKPT_TCN_URFD)" --out "$(OPS_TCN_URFD)" \
	  --pose_npz_dir "$(POSE_URFD)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

fit-ops-caucafall:
	# [check] fit-ops-caucafall
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn --val_dir "$(WIN_CAUC)/val" --ckpt "$(CKPT_TCN_CAUC)" --out "$(OPS_TCN_CAUC)" \
	  --pose_npz_dir "$(POSE_CAUC)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

fit-ops-muvim:
	# [check] fit-ops-muvim
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn --val_dir "$(WIN_MUVIM)/val" --ckpt "$(CKPT_TCN_MUVIM)" --out "$(OPS_TCN_MUVIM)" \
	  --pose_npz_dir "$(POSE_MUVIM)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

.PHONY: fit-ops-gcn-le2i fit-ops-gcn-urfd fit-ops-gcn-caucafall fit-ops-gcn-muvim

fit-ops-gcn-le2i:
	# [check] fit-ops-gcn-le2i
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_LE2I)/val" --ckpt "$(CKPT_GCN_LE2I)" --out "$(OPS_GCN_LE2I)" \
	  --pose_npz_dir "$(POSE_LE2I)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

fit-ops-gcn-urfd:
	# [check] fit-ops-gcn-urfd
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_URFD)/val" --ckpt "$(CKPT_GCN_URFD)" --out "$(OPS_GCN_URFD)" \
	  --pose_npz_dir "$(POSE_URFD)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

fit-ops-gcn-caucafall:
	# [check] fit-ops-gcn-caucafall
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_CAUC)/val" --ckpt "$(CKPT_GCN_CAUC)" --out "$(OPS_GCN_CAUC)" \
	  --pose_npz_dir "$(POSE_CAUC)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

fit-ops-gcn-muvim:
	# [check] fit-ops-gcn-muvim
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_MUVIM)/val" --ckpt "$(CKPT_GCN_MUVIM)" --out "$(OPS_GCN_MUVIM)" \
	  --pose_npz_dir "$(POSE_MUVIM)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

.PHONY: eval-le2i eval-urfd eval-caucafall eval-muvim eval-le2i-on-urfd

eval-le2i: fit-ops-le2i
eval-le2i-tcn: eval-le2i

	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_LE2I)/test" --ckpt "$(CKPT_TCN_LE2I)" \
	  --ops_yaml "$(OPS_TCN_LE2I)" --out_json "$(REPORTS_DIR)/le2i_tcn.json" \
	  --pose_npz_dir "$(POSE_LE2I)" --stride_frames_hint "$(WIN_S)"

eval-urfd: fit-ops-urfd
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_URFD)/test" --ckpt "$(CKPT_TCN_URFD)" \
	  --ops_yaml "$(OPS_TCN_URFD)" --out_json "$(REPORTS_DIR)/urfd_tcn.json" \
	  --pose_npz_dir "$(POSE_URFD)" --stride_frames_hint "$(WIN_S)"

eval-caucafall: fit-ops-caucafall
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_CAUC)/test" --ckpt "$(CKPT_TCN_CAUC)" \
	  --ops_yaml "$(OPS_TCN_CAUC)" --out_json "$(REPORTS_DIR)/caucafall_tcn.json" \
	  --pose_npz_dir "$(POSE_CAUC)" --stride_frames_hint "$(WIN_S)"

eval-muvim: fit-ops-muvim
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_MUVIM)/test" --ckpt "$(CKPT_TCN_MUVIM)" \
	  --ops_yaml "$(OPS_TCN_MUVIM)" --out_json "$(REPORTS_DIR)/muvim_tcn.json" \
	  --pose_npz_dir "$(POSE_MUVIM)" --stride_frames_hint "$(WIN_S)"

eval-le2i-on-urfd: fit-ops-le2i
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_URFD)/test" --ckpt "$(CKPT_TCN_LE2I)" \
	  --ops_yaml "$(OPS_TCN_LE2I)" --out_json "$(REPORTS_DIR)/le2i_on_urfd_tcn.json" \
	  --pose_npz_dir "$(POSE_URFD)" --stride_frames_hint "$(WIN_S)"


# -------------------------
# Cross-dataset evaluation (GCN): MUVIM-trained checkpoint evaluated on other datasets
# Notes:
#   - We fit ops (thresholds / policy) on the TARGET dataset val split.
#   - Then we evaluate on the TARGET dataset test split using the SAME MUVIM checkpoint.
#   - This measures cross-dataset generalisation + how much re-calibration helps.
# -------------------------
.PHONY: fit-ops-muvim-gcn-on-le2i eval-muvim-gcn-on-le2i
.PHONY: fit-ops-muvim-gcn-on-urfd eval-muvim-gcn-on-urfd
.PHONY: fit-ops-muvim-gcn-on-caucafall eval-muvim-gcn-on-caucafall
.PHONY: cross-muvim-gcn

fit-ops-muvim-gcn-on-le2i:
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_LE2I)/val" --ckpt "$(CKPT_GCN_MUVIM)" --out "$(OPS_DIR)/gcn_muvim_on_le2i.yaml" \
	  --pose_npz_dir "$(POSE_LE2I)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

eval-muvim-gcn-on-le2i: fit-ops-muvim-gcn-on-le2i
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_LE2I)/test" --ckpt "$(CKPT_GCN_MUVIM)" \
	  --ops_yaml "$(OPS_DIR)/gcn_muvim_on_le2i.yaml" --out_json "$(REPORTS_DIR)/muvim_gcn_on_le2i.json" \
	  --pose_npz_dir "$(POSE_LE2I)" --stride_frames_hint "$(WIN_S)"

fit-ops-muvim-gcn-on-urfd:
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_URFD)/val" --ckpt "$(CKPT_GCN_MUVIM)" --out "$(OPS_DIR)/gcn_muvim_on_urfd.yaml" \
	  --pose_npz_dir "$(POSE_URFD)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

eval-muvim-gcn-on-urfd: fit-ops-muvim-gcn-on-urfd
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_URFD)/test" --ckpt "$(CKPT_GCN_MUVIM)" \
	  --ops_yaml "$(OPS_DIR)/gcn_muvim_on_urfd.yaml" --out_json "$(REPORTS_DIR)/muvim_gcn_on_urfd.json" \
	  --pose_npz_dir "$(POSE_URFD)" --stride_frames_hint "$(WIN_S)"

fit-ops-muvim-gcn-on-caucafall:
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_CAUC)/val" --ckpt "$(CKPT_GCN_MUVIM)" --out "$(OPS_DIR)/gcn_muvim_on_caucafall.yaml" \
	  --pose_npz_dir "$(POSE_CAUC)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

eval-muvim-gcn-on-caucafall: fit-ops-muvim-gcn-on-caucafall
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_CAUC)/test" --ckpt "$(CKPT_GCN_MUVIM)" \
	  --ops_yaml "$(OPS_DIR)/gcn_muvim_on_caucafall.yaml" --out_json "$(REPORTS_DIR)/muvim_gcn_on_caucafall.json" \
	  --pose_npz_dir "$(POSE_CAUC)" --stride_frames_hint "$(WIN_S)"

cross-muvim-gcn: eval-muvim-gcn-on-le2i eval-muvim-gcn-on-urfd eval-muvim-gcn-on-caucafall
	@echo "[ok] cross-muvim-gcn done (reports in $(REPORTS_DIR))"

.PHONY: eval-le2i-gcn eval-urfd-gcn eval-caucafall-gcn eval-muvim-gcn

eval-le2i-gcn: fit-ops-gcn-le2i
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_LE2I)/test" --ckpt "$(CKPT_GCN_LE2I)" \
	  --ops_yaml "$(OPS_GCN_LE2I)" --out_json "$(REPORTS_DIR)/le2i_gcn.json" \
	  --pose_npz_dir "$(POSE_LE2I)" --stride_frames_hint "$(WIN_S)"

eval-urfd-gcn: fit-ops-gcn-urfd
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_URFD)/test" --ckpt "$(CKPT_GCN_URFD)" \
	  --ops_yaml "$(OPS_GCN_URFD)" --out_json "$(REPORTS_DIR)/urfd_gcn.json" \
	  --pose_npz_dir "$(POSE_URFD)" --stride_frames_hint "$(WIN_S)"

eval-caucafall-gcn: fit-ops-gcn-caucafall
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_CAUC)/test" --ckpt "$(CKPT_GCN_CAUC)" \
	  --ops_yaml "$(OPS_GCN_CAUC)" --out_json "$(REPORTS_DIR)/caucafall_gcn.json" \
	  --pose_npz_dir "$(POSE_CAUC)" --stride_frames_hint "$(WIN_S)"

eval-muvim-gcn: fit-ops-gcn-muvim
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_MUVIM)/test" --ckpt "$(CKPT_GCN_MUVIM)" \
	  --ops_yaml "$(OPS_GCN_MUVIM)" --out_json "$(REPORTS_DIR)/muvim_gcn.json" \
	  --pose_npz_dir "$(POSE_MUVIM)" --stride_frames_hint "$(WIN_S)"

plot-le2i: eval-le2i
plot-le2i-tcn: plot-le2i

	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/le2i_tcn.json" --title "LE2i (TCN) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/le2i_tcn_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

plot-urfd: eval-urfd
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/urfd_tcn.json" --title "URFD (TCN) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/urfd_tcn_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

plot-caucafall: eval-caucafall
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/caucafall_tcn.json" --title "CAUCAFall (TCN) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/caucafall_tcn_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

plot-muvim: eval-muvim
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/muvim_tcn.json" --title "MUVIM (TCN) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/muvim_tcn_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

plot-le2i-on-urfd: eval-le2i-on-urfd
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/le2i_on_urfd_tcn.json" --title "LE2i model on URFD (TCN) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/le2i_on_urfd_tcn_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

.PHONY: plot-le2i-gcn plot-urfd-gcn plot-caucafall-gcn plot-muvim-gcn

plot-le2i-gcn: eval-le2i-gcn
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/le2i_gcn.json" --title "LE2i (GCN) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/le2i_gcn_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

plot-urfd-gcn: eval-urfd-gcn
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/urfd_gcn.json" --title "URFD (GCN) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/urfd_gcn_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

plot-caucafall-gcn: eval-caucafall-gcn
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/caucafall_gcn.json" --title "CAUCAFall (GCN) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/caucafall_gcn_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

plot-muvim-gcn: eval-muvim-gcn
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/muvim_gcn.json" --title "MUVIM (GCN) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/muvim_gcn_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

.PHONY: eval-all plot-all eval-all-gcn plot-all-gcn pipeline-all pipeline-all-gcn

eval-all: eval-le2i eval-urfd eval-caucafall eval-muvim
plot-all: plot-le2i plot-urfd plot-caucafall plot-muvim
eval-all-gcn: eval-le2i-gcn eval-urfd-gcn eval-caucafall-gcn eval-muvim-gcn
plot-all-gcn: plot-le2i-gcn plot-urfd-gcn plot-caucafall-gcn plot-muvim-gcn


# If you want a single pipeline (no re-extract unless needed by dependencies):
pipeline-all: train-tcn-le2i train-tcn-urfd train-tcn-caucafall train-tcn-muvim eval-all plot-all



# make train-gcn-muvim OUT_TAG=_lr3e4 LR_GCN=3e-4
# make train-gcn-muvim OUT_TAG=_drop025 GCN_DROPOUT=0.25
# make train-gcn-muvim OUT_TAG=_mask0803 MASK_JOINT_P=0.08 MASK_FRAME_P=0.03
# make train-gcn-muvim LR_GCN=5e-4 POS_WEIGHT_GCN=none BALANCED_SAMPLER_GCN=1 MASK_JOINT_P=0.10 MASK_FRAME_P=0.05



# -------------------------
# Hard-negative mining (HNM)
# -------------------------
# Mine the top-scoring negative windows (likely future false alarms) and fine-tune briefly.
#
# Default mining source: VAL negatives (y==0 or y<0) to avoid needing an unlabeled pipeline.
# If you have long ADL-only clips windowed into e.g. .../unlabeled, just set:
#   HNM_SRC_LE2I=$(WIN_LE2I)/unlabeled   (and similarly for others)

HNM_DIR ?= outputs/hardneg
HNM_TOPK ?= 2000        # start smaller; increase only if needed
HNM_MAX_PER_CLIP ?= 50
HNM_MIN_P ?= 0.20       # good default (lower to 0.10 if mining returns too few)
HNM_BATCH ?= 256        # keep
HNM_MULT ?= 2           # reduce (5 is usually too strong)
HNM_EPOCHS ?= 5         # reduce (8 can over-correct)
HNM_LR ?= 2e-4          # keep
HNM_PATIENCE ?= 3       # optional: lower because epochs is small
HNM_TAG ?= _hnm

HNM_SRC_LE2I ?= $(WIN_LE2I_UNLAB)
HNM_SRC_URFD ?= $(WIN_URFD)/train
HNM_SRC_CAUC ?= $(WIN_CAUC)/train
HNM_SRC_MUVIM ?= $(WIN_MUVIM)/train

# ---- LE2i ----
mine-hardneg-tcn-le2i: 
	# [check] train-tcn-le2i windows-le2i-unlabeled
	@mkdir -p "$(HNM_DIR)"
	$(RUN) eval/mine_hard_negatives.py \
	  --ckpt "$(OUT_TCN_LE2I)$(OUT_TAG)/best.pt" \
	  --windows_dir "$(HNM_SRC_LE2I)" \
	  --neg_only 1 \
	  --out_txt "$(HNM_DIR)/le2i_tcn.txt" \
	  --min_p "$(HNM_MIN_P)" --top_k "$(HNM_TOPK)" --max_per_clip "$(HNM_MAX_PER_CLIP)" --batch "$(HNM_BATCH)"

finetune-tcn-le2i: mine-hardneg-tcn-le2i
	$(RUN) models/train_tcn.py --train_dir "$(WIN_LE2I)/train" --val_dir "$(WIN_LE2I)/val" \
	  --epochs "$(HNM_EPOCHS)" --batch "$(BATCH)" --lr "$(HNM_LR)" --seed "$(SPLIT_SEED)" --patience "$(HNM_PATIENCE)" \
	  --resume "$(OUT_TCN_LE2I)$(OUT_TAG)/best.pt" \
	  --hard_neg_list "$(HNM_DIR)/le2i_tcn.txt" --hard_neg_mult "$(HNM_MULT)" \
	  --save_dir "$(OUT_TCN_LE2I)$(OUT_TAG)$(HNM_TAG)"

mine-hardneg-gcn-le2i: 
	# [check] train-gcn-le2i windows-le2i-unlabeled
	@mkdir -p "$(HNM_DIR)"
	$(RUN) eval/mine_hard_negatives.py \
	  --ckpt "$(OUT_GCN_LE2I)$(OUT_TAG)/best.pt" \
	  --windows_dir "$(HNM_SRC_LE2I)" \
	  --neg_only 1 \
	  --out_txt "$(HNM_DIR)/le2i_gcn.txt" \
	  --min_p "$(HNM_MIN_P)" --top_k "$(HNM_TOPK)" --max_per_clip "$(HNM_MAX_PER_CLIP)" --batch "$(HNM_BATCH)"

finetune-gcn-le2i: mine-hardneg-gcn-le2i
	$(RUN) models/train_gcn.py --train_dir "$(WIN_LE2I)/train" --val_dir "$(WIN_LE2I)/val" \
	  --epochs "$(HNM_EPOCHS)" --batch "$(BATCH)" --lr "$(HNM_LR)" --seed "$(SPLIT_SEED)" --patience "$(HNM_PATIENCE)" \
	  --resume "$(OUT_GCN_LE2I)$(OUT_TAG)/best.pt" \
	  --hard_neg_list "$(HNM_DIR)/le2i_gcn.txt" --hard_neg_mult "$(HNM_MULT)" \
	  --save_dir "$(OUT_GCN_LE2I)$(OUT_TAG)$(HNM_TAG)"

hnm-le2i: finetune-tcn-le2i finetune-gcn-le2i
	@echo "[ok] LE2i HNM done"

# ---- URFD ----
mine-hardneg-tcn-urfd: 
	# [check] train-tcn-urfd
	@mkdir -p "$(HNM_DIR)"
	$(RUN) eval/mine_hard_negatives.py \
	  --ckpt "$(OUT_TCN_URFD)$(OUT_TAG)/best.pt" \
	  --windows_dir "$(HNM_SRC_URFD)" \
	  --neg_only 1 \
	  --out_txt "$(HNM_DIR)/urfd_tcn.txt" \
	  --min_p "$(HNM_MIN_P)" --top_k "$(HNM_TOPK)" --max_per_clip "$(HNM_MAX_PER_CLIP)" --batch "$(HNM_BATCH)"

finetune-tcn-urfd: mine-hardneg-tcn-urfd
	$(RUN) models/train_tcn.py --train_dir "$(WIN_URFD)/train" --val_dir "$(WIN_URFD)/val" \
	  --epochs "$(HNM_EPOCHS)" --batch "$(BATCH)" --lr "$(HNM_LR)" --seed "$(SPLIT_SEED)" --patience "$(HNM_PATIENCE)" \
	  --resume "$(OUT_TCN_URFD)$(OUT_TAG)/best.pt" \
	  --hard_neg_list "$(HNM_DIR)/urfd_tcn.txt" --hard_neg_mult "$(HNM_MULT)" \
	  --save_dir "$(OUT_TCN_URFD)$(OUT_TAG)$(HNM_TAG)"

mine-hardneg-gcn-urfd: 
	# [check] train-gcn-urfd
	@mkdir -p "$(HNM_DIR)"
	$(RUN) eval/mine_hard_negatives.py \
	  --ckpt "$(OUT_GCN_URFD)$(OUT_TAG)/best.pt" \
	  --windows_dir "$(HNM_SRC_URFD)" \
	  --neg_only 1 \
	  --out_txt "$(HNM_DIR)/urfd_gcn.txt" \
	  --min_p "$(HNM_MIN_P)" --top_k "$(HNM_TOPK)" --max_per_clip "$(HNM_MAX_PER_CLIP)" --batch "$(HNM_BATCH)"

finetune-gcn-urfd: mine-hardneg-gcn-urfd
	$(RUN) models/train_gcn.py --train_dir "$(WIN_URFD)/train" --val_dir "$(WIN_URFD)/val" \
	  --epochs "$(HNM_EPOCHS)" --batch "$(BATCH)" --lr "$(HNM_LR)" --seed "$(SPLIT_SEED)" --patience "$(HNM_PATIENCE)" \
	  --resume "$(OUT_GCN_URFD)$(OUT_TAG)/best.pt" \
	  --hard_neg_list "$(HNM_DIR)/urfd_gcn.txt" --hard_neg_mult "$(HNM_MULT)" \
	  --save_dir "$(OUT_GCN_URFD)$(OUT_TAG)$(HNM_TAG)"

hnm-urfd: finetune-tcn-urfd finetune-gcn-urfd
	@echo "[ok] URFD HNM done"

# ---- CAUCAFall ----
mine-hardneg-tcn-caucafall: 
	# [check] train-tcn-caucafall
	@mkdir -p "$(HNM_DIR)"
	$(RUN) eval/mine_hard_negatives.py \
	  --ckpt "$(OUT_TCN_CAUC)$(OUT_TAG)/best.pt" \
	  --windows_dir "$(HNM_SRC_CAUC)" \
	  --neg_only 1 \
	  --out_txt "$(HNM_DIR)/caucafall_tcn.txt" \
	  --min_p "$(HNM_MIN_P)" --top_k "$(HNM_TOPK)" --max_per_clip "$(HNM_MAX_PER_CLIP)" --batch "$(HNM_BATCH)"

finetune-tcn-caucafall: mine-hardneg-tcn-caucafall
	$(RUN) models/train_tcn.py --train_dir "$(WIN_CAUC)/train" --val_dir "$(WIN_CAUC)/val" \
	  --epochs "$(HNM_EPOCHS)" --batch "$(BATCH)" --lr "$(HNM_LR)" --seed "$(SPLIT_SEED)" --patience "$(HNM_PATIENCE)" \
	  --resume "$(OUT_TCN_CAUC)$(OUT_TAG)/best.pt" \
	  --hard_neg_list "$(HNM_DIR)/caucafall_tcn.txt" --hard_neg_mult "$(HNM_MULT)" \
	  --save_dir "$(OUT_TCN_CAUC)$(OUT_TAG)$(HNM_TAG)"

mine-hardneg-gcn-caucafall: 
	# [check] train-gcn-caucafall
	@mkdir -p "$(HNM_DIR)"
	$(RUN) eval/mine_hard_negatives.py \
	  --ckpt "$(OUT_GCN_CAUC)$(OUT_TAG)/best.pt" \
	  --windows_dir "$(HNM_SRC_CAUC)" \
	  --neg_only 1 \
	  --out_txt "$(HNM_DIR)/caucafall_gcn.txt" \
	  --min_p "$(HNM_MIN_P)" --top_k "$(HNM_TOPK)" --max_per_clip "$(HNM_MAX_PER_CLIP)" --batch "$(HNM_BATCH)"

finetune-gcn-caucafall: mine-hardneg-gcn-caucafall
	$(RUN) models/train_gcn.py --train_dir "$(WIN_CAUC)/train" --val_dir "$(WIN_CAUC)/val" \
	  --epochs "$(HNM_EPOCHS)" --batch "$(BATCH)" --lr "$(HNM_LR)" --seed "$(SPLIT_SEED)" --patience "$(HNM_PATIENCE)" \
	  --resume "$(OUT_GCN_CAUC)$(OUT_TAG)/best.pt" \
	  --hard_neg_list "$(HNM_DIR)/caucafall_gcn.txt" --hard_neg_mult "$(HNM_MULT)" \
	  --save_dir "$(OUT_GCN_CAUC)$(OUT_TAG)$(HNM_TAG)"

hnm-caucafall: finetune-tcn-caucafall finetune-gcn-caucafall
	@echo "[ok] CAUCAFall HNM done"

# ---- MUVIM ----
mine-hardneg-tcn-muvim: 
	# [check] train-tcn-muvim
	@mkdir -p "$(HNM_DIR)"
	$(RUN) eval/mine_hard_negatives.py \
	  --ckpt "$(OUT_TCN_MUVIM)$(OUT_TAG)/best.pt" \
	  --windows_dir "$(HNM_SRC_MUVIM)" \
	  --neg_only 1 \
	  --out_txt "$(HNM_DIR)/muvim_tcn.txt" \
	  --min_p "$(HNM_MIN_P)" --top_k "$(HNM_TOPK)" --max_per_clip "$(HNM_MAX_PER_CLIP)" --batch "$(HNM_BATCH)"

finetune-tcn-muvim: mine-hardneg-tcn-muvim
	$(RUN) models/train_tcn.py --train_dir "$(WIN_MUVIM)/train" --val_dir "$(WIN_MUVIM)/val" \
	  --epochs "$(HNM_EPOCHS)" --batch "$(BATCH)" --lr "$(HNM_LR)" --seed "$(SPLIT_SEED)" --patience "$(HNM_PATIENCE)" \
	  --resume "$(OUT_TCN_MUVIM)$(OUT_TAG)/best.pt" \
	  --hard_neg_list "$(HNM_DIR)/muvim_tcn.txt" --hard_neg_mult "$(HNM_MULT)" \
	  --save_dir "$(OUT_TCN_MUVIM)$(OUT_TAG)$(HNM_TAG)"

mine-hardneg-gcn-muvim: 
	# [check] train-gcn-muvim
	@mkdir -p "$(HNM_DIR)"
	$(RUN) eval/mine_hard_negatives.py \
	  --ckpt "$(OUT_GCN_MUVIM)$(OUT_TAG)/best.pt" \
	  --windows_dir "$(HNM_SRC_MUVIM)" \
	  --neg_only 1 \
	  --out_txt "$(HNM_DIR)/muvim_gcn.txt" \
	  --min_p "$(HNM_MIN_P)" --top_k "$(HNM_TOPK)" --max_per_clip "$(HNM_MAX_PER_CLIP)" --batch "$(HNM_BATCH)"

finetune-gcn-muvim: mine-hardneg-gcn-muvim
	$(RUN) models/train_gcn.py --train_dir "$(WIN_MUVIM)/train" --val_dir "$(WIN_MUVIM)/val" \
	  --epochs "$(HNM_EPOCHS)" --batch "$(BATCH)" --lr "$(HNM_LR)" --seed "$(SPLIT_SEED)" --patience "$(HNM_PATIENCE)" \
	  --resume "$(OUT_GCN_MUVIM)$(OUT_TAG)/best.pt" \
	  --hard_neg_list "$(HNM_DIR)/muvim_gcn.txt" --hard_neg_mult "$(HNM_MULT)" \
	  --save_dir "$(OUT_GCN_MUVIM)$(OUT_TAG)$(HNM_TAG)"

hnm-muvim: finetune-tcn-muvim finetune-gcn-muvim
	@echo "[ok] MUVIM HNM done"

hnm-all: hnm-le2i hnm-urfd hnm-caucafall hnm-muvim
	@echo "[ok] HNM all datasets done"



# -------------------------
# HNM: fit/eval/plot convenience targets (keeps baseline reports intact)
# -------------------------

# ---- LE2i ----
fit-ops-le2i-hnm: finetune-tcn-le2i
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn --val_dir "$(WIN_LE2I)/val" \
	  --ckpt "$(OUT_TCN_LE2I)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --out  "$(OPS_DIR)/tcn_le2i$(HNM_TAG).yaml" \
	  --pose_npz_dir "$(POSE_LE2I)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

eval-le2i-hnm: fit-ops-le2i-hnm
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_LE2I)/test" \
	  --ckpt "$(OUT_TCN_LE2I)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --ops_yaml "$(OPS_DIR)/tcn_le2i$(HNM_TAG).yaml" \
	  --out_json "$(REPORTS_DIR)/le2i_tcn$(HNM_TAG).json" \
	  --pose_npz_dir "$(POSE_LE2I)" --stride_frames_hint "$(WIN_S)"

plot-le2i-hnm: eval-le2i-hnm
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/le2i_tcn$(HNM_TAG).json" --title "LE2i (TCN+HNM) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/le2i_tcn$(HNM_TAG)_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

fit-ops-gcn-le2i-hnm: finetune-gcn-le2i
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_LE2I)/val" \
	  --ckpt "$(OUT_GCN_LE2I)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --out  "$(OPS_DIR)/gcn_le2i$(HNM_TAG).yaml" \
	  --pose_npz_dir "$(POSE_LE2I)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

eval-le2i-gcn-hnm: fit-ops-gcn-le2i-hnm
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_LE2I)/test" \
	  --ckpt "$(OUT_GCN_LE2I)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --ops_yaml "$(OPS_DIR)/gcn_le2i$(HNM_TAG).yaml" \
	  --out_json "$(REPORTS_DIR)/le2i_gcn$(HNM_TAG).json" \
	  --pose_npz_dir "$(POSE_LE2I)" --stride_frames_hint "$(WIN_S)"

plot-le2i-gcn-hnm: eval-le2i-gcn-hnm
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/le2i_gcn$(HNM_TAG).json" --title "LE2i (GCN+HNM) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/le2i_gcn$(HNM_TAG)_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

# ---- URFD ----
fit-ops-urfd-hnm: finetune-tcn-urfd
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn --val_dir "$(WIN_URFD)/val" \
	  --ckpt "$(OUT_TCN_URFD)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --out  "$(OPS_DIR)/tcn_urfd$(HNM_TAG).yaml" \
	  --pose_npz_dir "$(POSE_URFD)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

eval-urfd-hnm: fit-ops-urfd-hnm
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_URFD)/test" \
	  --ckpt "$(OUT_TCN_URFD)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --ops_yaml "$(OPS_DIR)/tcn_urfd$(HNM_TAG).yaml" \
	  --out_json "$(REPORTS_DIR)/urfd_tcn$(HNM_TAG).json" \
	  --pose_npz_dir "$(POSE_URFD)" --stride_frames_hint "$(WIN_S)"

plot-urfd-hnm: eval-urfd-hnm
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/urfd_tcn$(HNM_TAG).json" --title "URFD (TCN+HNM) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/urfd_tcn$(HNM_TAG)_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

fit-ops-gcn-urfd-hnm: finetune-gcn-urfd
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_URFD)/val" \
	  --ckpt "$(OUT_GCN_URFD)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --out  "$(OPS_DIR)/gcn_urfd$(HNM_TAG).yaml" \
	  --pose_npz_dir "$(POSE_URFD)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

eval-urfd-gcn-hnm: fit-ops-gcn-urfd-hnm
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_URFD)/test" \
	  --ckpt "$(OUT_GCN_URFD)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --ops_yaml "$(OPS_DIR)/gcn_urfd$(HNM_TAG).yaml" \
	  --out_json "$(REPORTS_DIR)/urfd_gcn$(HNM_TAG).json" \
	  --pose_npz_dir "$(POSE_URFD)" --stride_frames_hint "$(WIN_S)"

plot-urfd-gcn-hnm: eval-urfd-gcn-hnm
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/urfd_gcn$(HNM_TAG).json" --title "URFD (GCN+HNM) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/urfd_gcn$(HNM_TAG)_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

# ---- CAUCAFall ----
fit-ops-caucafall-hnm: finetune-tcn-caucafall
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn --val_dir "$(WIN_CAUC)/val" \
	  --ckpt "$(OUT_TCN_CAUC)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --out  "$(OPS_DIR)/tcn_caucafall$(HNM_TAG).yaml" \
	  --pose_npz_dir "$(POSE_CAUC)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

eval-caucafall-hnm: fit-ops-caucafall-hnm
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_CAUC)/test" \
	  --ckpt "$(OUT_TCN_CAUC)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --ops_yaml "$(OPS_DIR)/tcn_caucafall$(HNM_TAG).yaml" \
	  --out_json "$(REPORTS_DIR)/caucafall_tcn$(HNM_TAG).json" \
	  --pose_npz_dir "$(POSE_CAUC)" --stride_frames_hint "$(WIN_S)"

plot-caucafall-hnm: eval-caucafall-hnm
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/caucafall_tcn$(HNM_TAG).json" --title "CAUCAFall (TCN+HNM) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/caucafall_tcn$(HNM_TAG)_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

fit-ops-gcn-caucafall-hnm: finetune-gcn-caucafall
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_CAUC)/val" \
	  --ckpt "$(OUT_GCN_CAUC)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --out  "$(OPS_DIR)/gcn_caucafall$(HNM_TAG).yaml" \
	  --pose_npz_dir "$(POSE_CAUC)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

eval-caucafall-gcn-hnm: fit-ops-gcn-caucafall-hnm
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_CAUC)/test" \
	  --ckpt "$(OUT_GCN_CAUC)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --ops_yaml "$(OPS_DIR)/gcn_caucafall$(HNM_TAG).yaml" \
	  --out_json "$(REPORTS_DIR)/caucafall_gcn$(HNM_TAG).json" \
	  --pose_npz_dir "$(POSE_CAUC)" --stride_frames_hint "$(WIN_S)"

plot-caucafall-gcn-hnm: eval-caucafall-gcn-hnm
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/caucafall_gcn$(HNM_TAG).json" --title "CAUCAFall (GCN+HNM) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/caucafall_gcn$(HNM_TAG)_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

# ---- MUVIM ----
fit-ops-muvim-hnm: finetune-tcn-muvim
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn --val_dir "$(WIN_MUVIM)/val" \
	  --ckpt "$(OUT_TCN_MUVIM)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --out  "$(OPS_DIR)/tcn_muvim$(HNM_TAG).yaml" \
	  --pose_npz_dir "$(POSE_MUVIM)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

eval-muvim-hnm: fit-ops-muvim-hnm
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_MUVIM)/test" \
	  --ckpt "$(OUT_TCN_MUVIM)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --ops_yaml "$(OPS_DIR)/tcn_muvim$(HNM_TAG).yaml" \
	  --out_json "$(REPORTS_DIR)/muvim_tcn$(HNM_TAG).json" \
	  --pose_npz_dir "$(POSE_MUVIM)" --stride_frames_hint "$(WIN_S)"

plot-muvim-hnm: eval-muvim-hnm
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/muvim_tcn$(HNM_TAG).json" --title "MUVIM (TCN+HNM) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/muvim_tcn$(HNM_TAG)_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

fit-ops-gcn-muvim-hnm: finetune-gcn-muvim
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_MUVIM)/val" \
	  --ckpt "$(OUT_GCN_MUVIM)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --out  "$(OPS_DIR)/gcn_muvim$(HNM_TAG).yaml" \
	  --pose_npz_dir "$(POSE_MUVIM)" --stride_frames_hint "$(WIN_S)" \
	  $(FITOPS_POLICY_FLAGS) \
	  $(FITOPS_SWEEP_FLAGS)

eval-muvim-gcn-hnm: fit-ops-gcn-muvim-hnm
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --test_dir "$(WIN_MUVIM)/test" \
	  --ckpt "$(OUT_GCN_MUVIM)$(OUT_TAG)$(HNM_TAG)/best.pt" \
	  --ops_yaml "$(OPS_DIR)/gcn_muvim$(HNM_TAG).yaml" \
	  --out_json "$(REPORTS_DIR)/muvim_gcn$(HNM_TAG).json" \
	  --pose_npz_dir "$(POSE_MUVIM)" --stride_frames_hint "$(WIN_S)"

plot-muvim-gcn-hnm: eval-muvim-gcn-hnm
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/muvim_gcn$(HNM_TAG).json" --title "MUVIM (GCN+HNM) — FA/24h vs Recall" \
	  --out_fig "$(FIG_DIR)/muvim_gcn$(HNM_TAG)_fa_recall_W$(WIN_W)_S$(WIN_S).png" --xlog --plot_pareto

hnm-report-all: plot-le2i-hnm plot-le2i-gcn-hnm plot-urfd-hnm plot-urfd-gcn-hnm plot-caucafall-hnm plot-caucafall-gcn-hnm plot-muvim-hnm plot-muvim-gcn-hnm
	@echo "[ok] HNM fit/eval/plot done for all"


# -------------------------
# Unlabeled windows (LE2i) + scoring (FA/hour) + HNM source
# -------------------------

# -------------------------
# Unlabeled windows (LE2i) + scoring (FA/hour) + HNM source
# -------------------------
# LE2i has Office/Lecture clips without fall labels. We window them into:
#   $(WIN_LE2I)/test_unlabeled
#
# You provide a stems list (one NPZ stem per line) that exists under $(POSE_LE2I).
# Example helper:
#   find $(POSE_LE2I) -name '*.npz' -print | sed 's#.*/##; s/\.npz$$//' > configs/unlabeled/le2i_unlabeled_stems.txt

LE2I_UNLABELED_STEMS ?= configs/unlabeled/le2i_unlabeled_stems.txt
UNLAB_SUBSET ?= test_unlabeled
WIN_LE2I_UNLAB := $(WIN_LE2I)/$(UNLAB_SUBSET)

UNLAB_MAX_WINDOWS_PER_VIDEO ?= 800
UNLAB_MIN_VALID_FRAC ?= 0.00
UNLAB_MIN_AVG_CONF ?= 0.00
UNLAB_SKIP_EXISTING ?= 1

UNLAB_MASK_FLAG :=
ifeq ($(USE_PRECOMPUTED_MASK),1)
UNLAB_MASK_FLAG := --use_precomputed_mask
endif

UNLAB_SKIP_FLAG :=
ifeq ($(UNLAB_SKIP_EXISTING),1)
UNLAB_SKIP_FLAG := --skip_existing
endif

.PHONY: le2i-unlabeled-list windows-le2i-unlabeled score-unlabeled-le2i-tcn score-unlabeled-le2i-gcn


.PHONY: le2i-unlabeled-list

# Auto-generate a default unlabeled stems list if missing.
# This list is intended to contain ADL-only / negative streams (e.g., office/lecture clips).
# By default we include:
#   - stems with label==0 (ADL) from configs/labels/le2i.json (if available)
#   - stems present under $(POSE_LE2I) but missing from le2i.json (unannotated sequences)
# You should edit configs/unlabeled/le2i_unlabeled_stems.txt to point to your true long ADL streams.
le2i-unlabeled-list:
	@mkdir -p "$(dir $(LE2I_UNLABELED_STEMS))"
	@if [ -f "$(LE2I_UNLABELED_STEMS)" ]; then \
		echo "[ok] unlabeled stems file exists: $(LE2I_UNLABELED_STEMS)"; \
	else \
		$(VENV) && PYTHONPATH="$(PWD)" $(PY) -c "import json; from pathlib import Path; pose_dir=Path('$(POSE_LE2I)'); labels_path=Path('configs/labels/le2i.json'); out_path=Path('$(LE2I_UNLABELED_STEMS)'); out_path.parent.mkdir(parents=True, exist_ok=True); all_stems=sorted(p.stem for p in pose_dir.glob('*.npz')); labels=json.loads(labels_path.read_text(encoding='utf-8')) if labels_path.exists() else {}; adl=[s for s,v in labels.items() if ((isinstance(v,(int,float)) and int(v)==0) or (isinstance(v,str) and v.strip().lower() in ('0','adl','nonfall','normal','no_fall','nofall')) )]; unl=[s for s in all_stems if s not in labels]; cand=adl+unl; pref=[s for s in cand if ('office' in s.lower() or 'lecture' in s.lower())]; rest=[s for s in cand if s not in pref]; final=list(dict.fromkeys(pref+rest)); out_path.write_text('\n'.join(final)+('\n' if final else ''), encoding='utf-8'); print(f'[gen] wrote unlabeled stems: {len(final)} -> {out_path}'); import sys; sys.exit(1 if not final else 0)"; \
	fi

windows-le2i-unlabeled: preprocess-le2i-only le2i-unlabeled-list
	@mkdir -p "configs/unlabeled"
	@if [ ! -f "$(LE2I_UNLABELED_STEMS)" ]; then \
	  echo "[err] missing stems file: $(LE2I_UNLABELED_STEMS)"; \
	  echo "      Create it (one stem per line; stems must exist under $(POSE_LE2I))."; \
	  exit 2; \
	fi
	@mkdir -p "$(WIN_LE2I)"
	$(RUN) windows/make_unlabeled_windows.py \
	  --npz_dir "$(POSE_LE2I)" \
	  --stems_txt "$(LE2I_UNLABELED_STEMS)" \
	  --out_dir "$(WIN_LE2I)" --subset "$(UNLAB_SUBSET)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_LE2I)" \
	  --conf_gate "$(CONF_GATE)" --min_valid_frac "$(UNLAB_MIN_VALID_FRAC)" --min_avg_conf "$(UNLAB_MIN_AVG_CONF)" \
	  --max_windows_per_video "$(UNLAB_MAX_WINDOWS_PER_VIDEO)" \
	  $(UNLAB_MASK_FLAG) \
	  $(UNLAB_SKIP_FLAG)

# Score unlabeled windows using the real-time alert policy (EMA + k-of-n + hysteresis + cooldown).
# This estimates "false alerts per hour/day" on long negative streams.
score-unlabeled-le2i-tcn: windows-le2i-unlabeled
	@mkdir -p "$(REPORTS_DIR)"
	@if [ ! -f "$(CKPT_TCN_LE2I)" ]; then \
	  echo "[err] missing checkpoint: $(CKPT_TCN_LE2I)"; \
	  echo "      Run: make train-tcn-le2i (or set CKPT_TCN_LE2I=...)"; \
	  exit 2; \
	fi
	$(RUN) eval/score_unlabeled_alert_rate.py \
	  --win_dir "$(WIN_LE2I_UNLAB)" --ckpt "$(CKPT_TCN_LE2I)" \
	  --ema_alpha "$(ALERT_EMA_ALPHA)" --k "$(ALERT_K)" --n "$(ALERT_N)" \
	  --tau_high "$(ALERT_TAU_HIGH)" --tau_low "$(ALERT_TAU_LOW)" --cooldown_s "$(ALERT_COOLDOWN_S)" \
	  --out_json "$(REPORTS_DIR)/le2i_tcn_unlabeled_alerts.json"

score-unlabeled-le2i-gcn: windows-le2i-unlabeled
	@mkdir -p "$(REPORTS_DIR)"
	@if [ ! -f "$(CKPT_GCN_LE2I)" ]; then \
	  echo "[err] missing checkpoint: $(CKPT_GCN_LE2I)"; \
	  echo "      Run: make train-gcn-le2i (or set CKPT_GCN_LE2I=...)"; \
	  exit 2; \
	fi
	$(RUN) eval/score_unlabeled_alert_rate.py \
	  --win_dir "$(WIN_LE2I_UNLAB)" --ckpt "$(CKPT_GCN_LE2I)" \
	  --ema_alpha "$(ALERT_EMA_ALPHA)" --k "$(ALERT_K)" --n "$(ALERT_N)" \
	  --tau_high "$(ALERT_TAU_HIGH)" --tau_low "$(ALERT_TAU_LOW)" --cooldown_s "$(ALERT_COOLDOWN_S)" \
	  --out_json "$(REPORTS_DIR)/le2i_gcn_unlabeled_alerts.json"

# ---- Unlabeled scoring presets (≈1s persistence) ----
.PHONY: score-unlabeled-le2i-tcn-s4 score-unlabeled-le2i-tcn-s8
.PHONY: score-unlabeled-le2i-gcn-s4 score-unlabeled-le2i-gcn-s8

score-unlabeled-le2i-tcn-s4:
	$(MAKE) WIN_W=48 WIN_S=4 windows-le2i-unlabeled
	$(MAKE) WIN_W=48 WIN_S=4 score-unlabeled-le2i-tcn ALERT_N=7 ALERT_K=5

score-unlabeled-le2i-tcn-s8:
	$(MAKE) WIN_W=48 WIN_S=8 windows-le2i-unlabeled
	$(MAKE) WIN_W=48 WIN_S=8 score-unlabeled-le2i-tcn ALERT_N=4 ALERT_K=3

score-unlabeled-le2i-gcn-s4:
	$(MAKE) WIN_W=48 WIN_S=4 windows-le2i-unlabeled
	$(MAKE) WIN_W=48 WIN_S=4 score-unlabeled-le2i-gcn ALERT_N=7 ALERT_K=5 

score-unlabeled-le2i-gcn-s8:
	$(MAKE) WIN_W=48 WIN_S=8 windows-le2i-unlabeled
	$(MAKE) WIN_W=48 WIN_S=8 score-unlabeled-le2i-gcn ALERT_N=4 ALERT_K=3


# If you want HNM to mine from unlabeled streams (recommended for LE2i), run:
#   make HNM_SRC_LE2I=$(WIN_LE2I)/test_unlabeled mine-hardneg-tcn-le2i
#   make HNM_SRC_LE2I=$(WIN_LE2I)/test_unlabeled mine-hardneg-gcn-le2i

