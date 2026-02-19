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

# Windows (default: W48 S12)
WIN_W ?= 48
WIN_S ?= 12

# Safety: avoid stale windows causing split leakage.
# Set WIN_CLEAN=1 to remove existing train/val/test dirs before regenerating windows.
WIN_CLEAN ?= 1
# Separate knob for evaluation windows.
WIN_EVAL_CLEAN ?= 0

# Pose preprocess knobs
CONF_THR   ?= 0.20
SMOOTH_K   ?= 5
MAX_GAP    ?= 4
NORM_MODE  ?= torso
PELVIS_FILL?= nearest

# Common train knobs
EPOCHS ?= 200
# -------------------------
# GCN training knobs
# -------------------------
OUT_TAG ?=
LR_GCN ?= $(LR)
EPOCHS_GCN ?= $(EPOCHS)
BATCH_GCN ?= $(BATCH)

GCN_PATIENCE ?= 30
GCN_GCN_HIDDEN ?= 96
GCN_TCN_HIDDEN ?= 192
GCN_DROPOUT ?= 0.35
GCN_USE_SE ?= 1
GCN_TWO_STREAM ?= 1
GCN_FUSE ?= concat

# New GCN trainer model knobs (CTR-GCN style)
GCN_HIDDEN ?= 96
GCN_NUM_BLOCKS ?= 6
GCN_TEMPORAL_KERNEL ?= 9
GCN_BASE_CHANNELS ?= 48

MASK_JOINT_P ?= 0.15
MASK_FRAME_P ?= 0.10
GCN_POS_WEIGHT ?= auto
GCN_BALANCED_SAMPLER ?= 0
GCN_BALANCED_SAMPLER_FLAG = $(if $(filter 1,$(GCN_BALANCED_SAMPLER)),--balanced_sampler,)

GCN_WEIGHT_DECAY ?= 1e-4
GCN_MONITOR ?= f1
GCN_LABEL_SMOOTHING ?= 0.0
GCN_NUM_WORKERS ?= 0

# Loss (default bce; focal can help with heavy imbalance)
GCN_LOSS ?= bce
GCN_FOCAL_ALPHA ?= 0.25
GCN_FOCAL_GAMMA ?= 2.0


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
TCN_PATIENCE ?= 30
TCN_GRAD_CLIP ?= 1.0
TCN_POS_WEIGHT ?= auto
TCN_BALANCED_SAMPLER ?= 0
TCN_MASK_JOINT_P ?= 0.15
TCN_MASK_FRAME_P ?= 0.10
TCN_MONITOR ?= f1

# Threshold sweep for best-F1 selection on val
TCN_THR_MIN ?= 0.05
TCN_THR_MAX ?= 0.95
TCN_THR_STEP ?= 0.01

# Loss (default bce; focal can help with heavy imbalance)
TCN_LOSS ?= bce
TCN_FOCAL_ALPHA ?= 0.25
TCN_FOCAL_GAMMA ?= 2.0

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
ALERT_CONFIRM ?= 1
ALERT_CONFIRM_S ?= 2.0
ALERT_CONFIRM_MIN_LYING ?= 0.65
ALERT_CONFIRM_MAX_MOTION ?= 0.08
ALERT_CONFIRM_REQUIRE_LOW ?= 1

# --- fit_ops threshold sweep (tau_high is FITTED on val; tau_low = tau_low_ratio * tau_high) ---
FIT_THR_MIN ?= 0.01
FIT_THR_MAX ?= 0.95
FIT_THR_STEP ?= 0.01
# Hysteresis ratio used by fit_ops.py to derive tau_low from tau_high:
#   tau_low = tau_high * FIT_TAU_LOW_RATIO
# Supervisor-friendly default: if tau_high≈0.90 then tau_low≈0.70.
FIT_TAU_LOW_RATIO ?= 0.78

# Event grouping + matching (seconds)
FIT_TIME_MODE ?= center  # start|center|end mapping from window index → time
FIT_MERGE_GAP_S ?= 1.0  # merge predicted alert events if gap <= this
FIT_OVERLAP_SLACK_S ?= 0.5  # slack when matching predicted alerts to GT falls

# OP selection targets (tune per dataset if needed)
FIT_OP1_RECALL ?= 0.95  # High Safety: pick first OP with recall >= this (if possible)
FIT_OP3_FA24H ?= 1.0  # Low Alarms: pick first OP with FA/24h <= this (if possible)

# fit_ops composed flags
FITOPS_POLICY_FLAGS = --ema_alpha "$(ALERT_EMA_ALPHA)" --k "$(ALERT_K)" --n "$(ALERT_N)" --cooldown_s "$(ALERT_COOLDOWN_S)" --tau_low_ratio "$(FIT_TAU_LOW_RATIO)" --confirm "$(ALERT_CONFIRM)" --confirm_s "$(ALERT_CONFIRM_S)" --confirm_min_lying "$(ALERT_CONFIRM_MIN_LYING)" --confirm_max_motion "$(ALERT_CONFIRM_MAX_MOTION)" --confirm_require_low "$(ALERT_CONFIRM_REQUIRE_LOW)"
FITOPS_SWEEP_FLAGS  = --thr_min "$(FIT_THR_MIN)" --thr_max "$(FIT_THR_MAX)" --thr_step "$(FIT_THR_STEP)" --time_mode "$(strip $(FIT_TIME_MODE))" --merge_gap_s "$(strip $(FIT_MERGE_GAP_S))" --overlap_slack_s "$(strip $(FIT_OVERLAP_SLACK_S))" --op1_recall "$(strip $(FIT_OP1_RECALL))" --op3_fa24h "$(strip $(FIT_OP3_FA24H))"

# --- fit_ops picker + tie-break (used by eval/fit_ops.py) ---
FITOPS_PICKER ?= conservative    # conservative|core
FITOPS_TIE_BREAK ?= max_thr      # max_thr|min_thr
FITOPS_TIE_EPS ?= 1e-3
FITOPS_SAVE_SWEEP_JSON ?= 1

# Minimum deployable tau_high (floor) per dataset (prevents ultra-low thresholds)
FITOPS_MIN_TAU_HIGH_LE2I ?= 0.20
FITOPS_MIN_TAU_HIGH_URFD ?= 0.20
FITOPS_MIN_TAU_HIGH_CAUC ?= 0.20
FITOPS_MIN_TAU_HIGH_MUVIM ?= 0.20

FITOPS_PICKER_FLAGS = --ops_picker "$(strip $(FITOPS_PICKER))" --op_tie_break "$(strip $(FITOPS_TIE_BREAK))" --tie_eps "$(strip $(FITOPS_TIE_EPS))" --save_sweep_json "$(strip $(FITOPS_SAVE_SWEEP_JSON))"

# --- metrics.py sweep (for FA/hour vs Recall curves) ---
METR_THR_MIN ?= 0.001
METR_THR_MAX ?= 0.95
METR_THR_STEP ?= 0.01
METRICS_SWEEP_FLAGS = --thr_min "$(METR_THR_MIN)" --thr_max "$(METR_THR_MAX)" --thr_step "$(METR_THR_STEP)"

# --- plotting mode (FULL by default; set PLOT_PARETO=1 for frontier only) ---
PLOT_PARETO ?= 0
PLOT_PARETO_FLAG = $(if $(filter 1 yes true,$(strip $(PLOT_PARETO))),--plot_pareto,)



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
WIN_EXTRA ?= --strategy balanced --min_overlap_frames 1 --pos_per_span 20 --neg_ratio 2.0 --max_neg_per_video 250 --max_windows_per_video_no_spans 120 --min_valid_frac 0.0 --spans_end_exclusive

# ---- FA windows (negative-only stream for FA/24h fitting) ----
# Optional "normal-life" windows used for estimating FA/24h more realistically.
# Create them from evaluation windows:
#   make fa-windows-le2i
#   make fa-windows-caucafall
ONLY_NEG_VIDEOS ?= 0
FA_MODE ?= symlink   # symlink|copy

FA_WIN_LE2I  ?= $(PROCESSED)/le2i/fa_windows_W$(WIN_W)_S$(WIN_S)
FA_WIN_URFD  ?= $(PROCESSED)/urfd/fa_windows_W$(WIN_W)_S$(WIN_S)
FA_WIN_CAUC  ?= $(PROCESSED)/caucafall/fa_windows_W$(WIN_W)_S$(WIN_S)
FA_WIN_MUVIM ?= $(PROCESSED)/muvim/fa_windows_W$(WIN_W)_S$(WIN_S)

# FITOPS: optionally pass --fa_dir to eval/fit_ops.py
FITOPS_USE_FA ?= 0
FITOPS_FA_DIR_LE2I  ?= $(FA_WIN_LE2I)/val
FITOPS_FA_DIR_URFD  ?= $(FA_WIN_URFD)/val
FITOPS_FA_DIR_CAUC  ?= $(FA_WIN_CAUC)/val
FITOPS_FA_DIR_MUVIM ?= $(FA_WIN_MUVIM)/val

FITOPS_FA_ARG_LE2I  = $(if $(and $(filter 1,$(strip $(FITOPS_USE_FA))),$(strip $(FITOPS_FA_DIR_LE2I))),--fa_dir "$(strip $(FITOPS_FA_DIR_LE2I))",)
FITOPS_FA_ARG_URFD  = $(if $(and $(filter 1,$(strip $(FITOPS_USE_FA))),$(strip $(FITOPS_FA_DIR_URFD))),--fa_dir "$(strip $(FITOPS_FA_DIR_URFD))",)
FITOPS_FA_ARG_CAUC  = $(if $(and $(filter 1,$(strip $(FITOPS_USE_FA))),$(strip $(FITOPS_FA_DIR_CAUC))),--fa_dir "$(strip $(FITOPS_FA_DIR_CAUC))",)
FITOPS_FA_ARG_MUVIM = $(if $(and $(filter 1,$(strip $(FITOPS_USE_FA))),$(strip $(FITOPS_FA_DIR_MUVIM))),--fa_dir "$(strip $(FITOPS_FA_DIR_MUVIM))",)

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

FEAT_CENTER ?= pelvis
FEAT_SCALE ?= torso
CENTER ?= $(FEAT_CENTER)
SCALE ?= $(FEAT_SCALE)
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
	@mkdir -p "$(POSE_MUVIM_RAW)"
	$(RUN) pose/extract_2d_from_images.py \
	  --images_glob "$(RAW_MUVIM)/ZED_RGB/**/*.jpg" "$(RAW_MUVIM)/ZED_RGB/**/*.png" \
	  --sequence_id_depth "2" \
	  --out_dir "$(POSE_MUVIM_RAW)" \
	  --dataset muvim \
	  --fps "$(FPS_MUVIM)" \
	  --skip_existing


# -------------------------
# Preprocess (clean/gate/smooth/normalize)
# -------------------------
.PHONY: preprocess-le2i preprocess-urfd preprocess-caucafall preprocess-muvim
.PHONY: preprocess-le2i-only preprocess-urfd-only preprocess-caucafall-only preprocess-muvim-only

preprocess-le2i:
	$(MAKE) extract-le2i
	$(MAKE) preprocess-le2i-only
preprocess-le2i-only:
	@mkdir -p "$(POSE_LE2I)"
	$(RUN) pose/preprocess_pose_npz.py \
	  --in_dir  "$(POSE_LE2I_RAW)" \
	  --out_dir "$(POSE_LE2I)" \
	  --recursive --skip_existing \
	  --conf_thr "$(CONF_THR)" --smooth_k "$(SMOOTH_K)" --max_gap "$(MAX_GAP)" \
	  --normalize "$(NORM_MODE)" --pelvis_fill "$(PELVIS_FILL)"

preprocess-urfd:
	$(MAKE) extract-urfd
	$(MAKE) preprocess-urfd-only
preprocess-urfd-only:
	@mkdir -p "$(POSE_URFD)"
	$(RUN) pose/preprocess_pose_npz.py \
	  --in_dir  "$(POSE_URFD_RAW)" \
	  --out_dir "$(POSE_URFD)" \
	  --recursive --skip_existing \
	  --conf_thr "$(CONF_THR)" --smooth_k "$(SMOOTH_K)" --max_gap "$(MAX_GAP)" \
	  --normalize "$(NORM_MODE)" --pelvis_fill "$(PELVIS_FILL)"

preprocess-caucafall:
	$(MAKE) extract-caucafall
	$(MAKE) preprocess-caucafall-only
preprocess-caucafall-only:
	@mkdir -p "$(POSE_CAUC)"
	$(RUN) pose/preprocess_pose_npz.py \
	  --in_dir  "$(POSE_CAUC_RAW)" \
	  --out_dir "$(POSE_CAUC)" \
	  --recursive --skip_existing \
	  --conf_thr "$(CONF_THR)" --smooth_k "$(SMOOTH_K)" --max_gap "$(MAX_GAP)" \
	  --normalize "$(NORM_MODE)" --pelvis_fill "$(PELVIS_FILL)"

preprocess-muvim:
	$(MAKE) extract-muvim
	$(MAKE) preprocess-muvim-only
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
USE_PER_FRAME_ACTION_TXT ?= 1

CAUCA_MIN_RUN ?= 3
CAUCA_GAP_FILL ?= 2

# Span enforcement (recommended for realistic event metrics)
CAUCA_REQUIRE_SPANS ?= 1

# Leak-safe split for CAUCAFall (subjects)
CAUCA_SPLIT_GROUP_MODE ?= caucafall_subject
CAUCA_SPLIT_BALANCE_BY ?= groups

.PHONY: labels-le2i labels-urfd labels-caucafall labels-muvim

labels-le2i:
	@if [ ! -d "$(POSE_LE2I)" ] || [ "$$(find "$(POSE_LE2I)" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)" -eq 0 ]; then \
	  echo "[err] missing or empty pose_npz: $(POSE_LE2I)"; \
	  echo "      Run: make preprocess-le2i-only   (or preprocess-le2i)"; \
	  exit 2; \
	fi
	@mkdir -p "$(LABELS_DIR)"
	$(RUN) labels/make_le2i_labels.py \
	  --npz_dir "$(POSE_LE2I)" \
	  --raw_root "$(RAW_LE2I)" \
	  --out_labels "$(LABELS_LE2I)" \
	  --out_spans  "$(SPANS_LE2I)"

labels-urfd:
	@if [ ! -d "$(POSE_URFD)" ] || [ "$$(find "$(POSE_URFD)" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)" -eq 0 ]; then \
	  echo "[err] missing or empty pose_npz: $(POSE_URFD)"; \
	  echo "      Run: make preprocess-urfd-only   (or preprocess-urfd)"; \
	  exit 2; \
	fi
	@mkdir -p "$(LABELS_DIR)"
	@ANN=""; \
	if [ -n "$(URFD_ANN_GLOB)" ]; then \
	  ANN="--ann_glob \"$(URFD_ANN_GLOB)\" --use_per_frame_action_txt $(USE_PER_FRAME_ACTION_TXT) --fall_class_id $(URFD_FALL_CLASS_ID) --min_run $(URFD_MIN_RUN) --gap_fill $(URFD_GAP_FILL)"; \
	fi; \
	eval "$(RUN) labels/make_urfd_labels.py --npz_dir \"$(POSE_URFD)\" --out_labels \"$(LABELS_URFD)\" --out_spans \"$(SPANS_URFD)\" $$ANN"

labels-caucafall:
	@if [ ! -d "$(POSE_CAUC)" ] || [ "$$(find "$(POSE_CAUC)" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)" -eq 0 ]; then \
	  echo "[err] missing or empty pose_npz: $(POSE_CAUC)"; \
	  echo "      Run: make preprocess-caucafall-only   (or preprocess-caucafall)"; \
	  exit 2; \
	fi
	@mkdir -p "$(LABELS_DIR)"
	@SPAN_ARGS=""; \
	if [ "$(USE_PER_FRAME_ACTION_TXT)" = "1" ]; then \
	  SPAN_ARGS="--use_per_frame_action_txt 1 --fall_class_id $(CAUCA_FALL_CLASS_ID) --min_run $(CAUCA_MIN_RUN) --gap_fill $(CAUCA_GAP_FILL) --frame_label_mode auto --clamp_to_npz_len"; \
	fi; \
	eval "$(RUN) labels/make_caucafall_labels.py \
	  --raw_root \"$(RAW_CAUC)\" \
	  --npz_dir \"$(POSE_CAUC)\" \
	  --out_labels \"$(LABELS_CAUC)\" \
	  --out_spans \"$(SPANS_CAUC)\" \
	  $$SPAN_ARGS \
	  --verbose"

labels-muvim:
	@if [ ! -d "$(POSE_MUVIM)" ] || [ "$$(find "$(POSE_MUVIM)" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)" -eq 0 ]; then \
	  echo "[err] missing or empty pose_npz: $(POSE_MUVIM)"; \
	  echo "      Run: make preprocess-muvim-only   (or preprocess-muvim)"; \
	  exit 2; \
	fi
	@mkdir -p "$(LABELS_DIR)"
	$(RUN) labels/make_muvim_labels.py \
	  --npz_dir "$(POSE_MUVIM)" \
	  --zed_csv "$(MUVIM_ZED_CSV)" \
	  --out_labels "$(LABELS_MUVIM)" \
	  --out_spans  "$(SPANS_MUVIM)" \
	  --stop_inclusive

# -------------------------
# Splits
# -------------------------
.PHONY: splits-le2i splits-urfd splits-caucafall splits-muvim

splits-le2i:
	@if [ ! -f "$(LABELS_LE2I)" ]; then \
	  echo "[err] missing labels: $(LABELS_LE2I)"; \
	  echo "      Run: make labels-le2i"; \
	  exit 2; \
	fi
	@mkdir -p "$(SPLITS_DIR)"
	$(RUN) split/make_splits.py \
	  --labels_json "$(LABELS_LE2I)" \
	  --out_dir "$(SPLITS_DIR)" \
	  --prefix le2i \
	  --train "$(TRAIN_FRAC)" --val "$(VAL_FRAC)" --test "$(TEST_FRAC)" \
	  --seed "$(SPLIT_SEED)" \
	  --summary_json "$(SPLITS_DIR)/le2i_split_summary.json"

splits-urfd:
	@if [ ! -f "$(LABELS_URFD)" ]; then \
	  echo "[err] missing labels: $(LABELS_URFD)"; \
	  echo "      Run: make labels-urfd"; \
	  exit 2; \
	fi
	@mkdir -p "$(SPLITS_DIR)"
	$(RUN) split/make_splits.py \
	  --labels_json "$(LABELS_URFD)" \
	  --out_dir "$(SPLITS_DIR)" \
	  --prefix urfd \
	  --train "$(TRAIN_FRAC)" --val "$(VAL_FRAC)" --test "$(TEST_FRAC)" \
	  --seed "$(SPLIT_SEED)" \
	  --summary_json "$(SPLITS_DIR)/urfd_split_summary.json"

splits-caucafall:
	@if [ ! -f "$(LABELS_CAUC)" ]; then \
	  echo "[err] missing labels: $(LABELS_CAUC)"; \
	  echo "      Run: make labels-caucafall"; \
	  exit 2; \
	fi
	@mkdir -p "$(SPLITS_DIR)"
	$(RUN) split/make_splits.py \
	  --labels_json "$(LABELS_CAUC)" \
	  --out_dir "$(SPLITS_DIR)" \
	  --prefix caucafall \
	  --group_mode "$(CAUCA_SPLIT_GROUP_MODE)" --balance_by "$(CAUCA_SPLIT_BALANCE_BY)" \
	  --train "$(TRAIN_FRAC)" --val "$(VAL_FRAC)" --test "$(TEST_FRAC)" \
	  --seed "$(SPLIT_SEED)" \
	  --summary_json "$(SPLITS_DIR)/caucafall_split_summary.json"

splits-muvim:
	@if [ ! -f "$(LABELS_MUVIM)" ]; then \
	  echo "[err] missing labels: $(LABELS_MUVIM)"; \
	  echo "      Run: make labels-muvim"; \
	  exit 2; \
	fi
	@mkdir -p "$(SPLITS_DIR)"
	$(RUN) split/make_splits.py \
	  --labels_json "$(LABELS_MUVIM)" \
	  --out_dir "$(SPLITS_DIR)" \
	  --prefix muvim \
	  --train "$(TRAIN_FRAC)" --val "$(VAL_FRAC)" --test "$(TEST_FRAC)" \
	  --seed "$(SPLIT_SEED)" \
	  --summary_json "$(SPLITS_DIR)/muvim_split_summary.json"

# -------------------------
# Windows
# -------------------------
.PHONY: windows-le2i windows-urfd windows-caucafall windows-muvim

windows-le2i:
	@if [ ! -d "$(POSE_LE2I)" ] || [ "$$(find "$(POSE_LE2I)" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)" -eq 0 ]; then \
	  echo "[err] missing or empty pose_npz: $(POSE_LE2I)"; \
	  echo "      Run: make preprocess-le2i-only   (or preprocess-le2i)"; \
	  exit 2; \
	fi
	@if [ ! -f "$(LABELS_LE2I)" ]; then \
	  echo "[err] missing labels: $(LABELS_LE2I)"; \
	  echo "      Run: make labels-le2i"; \
	  exit 2; \
	fi
	@if [ ! -f "$(SPLIT_LE2I_TRAIN)" ] || [ ! -f "$(SPLIT_LE2I_VAL)" ] || [ ! -f "$(SPLIT_LE2I_TEST)" ]; then \
	  echo "[err] missing split lists under configs/splits for le2i"; \
	  echo "      Expected: $(SPLIT_LE2I_TRAIN), $(SPLIT_LE2I_VAL), $(SPLIT_LE2I_TEST)"; \
	  echo "      Run: make splits-le2i"; \
	  exit 2; \
	fi
	@if [ "$(WIN_CLEAN)" = "1" ]; then \
	  echo "[clean] removing existing split dirs under $(WIN_LE2I)"; \
	  rm -rf "$(WIN_LE2I)/train" "$(WIN_LE2I)/val" "$(WIN_LE2I)/test"; \
	fi
	@mkdir -p "$(WIN_LE2I)"
	$(RUN) windows/make_windows.py \
	  --npz_dir "$(POSE_LE2I)" \
	  --labels_json "$(LABELS_LE2I)" \
	  --spans_json  "$(SPANS_LE2I)" \
	  --out_dir "$(WIN_LE2I)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_LE2I)" \
	  --train_list "$(SPLIT_LE2I_TRAIN)" --val_list "$(SPLIT_LE2I_VAL)" --test_list "$(SPLIT_LE2I_TEST)" \
	  $(WIN_EXTRA)

windows-urfd:
	@if [ ! -d "$(POSE_URFD)" ] || [ "$$(find "$(POSE_URFD)" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)" -eq 0 ]; then \
	  echo "[err] missing or empty pose_npz: $(POSE_URFD)"; \
	  echo "      Run: make preprocess-urfd-only   (or preprocess-urfd)"; \
	  exit 2; \
	fi
	@if [ ! -f "$(LABELS_URFD)" ]; then \
	  echo "[err] missing labels: $(LABELS_URFD)"; \
	  echo "      Run: make labels-urfd"; \
	  exit 2; \
	fi
	@if [ ! -f "$(SPLIT_URFD_TRAIN)" ] || [ ! -f "$(SPLIT_URFD_VAL)" ] || [ ! -f "$(SPLIT_URFD_TEST)" ]; then \
	  echo "[err] missing split lists under configs/splits for urfd"; \
	  echo "      Expected: $(SPLIT_URFD_TRAIN), $(SPLIT_URFD_VAL), $(SPLIT_URFD_TEST)"; \
	  echo "      Run: make splits-urfd"; \
	  exit 2; \
	fi
	@if [ "$(WIN_CLEAN)" = "1" ]; then \
	  echo "[clean] removing existing split dirs under $(WIN_URFD)"; \
	  rm -rf "$(WIN_URFD)/train" "$(WIN_URFD)/val" "$(WIN_URFD)/test"; \
	fi
	@mkdir -p "$(WIN_URFD)"
	$(RUN) windows/make_windows.py \
	  --npz_dir "$(POSE_URFD)" \
	  --labels_json "$(LABELS_URFD)" \
	  --spans_json  "$(SPANS_URFD)" \
	  --out_dir "$(WIN_URFD)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_URFD)" \
	  --train_list "$(SPLIT_URFD_TRAIN)" --val_list "$(SPLIT_URFD_VAL)" --test_list "$(SPLIT_URFD_TEST)" \
	  $(WIN_EXTRA)

windows-caucafall:
	@if [ ! -d "$(POSE_CAUC)" ] || [ "$$(find "$(POSE_CAUC)" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)" -eq 0 ]; then \
	  echo "[err] missing or empty pose_npz: $(POSE_CAUC)"; \
	  echo "      Run: make preprocess-caucafall-only   (or preprocess-caucafall)"; \
	  exit 2; \
	fi
	@if [ ! -f "$(LABELS_CAUC)" ]; then \
	  echo "[err] missing labels: $(LABELS_CAUC)"; \
	  echo "      Run: make labels-caucafall"; \
	  exit 2; \
	fi
	@if [ ! -f "$(SPLIT_CAUC_TRAIN)" ] || [ ! -f "$(SPLIT_CAUC_VAL)" ] || [ ! -f "$(SPLIT_CAUC_TEST)" ]; then \
	  echo "[err] missing split lists under configs/splits for caucafall"; \
	  echo "      Expected: $(SPLIT_CAUC_TRAIN), $(SPLIT_CAUC_VAL), $(SPLIT_CAUC_TEST)"; \
	  echo "      Run: make splits-caucafall"; \
	  exit 2; \
	fi
	$(MAKE) check-spans-caucafall
	@if [ "$(WIN_CLEAN)" = "1" ]; then \
	  echo "[clean] removing existing split dirs under $(WIN_CAUC)"; \
	  rm -rf "$(WIN_CAUC)/train" "$(WIN_CAUC)/val" "$(WIN_CAUC)/test"; \
	fi
	@mkdir -p "$(WIN_CAUC)"
	$(RUN) windows/make_windows.py \
	  --npz_dir "$(POSE_CAUC)" \
	  --labels_json "$(LABELS_CAUC)" \
	  --spans_json  "$(SPANS_CAUC)" \
	  --require_spans "$(CAUCA_REQUIRE_SPANS)" \
	  --out_dir "$(WIN_CAUC)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_CAUC)" \
	  --train_list "$(SPLIT_CAUC_TRAIN)" --val_list "$(SPLIT_CAUC_VAL)" --test_list "$(SPLIT_CAUC_TEST)" \
	  $(WIN_EXTRA)

windows-muvim:
	@if [ ! -d "$(POSE_MUVIM)" ] || [ "$$(find "$(POSE_MUVIM)" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)" -eq 0 ]; then \
	  echo "[err] missing or empty pose_npz: $(POSE_MUVIM)"; \
	  echo "      Run: make preprocess-muvim-only   (or preprocess-muvim)"; \
	  exit 2; \
	fi
	@if [ ! -f "$(LABELS_MUVIM)" ]; then \
	  echo "[err] missing labels: $(LABELS_MUVIM)"; \
	  echo "      Run: make labels-muvim"; \
	  exit 2; \
	fi
	@if [ ! -f "$(SPLIT_MUVIM_TRAIN)" ] || [ ! -f "$(SPLIT_MUVIM_VAL)" ] || [ ! -f "$(SPLIT_MUVIM_TEST)" ]; then \
	  echo "[err] missing split lists under configs/splits for muvim"; \
	  echo "      Expected: $(SPLIT_MUVIM_TRAIN), $(SPLIT_MUVIM_VAL), $(SPLIT_MUVIM_TEST)"; \
	  echo "      Run: make splits-muvim"; \
	  exit 2; \
	fi
	@if [ "$(WIN_CLEAN)" = "1" ]; then \
	  echo "[clean] removing existing split dirs under $(WIN_MUVIM)"; \
	  rm -rf "$(WIN_MUVIM)/train" "$(WIN_MUVIM)/val" "$(WIN_MUVIM)/test"; \
	fi
	@mkdir -p "$(WIN_MUVIM)"
	$(RUN) windows/make_windows.py \
	  --npz_dir "$(POSE_MUVIM)" \
	  --labels_json "$(LABELS_MUVIM)" \
	  --spans_json  "$(SPANS_MUVIM)" \
	  --fallback_if_no_span skip_fall \
	  --out_dir "$(WIN_MUVIM)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_MUVIM)" \
	  --train_list "$(SPLIT_MUVIM_TRAIN)" --val_list "$(SPLIT_MUVIM_VAL)" --test_list "$(SPLIT_MUVIM_TEST)" \
	  $(WIN_EXTRA)



# -------------------------
# -------------------------
# Check spans (sanity checks)
# -------------------------
.PHONY: check-spans-caucafall

check-spans-caucafall:
	$(RUN) core/check_spans.py --labels_json "$(LABELS_CAUC)" --spans_json "$(SPANS_CAUC)" --require_nonempty 1

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


.PHONY: fa-windows-le2i fa-windows-caucafall
fa-windows-le2i: windows-eval-le2i
	@mkdir -p "$(FA_WIN_LE2I)"
	$(RUN) windows/make_fa_windows.py --in_root "$(WIN_LE2I_EVAL)" --out_root "$(FA_WIN_LE2I)" --split val --mode "$(strip $(FA_MODE))" --only_neg_videos "$(ONLY_NEG_VIDEOS)"

fa-windows-caucafall: windows-eval-caucafall
	@mkdir -p "$(FA_WIN_CAUC)"
	$(RUN) windows/make_fa_windows.py --in_root "$(WIN_CAUC_EVAL)" --out_root "$(FA_WIN_CAUC)" --split val --mode "$(strip $(FA_MODE))" --only_neg_videos "$(ONLY_NEG_VIDEOS)"
check-windows-muvim:
	$(RUN) windows/check_windows.py --root "$(WIN_MUVIM)"

# -------------------------
# Pipeline (with extract)
# -------------------------
.PHONY: pipeline-data-le2i pipeline-data-urfd pipeline-data-caucafall pipeline-data-muvim
pipeline-data-le2i:
	$(MAKE) preprocess-le2i
	$(MAKE) windows-le2i
pipeline-data-urfd:
	$(MAKE) preprocess-urfd
	$(MAKE) windows-urfd
pipeline-data-caucafall:
	$(MAKE) preprocess-caucafall
	$(MAKE) windows-caucafall
pipeline-data-muvim:
	$(MAKE) preprocess-muvim
	$(MAKE) windows-muvim

# Backward-compatible aliases (data prep only)
# These used to be named pipeline-<ds>. The arch-specific pipelines are now pipeline-<ds> (TCN) and pipeline-<ds>-gcn.
pipeline-le2i-data:
	$(MAKE) pipeline-data-le2i
pipeline-urfd-data:
	$(MAKE) pipeline-data-urfd
pipeline-caucafall-data:
	$(MAKE) pipeline-data-caucafall
pipeline-muvim-data:
	$(MAKE) pipeline-data-muvim


# -------------------------
# Pipeline (NO extract)
# -------------------------
.PHONY: pipeline-le2i-noextract pipeline-urfd-noextract pipeline-caucafall-noextract pipeline-muvim-noextract

pipeline-le2i-noextract:
	$(MAKE) preprocess-le2i-only
	$(MAKE) windows-le2i

pipeline-urfd-noextract:
	$(MAKE) preprocess-urfd-only
	$(MAKE) windows-urfd

pipeline-caucafall-noextract:
	$(MAKE) preprocess-caucafall-only
	$(MAKE) windows-caucafall

pipeline-muvim-noextract:
	$(MAKE) preprocess-muvim-only
	$(MAKE) windows-muvim

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
	  --thr_step "$(GCN_THR_STEP_MUVIM)" \
	  --monitor "$(GCN_MONITOR_MUVIM)" \
	  --dropout "$(GCN_DROPOUT_MUVIM)" \
	  --pos_weight "$(GCN_POS_WEIGHT_MUVIM)" $(GCN_BALANCED_SAMPLER_FLAG_MUVIM) \
	  --save_dir "$(OUT_GCN_MUVIM)$(OUT_TAG)"


# (removed) old pipeline-all-gcn definition; see unified one near the end of the file

pipeline-all-gcn-noextract:
	$(MAKE) pipeline-le2i-noextract
	$(MAKE) pipeline-urfd-noextract
	$(MAKE) pipeline-caucafall-noextract
	$(MAKE) pipeline-muvim-noextract
# -------------------------
# Feature config (shared) — training only
# (Saved into checkpoint bundle; eval scripts load from ckpt.)
# -------------------------
FEAT_USE_MOTION ?= 1
FEAT_USE_CONF_CHANNEL ?= 1
FEAT_USE_BONE ?= 1
FEAT_USE_BONE_LEN ?= 1
# New GCN trainer expects these:
FEAT_USE_ANGLES ?= 0
FEAT_INCLUDE_CENTERED ?= 1
FEAT_INCLUDE_ABS ?= 1
FEAT_INCLUDE_VEL ?= 1
FEAT_MOTION_SCALE_BY_FPS ?= 1
FEAT_CONF_GATE ?= 0.20
FEAT_USE_PRECOMPUTED_MASK ?= 1

# -------------------------
# Feature flags (trainer-specific)
# -------------------------

# TCN trainer (old flag names)
FEAT_FLAGS_TCN = \
  --center "$(CENTER)" \
  --use_motion "$(FEAT_USE_MOTION)" \
  --use_conf_channel "$(FEAT_USE_CONF_CHANNEL)" \
  --use_bone "$(FEAT_USE_BONE)" \
  --use_bone_length "$(FEAT_USE_BONE_LEN)" \
  --motion_scale_by_fps "$(FEAT_MOTION_SCALE_BY_FPS)" \
  --conf_gate "$(FEAT_CONF_GATE)" \
  --use_precomputed_mask "$(FEAT_USE_PRECOMPUTED_MASK)"

# GCN trainer (new flag names from your train_gcn.py usage)
FEAT_FLAGS_GCN = \
  --use_motion "$(FEAT_USE_MOTION)" \
  --use_conf "$(FEAT_USE_CONF_CHANNEL)" \
  --use_bone "$(FEAT_USE_BONE)" \
  --use_bonelen "$(FEAT_USE_BONE_LEN)" \
  --use_angles "$(FEAT_USE_ANGLES)" \
  --normalize "$(NORM_MODE)" \
  --include_centered "$(FEAT_INCLUDE_CENTERED)" \
  --include_abs "$(FEAT_INCLUDE_ABS)" \
  --include_vel "$(FEAT_INCLUDE_VEL)"


# TCN flags
TCN_BALANCED_SAMPLER_FLAG = $(if $(filter 1,$(TCN_BALANCED_SAMPLER)),--balanced_sampler,)
TCN_BALANCED_SAMPLER_FLAG_MUVIM = $(if $(filter 1,$(TCN_BALANCED_SAMPLER_MUVIM)),--balanced_sampler,)
TCN_BASE_FLAGS = \
  $(FEAT_FLAGS_TCN) \
  --loss "$(TCN_LOSS)" --focal_alpha "$(TCN_FOCAL_ALPHA)" --focal_gamma "$(TCN_FOCAL_GAMMA)" \
  --hidden "$(TCN_HIDDEN)" --num_blocks "$(TCN_NUM_BLOCKS)" --kernel "$(TCN_KERNEL)" \
  --grad_clip "$(TCN_GRAD_CLIP)" --patience "$(TCN_PATIENCE)" \
  --thr_min "$(TCN_THR_MIN)" --thr_max "$(TCN_THR_MAX)" --thr_step "$(TCN_THR_STEP)"

# GCN flags
GCN_MODEL_FLAGS = \
  $(FEAT_FLAGS_GCN) \
  --loss "$(GCN_LOSS)" --focal_alpha "$(GCN_FOCAL_ALPHA)" --focal_gamma "$(GCN_FOCAL_GAMMA)" \
  --hidden "$(GCN_HIDDEN)" \
  --num_blocks "$(GCN_NUM_BLOCKS)" --temporal_kernel "$(GCN_TEMPORAL_KERNEL)" --base_channels "$(GCN_BASE_CHANNELS)" \
  --two_stream "$(GCN_TWO_STREAM)" --fuse "$(GCN_FUSE)" \
  --grad_clip "$(GCN_GRAD_CLIP)" --patience "$(GCN_PATIENCE)" --min_epochs "$(GCN_MIN_EPOCHS)" \
  --mask_joint_p "$(MASK_JOINT_P)" --mask_frame_p "$(MASK_FRAME_P)" \
  --weight_decay "$(GCN_WEIGHT_DECAY)" --label_smoothing "$(GCN_LABEL_SMOOTHING)" --num_workers "$(GCN_NUM_WORKERS)" \
  --thr_min "$(GCN_THR_MIN)" --thr_max "$(GCN_THR_MAX)" --thr_step "$(GCN_THR_STEP)"


# ----------------------
# Fit OPs (TCN)
# ----------------------
# ----------------------
# Runtime feature flags (must match training)
# ----------------------
# If your TCN was trained without motion/conf, keep these at 0.
TCN_USE_MOTION ?= $(FEAT_USE_MOTION)
TCN_USE_CONF_CHANNEL ?= $(FEAT_USE_CONF_CHANNEL)
TCN_USE_BONE ?= $(FEAT_USE_BONE)
TCN_USE_BONE_LEN ?= $(FEAT_USE_BONE_LEN)
TCN_MOTION_SCALE_BY_FPS ?= $(FEAT_MOTION_SCALE_BY_FPS)
TCN_CONF_GATE ?= $(FEAT_CONF_GATE)
TCN_USE_PRECOMPUTED_MASK ?= $(FEAT_USE_PRECOMPUTED_MASK)
# Your current GCN training command uses these as 1.
GCN_USE_MOTION ?= $(FEAT_USE_MOTION)
GCN_USE_CONF_CHANNEL ?= $(FEAT_USE_CONF_CHANNEL)
GCN_USE_BONE ?= $(FEAT_USE_BONE)
GCN_USE_BONE_LEN ?= $(FEAT_USE_BONE_LEN)
GCN_MOTION_SCALE_BY_FPS ?= $(FEAT_MOTION_SCALE_BY_FPS)
GCN_CONF_GATE ?= $(FEAT_CONF_GATE)
GCN_USE_PRECOMPUTED_MASK ?= $(FEAT_USE_PRECOMPUTED_MASK)
FIT_FLAGS_TCN = --center "$(CENTER)" --use_motion "$(TCN_USE_MOTION)" --use_conf_channel "$(TCN_USE_CONF_CHANNEL)" --use_bone "$(TCN_USE_BONE)" --use_bone_length "$(TCN_USE_BONE_LEN)" --motion_scale_by_fps "$(TCN_MOTION_SCALE_BY_FPS)" --conf_gate "$(TCN_CONF_GATE)" --use_precomputed_mask "$(TCN_USE_PRECOMPUTED_MASK)"
FIT_FLAGS_GCN = --center "$(CENTER)" --use_motion "$(GCN_USE_MOTION)" --use_conf_channel "$(GCN_USE_CONF_CHANNEL)" --use_bone "$(GCN_USE_BONE)" --use_bone_length "$(GCN_USE_BONE_LEN)" --motion_scale_by_fps "$(GCN_MOTION_SCALE_BY_FPS)" --conf_gate "$(GCN_CONF_GATE)" --use_precomputed_mask "$(GCN_USE_PRECOMPUTED_MASK)"

# Training flags for GCN (kept consistent with fit/eval feature flags)
SERVER_HOST ?= 127.0.0.1
SERVER_PORT ?= 8000

serve-dev:
	$(RUN) -m uvicorn server.app:app --host "$(SERVER_HOST)" --port "$(SERVER_PORT)" --reload


# =========================
# Eval / Fit / Plot targets
# =========================

# Separate windowing for evaluation (dense "all" windows)
# NOTE: OP fitting + FA/24h metrics need chronological coverage; do NOT use balanced sampling for eval.
W := $(WIN_W)
S := $(WIN_S)

WIN_STRATEGY_EVAL ?= all
# Keep eval windows dense and deterministic; use precomputed masks if available.
WIN_EVAL_EXTRA ?= --strategy "$(WIN_STRATEGY_EVAL)" --min_overlap_frames 1 --min_valid_frac 0.0 --conf_gate "$(CONF_GATE)" --use_precomputed_mask --seed "$(SPLIT_SEED)"

WIN_LE2I_EVAL := $(PROCESSED)/le2i/windows_eval_W$(WIN_W)_S$(WIN_S)
WIN_URFD_EVAL := $(PROCESSED)/urfd/windows_eval_W$(WIN_W)_S$(WIN_S)
WIN_CAUC_EVAL := $(PROCESSED)/caucafall/windows_eval_W$(WIN_W)_S$(WIN_S)
WIN_MUVIM_EVAL := $(PROCESSED)/muvim/windows_eval_W$(WIN_W)_S$(WIN_S)

.PHONY: windows-eval-le2i windows-eval-urfd windows-eval-caucafall windows-eval-muvim

windows-eval-le2i:
	@if [ ! -d "$(POSE_LE2I)" ] || [ "$$(find "$(POSE_LE2I)" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)" -eq 0 ]; then \
	  echo "[err] missing or empty pose_npz: $(POSE_LE2I)"; \
	  echo "      Run: make preprocess-le2i-only   (or preprocess-le2i)"; \
	  exit 2; \
	fi
	@if [ ! -f "$(LABELS_LE2I)" ]; then \
	  echo "[err] missing labels: $(LABELS_LE2I)"; \
	  echo "      Run: make labels-le2i"; \
	  exit 2; \
	fi
	@if [ ! -f "$(SPLIT_LE2I_TRAIN)" ] || [ ! -f "$(SPLIT_LE2I_VAL)" ] || [ ! -f "$(SPLIT_LE2I_TEST)" ]; then \
	  echo "[err] missing split lists under configs/splits for le2i"; \
	  echo "      Expected: $(SPLIT_LE2I_TRAIN), $(SPLIT_LE2I_VAL), $(SPLIT_LE2I_TEST)"; \
	  echo "      Run: make splits-le2i"; \
	  exit 2; \
	fi
	@if [ "$(WIN_EVAL_CLEAN)" = "1" ]; then \
	  echo "[clean] removing existing split dirs under $(WIN_LE2I_EVAL)"; \
	  rm -rf "$(WIN_LE2I_EVAL)/train" "$(WIN_LE2I_EVAL)/val" "$(WIN_LE2I_EVAL)/test"; \
	fi
	@mkdir -p "$(WIN_LE2I_EVAL)"
	$(RUN) windows/make_windows.py \
	  --npz_dir "$(POSE_LE2I)" \
	  --labels_json "$(LABELS_LE2I)" \
	  --spans_json  "$(SPANS_LE2I)" \
	  --out_dir "$(WIN_LE2I_EVAL)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_LE2I)" \
	  --train_list "$(SPLIT_LE2I_TRAIN)" --val_list "$(SPLIT_LE2I_VAL)" --test_list "$(SPLIT_LE2I_TEST)" \
	  $(WIN_EVAL_EXTRA) --skip_existing

windows-eval-urfd:
	@if [ ! -d "$(POSE_URFD)" ] || [ "$$(find "$(POSE_URFD)" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)" -eq 0 ]; then \
	  echo "[err] missing or empty pose_npz: $(POSE_URFD)"; \
	  echo "      Run: make preprocess-urfd-only   (or preprocess-urfd)"; \
	  exit 2; \
	fi
	@if [ ! -f "$(LABELS_URFD)" ]; then \
	  echo "[err] missing labels: $(LABELS_URFD)"; \
	  echo "      Run: make labels-urfd"; \
	  exit 2; \
	fi
	@if [ ! -f "$(SPLIT_URFD_TRAIN)" ] || [ ! -f "$(SPLIT_URFD_VAL)" ] || [ ! -f "$(SPLIT_URFD_TEST)" ]; then \
	  echo "[err] missing split lists under configs/splits for urfd"; \
	  echo "      Expected: $(SPLIT_URFD_TRAIN), $(SPLIT_URFD_VAL), $(SPLIT_URFD_TEST)"; \
	  echo "      Run: make splits-urfd"; \
	  exit 2; \
	fi
	@if [ "$(WIN_EVAL_CLEAN)" = "1" ]; then \
	  echo "[clean] removing existing split dirs under $(WIN_URFD_EVAL)"; \
	  rm -rf "$(WIN_URFD_EVAL)/train" "$(WIN_URFD_EVAL)/val" "$(WIN_URFD_EVAL)/test"; \
	fi
	@mkdir -p "$(WIN_URFD_EVAL)"
	$(RUN) windows/make_windows.py \
	  --npz_dir "$(POSE_URFD)" \
	  --labels_json "$(LABELS_URFD)" \
	  --spans_json  "$(SPANS_URFD)" \
	  --out_dir "$(WIN_URFD_EVAL)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_URFD)" \
	  --train_list "$(SPLIT_URFD_TRAIN)" --val_list "$(SPLIT_URFD_VAL)" --test_list "$(SPLIT_URFD_TEST)" \
	  $(WIN_EVAL_EXTRA) --skip_existing

windows-eval-caucafall:
	@if [ ! -d "$(POSE_CAUC)" ] || [ "$$(find "$(POSE_CAUC)" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)" -eq 0 ]; then \
	  echo "[err] missing or empty pose_npz: $(POSE_CAUC)"; \
	  echo "      Run: make preprocess-caucafall-only   (or preprocess-caucafall)"; \
	  exit 2; \
	fi
	@if [ ! -f "$(LABELS_CAUC)" ]; then \
	  echo "[err] missing labels: $(LABELS_CAUC)"; \
	  echo "      Run: make labels-caucafall"; \
	  exit 2; \
	fi
	@if [ ! -f "$(SPLIT_CAUC_TRAIN)" ] || [ ! -f "$(SPLIT_CAUC_VAL)" ] || [ ! -f "$(SPLIT_CAUC_TEST)" ]; then \
	  echo "[err] missing split lists under configs/splits for caucafall"; \
	  echo "      Expected: $(SPLIT_CAUC_TRAIN), $(SPLIT_CAUC_VAL), $(SPLIT_CAUC_TEST)"; \
	  echo "      Run: make splits-caucafall"; \
	  exit 2; \
	fi
	$(MAKE) check-spans-caucafall
	@if [ "$(WIN_EVAL_CLEAN)" = "1" ]; then \
	  echo "[clean] removing existing split dirs under $(WIN_CAUC_EVAL)"; \
	  rm -rf "$(WIN_CAUC_EVAL)/train" "$(WIN_CAUC_EVAL)/val" "$(WIN_CAUC_EVAL)/test"; \
	fi
	@mkdir -p "$(WIN_CAUC_EVAL)"
	$(RUN) windows/make_windows.py \
	  --npz_dir "$(POSE_CAUC)" \
	  --labels_json "$(LABELS_CAUC)" \
	  --spans_json  "$(SPANS_CAUC)" \
	  --require_spans "$(CAUCA_REQUIRE_SPANS)" \
	  --out_dir "$(WIN_CAUC_EVAL)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_CAUC)" \
	  --train_list "$(SPLIT_CAUC_TRAIN)" --val_list "$(SPLIT_CAUC_VAL)" --test_list "$(SPLIT_CAUC_TEST)" \
	  $(WIN_EVAL_EXTRA) --skip_existing

windows-eval-muvim:
	@if [ ! -d "$(POSE_MUVIM)" ] || [ "$$(find "$(POSE_MUVIM)" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)" -eq 0 ]; then \
	  echo "[err] missing or empty pose_npz: $(POSE_MUVIM)"; \
	  echo "      Run: make preprocess-muvim-only   (or preprocess-muvim)"; \
	  exit 2; \
	fi
	@if [ ! -f "$(LABELS_MUVIM)" ]; then \
	  echo "[err] missing labels: $(LABELS_MUVIM)"; \
	  echo "      Run: make labels-muvim"; \
	  exit 2; \
	fi
	@if [ ! -f "$(SPLIT_MUVIM_TRAIN)" ] || [ ! -f "$(SPLIT_MUVIM_VAL)" ] || [ ! -f "$(SPLIT_MUVIM_TEST)" ]; then \
	  echo "[err] missing split lists under configs/splits for muvim"; \
	  echo "      Expected: $(SPLIT_MUVIM_TRAIN), $(SPLIT_MUVIM_VAL), $(SPLIT_MUVIM_TEST)"; \
	  echo "      Run: make splits-muvim"; \
	  exit 2; \
	fi
	@if [ "$(WIN_EVAL_CLEAN)" = "1" ]; then \
	  echo "[clean] removing existing split dirs under $(WIN_MUVIM_EVAL)"; \
	  rm -rf "$(WIN_MUVIM_EVAL)/train" "$(WIN_MUVIM_EVAL)/val" "$(WIN_MUVIM_EVAL)/test"; \
	fi
	@mkdir -p "$(WIN_MUVIM_EVAL)"
	$(RUN) windows/make_windows.py \
	  --npz_dir "$(POSE_MUVIM)" \
	  --labels_json "$(LABELS_MUVIM)" \
	  --spans_json  "$(SPANS_MUVIM)" \
	  --out_dir "$(WIN_MUVIM_EVAL)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_MUVIM)" \
	  --train_list "$(SPLIT_MUVIM_TRAIN)" --val_list "$(SPLIT_MUVIM_VAL)" --test_list "$(SPLIT_MUVIM_TEST)" \
	  $(WIN_EVAL_EXTRA) --skip_existing

CAL_DIR := $(OUT_DIR)/calibration
MET_DIR := $(OUT_DIR)/metrics
PLOT_DIR := $(OUT_DIR)/plots

CAL_TCN_LE2I := $(CAL_DIR)/tcn_le2i.yaml
CAL_TCN_URFD := $(CAL_DIR)/tcn_urfd.yaml
CAL_TCN_CAUC := $(CAL_DIR)/tcn_caucafall.yaml
CAL_TCN_MUVIM := $(CAL_DIR)/tcn_muvim.yaml

CAL_GCN_LE2I := $(CAL_DIR)/gcn_le2i.yaml
CAL_GCN_URFD := $(CAL_DIR)/gcn_urfd.yaml
CAL_GCN_CAUC := $(CAL_DIR)/gcn_caucafall.yaml
CAL_GCN_MUVIM := $(CAL_DIR)/gcn_muvim.yaml



MET_TCN_LE2I := $(MET_DIR)/tcn_le2i.json
MET_TCN_URFD := $(MET_DIR)/tcn_urfd.json
MET_TCN_CAUC := $(MET_DIR)/tcn_caucafall.json
MET_TCN_MUVIM := $(MET_DIR)/tcn_muvim.json

MET_GCN_LE2I := $(MET_DIR)/gcn_le2i.json
MET_GCN_URFD := $(MET_DIR)/gcn_urfd.json
MET_GCN_CAUC := $(MET_DIR)/gcn_caucafall.json
MET_GCN_MUVIM := $(MET_DIR)/gcn_muvim.json

PLOT_TCN_LE2I_RF := $(PLOT_DIR)/tcn_le2i_recall_vs_fa.png
PLOT_TCN_LE2I_F1 := $(PLOT_DIR)/tcn_le2i_f1_vs_tau.png
PLOT_TCN_URFD_RF := $(PLOT_DIR)/tcn_urfd_recall_vs_fa.png
PLOT_TCN_URFD_F1 := $(PLOT_DIR)/tcn_urfd_f1_vs_tau.png
PLOT_TCN_CAUC_RF := $(PLOT_DIR)/tcn_caucafall_recall_vs_fa.png
PLOT_TCN_CAUC_F1 := $(PLOT_DIR)/tcn_caucafall_f1_vs_tau.png
PLOT_TCN_MUVIM_RF := $(PLOT_DIR)/tcn_muvim_recall_vs_fa.png
PLOT_TCN_MUVIM_F1 := $(PLOT_DIR)/tcn_muvim_f1_vs_tau.png

PLOT_GCN_LE2I_RF := $(PLOT_DIR)/gcn_le2i_recall_vs_fa.png
PLOT_GCN_LE2I_F1 := $(PLOT_DIR)/gcn_le2i_f1_vs_tau.png
PLOT_GCN_URFD_RF := $(PLOT_DIR)/gcn_urfd_recall_vs_fa.png
PLOT_GCN_URFD_F1 := $(PLOT_DIR)/gcn_urfd_f1_vs_tau.png
PLOT_GCN_CAUC_RF := $(PLOT_DIR)/gcn_caucafall_recall_vs_fa.png
PLOT_GCN_CAUC_F1 := $(PLOT_DIR)/gcn_caucafall_f1_vs_tau.png
PLOT_GCN_MUVIM_RF := $(PLOT_DIR)/gcn_muvim_recall_vs_fa.png
PLOT_GCN_MUVIM_F1 := $(PLOT_DIR)/gcn_muvim_f1_vs_tau.png


fit-ops-le2i:
	@mkdir -p "$(OPS_DIR)" "$(CAL_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn --val_dir "$(WIN_LE2I_EVAL)/val" --ckpt "$(OUT_TCN_LE2I)/best.pt" --calibration_yaml "$(CAL_TCN_LE2I)" --out "$(OPS_TCN_LE2I)" --fps_default "$(FPS_LE2I)" $(FITOPS_POLICY_FLAGS) $(FITOPS_SWEEP_FLAGS) $(FITOPS_PICKER_FLAGS) --min_tau_high "$(FITOPS_MIN_TAU_HIGH_LE2I)" $(FITOPS_FA_ARG_LE2I)

fit-ops-urfd:
	@mkdir -p "$(OPS_DIR)" "$(CAL_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn --val_dir "$(WIN_URFD_EVAL)/val" --ckpt "$(OUT_TCN_URFD)/best.pt" --calibration_yaml "$(CAL_TCN_URFD)" --out "$(OPS_TCN_URFD)" --fps_default "$(FPS_URFD)" $(FITOPS_POLICY_FLAGS) $(FITOPS_SWEEP_FLAGS) $(FITOPS_PICKER_FLAGS) --min_tau_high "$(FITOPS_MIN_TAU_HIGH_URFD)" $(FITOPS_FA_ARG_URFD)

fit-ops-caucafall:
	@mkdir -p "$(OPS_DIR)" "$(CAL_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn --val_dir "$(WIN_CAUC_EVAL)/val" --ckpt "$(OUT_TCN_CAUC)/best.pt" --calibration_yaml "$(CAL_TCN_CAUC)" --out "$(OPS_TCN_CAUC)" --fps_default "$(FPS_CAUC)" $(FITOPS_POLICY_FLAGS) $(FITOPS_SWEEP_FLAGS) $(FITOPS_PICKER_FLAGS) --min_tau_high "$(FITOPS_MIN_TAU_HIGH_CAUC)" $(FITOPS_FA_ARG_CAUC)

fit-ops-muvim:
	@mkdir -p "$(OPS_DIR)" "$(CAL_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn --val_dir "$(WIN_MUVIM_EVAL)/val" --ckpt "$(OUT_TCN_MUVIM)/best.pt" --calibration_yaml "$(CAL_TCN_MUVIM)" --out "$(OPS_TCN_MUVIM)" --fps_default "$(FPS_MUVIM)" $(FITOPS_POLICY_FLAGS) $(FITOPS_SWEEP_FLAGS) $(FITOPS_PICKER_FLAGS) --min_tau_high "$(FITOPS_MIN_TAU_HIGH_MUVIM)" $(FITOPS_FA_ARG_MUVIM)

fit-ops-gcn-le2i:
	@mkdir -p "$(OPS_DIR)" "$(CAL_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_LE2I_EVAL)/val" --ckpt "$(OUT_GCN_LE2I)/best.pt" --calibration_yaml "$(CAL_GCN_LE2I)" --out "$(OPS_GCN_LE2I)" --fps_default "$(FPS_LE2I)" $(FITOPS_POLICY_FLAGS) $(FITOPS_SWEEP_FLAGS) $(FITOPS_PICKER_FLAGS) --min_tau_high "$(FITOPS_MIN_TAU_HIGH_LE2I)" $(FITOPS_FA_ARG_LE2I)

fit-ops-gcn-urfd:
	@mkdir -p "$(OPS_DIR)" "$(CAL_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_URFD_EVAL)/val" --ckpt "$(OUT_GCN_URFD)/best.pt" --calibration_yaml "$(CAL_GCN_URFD)" --out "$(OPS_GCN_URFD)" --fps_default "$(FPS_URFD)" $(FITOPS_POLICY_FLAGS) $(FITOPS_SWEEP_FLAGS) $(FITOPS_PICKER_FLAGS) --min_tau_high "$(FITOPS_MIN_TAU_HIGH_URFD)" $(FITOPS_FA_ARG_URFD)

fit-ops-gcn-caucafall:
	@mkdir -p "$(OPS_DIR)" "$(CAL_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_CAUC_EVAL)/val" --ckpt "$(OUT_GCN_CAUC)/best.pt" --calibration_yaml "$(CAL_GCN_CAUC)" --out "$(OPS_GCN_CAUC)" --fps_default "$(FPS_CAUC)" $(FITOPS_POLICY_FLAGS) $(FITOPS_SWEEP_FLAGS) $(FITOPS_PICKER_FLAGS) --min_tau_high "$(FITOPS_MIN_TAU_HIGH_CAUC)" $(FITOPS_FA_ARG_CAUC)

fit-ops-gcn-muvim:
	@mkdir -p "$(OPS_DIR)" "$(CAL_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn --val_dir "$(WIN_MUVIM_EVAL)/val" --ckpt "$(OUT_GCN_MUVIM)/best.pt" --calibration_yaml "$(CAL_GCN_MUVIM)" --out "$(OPS_GCN_MUVIM)" --fps_default "$(FPS_MUVIM)" $(FITOPS_POLICY_FLAGS) $(FITOPS_SWEEP_FLAGS) $(FITOPS_PICKER_FLAGS) --min_tau_high "$(FITOPS_MIN_TAU_HIGH_MUVIM)" $(FITOPS_FA_ARG_MUVIM)

eval-le2i:
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/metrics.py --win_dir "$(WIN_LE2I_EVAL)/test" --ckpt "$(OUT_TCN_LE2I)/best.pt" --ops_yaml "$(OPS_TCN_LE2I)" --out_json "$(MET_TCN_LE2I)" --fps_default "$(FPS_LE2I)" $(METRICS_SWEEP_FLAGS)

eval-urfd:
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/metrics.py --win_dir "$(WIN_URFD_EVAL)/test" --ckpt "$(OUT_TCN_URFD)/best.pt" --ops_yaml "$(OPS_TCN_URFD)" --out_json "$(MET_TCN_URFD)" --fps_default "$(FPS_URFD)" $(METRICS_SWEEP_FLAGS)

eval-caucafall:
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/metrics.py --win_dir "$(WIN_CAUC_EVAL)/test" --ckpt "$(OUT_TCN_CAUC)/best.pt" --ops_yaml "$(OPS_TCN_CAUC)" --out_json "$(MET_TCN_CAUC)" --fps_default "$(FPS_CAUC)" $(METRICS_SWEEP_FLAGS)

eval-muvim:
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/metrics.py --win_dir "$(WIN_MUVIM_EVAL)/test" --ckpt "$(OUT_TCN_MUVIM)/best.pt" --ops_yaml "$(OPS_TCN_MUVIM)" --out_json "$(MET_TCN_MUVIM)" --fps_default "$(FPS_MUVIM)" $(METRICS_SWEEP_FLAGS)

eval-gcn-le2i:
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/metrics.py --win_dir "$(WIN_LE2I_EVAL)/test" --ckpt "$(OUT_GCN_LE2I)/best.pt" --ops_yaml "$(OPS_GCN_LE2I)" --out_json "$(MET_GCN_LE2I)" --fps_default "$(FPS_LE2I)" $(METRICS_SWEEP_FLAGS)

eval-gcn-urfd:
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/metrics.py --win_dir "$(WIN_URFD_EVAL)/test" --ckpt "$(OUT_GCN_URFD)/best.pt" --ops_yaml "$(OPS_GCN_URFD)" --out_json "$(MET_GCN_URFD)" --fps_default "$(FPS_URFD)" $(METRICS_SWEEP_FLAGS)

eval-gcn-caucafall:
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/metrics.py --win_dir "$(WIN_CAUC_EVAL)/test" --ckpt "$(OUT_GCN_CAUC)/best.pt" --ops_yaml "$(OPS_GCN_CAUC)" --out_json "$(MET_GCN_CAUC)" --fps_default "$(FPS_CAUC)" $(METRICS_SWEEP_FLAGS)

eval-gcn-muvim:
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/metrics.py --win_dir "$(WIN_MUVIM_EVAL)/test" --ckpt "$(OUT_GCN_MUVIM)/best.pt" --ops_yaml "$(OPS_GCN_MUVIM)" --out_json "$(MET_GCN_MUVIM)" --fps_default "$(FPS_MUVIM)" $(METRICS_SWEEP_FLAGS)

plot-le2i:
	@mkdir -p "$(PLOT_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(MET_TCN_LE2I)" --out_fig "$(PLOT_TCN_LE2I_RF)"
	$(RUN) eval/plot_f1_vs_tau.py --reports "$(MET_TCN_LE2I)" --out_fig "$(PLOT_TCN_LE2I_F1)"

plot-urfd:
	@mkdir -p "$(PLOT_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(MET_TCN_URFD)" --out_fig "$(PLOT_TCN_URFD_RF)"
	$(RUN) eval/plot_f1_vs_tau.py --reports "$(MET_TCN_URFD)" --out_fig "$(PLOT_TCN_URFD_F1)"

plot-caucafall:
	@mkdir -p "$(PLOT_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(MET_TCN_CAUC)" --out_fig "$(PLOT_TCN_CAUC_RF)"
	$(RUN) eval/plot_f1_vs_tau.py --reports "$(MET_TCN_CAUC)" --out_fig "$(PLOT_TCN_CAUC_F1)"

plot-muvim:
	@mkdir -p "$(PLOT_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(MET_TCN_MUVIM)" --out_fig "$(PLOT_TCN_MUVIM_RF)"
	$(RUN) eval/plot_f1_vs_tau.py --reports "$(MET_TCN_MUVIM)" --out_fig "$(PLOT_TCN_MUVIM_F1)"

plot-gcn-le2i:
	@mkdir -p "$(PLOT_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(MET_GCN_LE2I)" --out_fig "$(PLOT_GCN_LE2I_RF)"
	$(RUN) eval/plot_f1_vs_tau.py --reports "$(MET_GCN_LE2I)" --out_fig "$(PLOT_GCN_LE2I_F1)"

plot-gcn-urfd:
	@mkdir -p "$(PLOT_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(MET_GCN_URFD)" --out_fig "$(PLOT_GCN_URFD_RF)"
	$(RUN) eval/plot_f1_vs_tau.py --reports "$(MET_GCN_URFD)" --out_fig "$(PLOT_GCN_URFD_F1)"

plot-gcn-caucafall:
	@mkdir -p "$(PLOT_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(MET_GCN_CAUC)" --out_fig "$(PLOT_GCN_CAUC_RF)"
	$(RUN) eval/plot_f1_vs_tau.py --reports "$(MET_GCN_CAUC)" --out_fig "$(PLOT_GCN_CAUC_F1)"

plot-gcn-muvim:
	@mkdir -p "$(PLOT_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(MET_GCN_MUVIM)" --out_fig "$(PLOT_GCN_MUVIM_RF)"
	$(RUN) eval/plot_f1_vs_tau.py --reports "$(MET_GCN_MUVIM)" --out_fig "$(PLOT_GCN_MUVIM_F1)"

.PHONY: eval-all plot-all eval-all-gcn plot-all-gcn \
	pipeline-le2i pipeline-urfd pipeline-caucafall pipeline-muvim pipeline-all \
	pipeline-gcn-le2i pipeline-gcn-urfd pipeline-gcn-caucafall pipeline-gcn-muvim pipeline-all-gcn

eval-all:
	$(MAKE) eval-le2i
	$(MAKE) eval-caucafall
	$(MAKE) eval-urfd
	$(MAKE) eval-muvim
plot-all:
	$(MAKE) plot-le2i
	$(MAKE) plot-caucafall
	$(MAKE) plot-urfd
	$(MAKE) plot-muvim

eval-all-gcn:
	$(MAKE) eval-gcn-le2i
	$(MAKE) eval-gcn-caucafall
	$(MAKE) eval-gcn-urfd
	$(MAKE) eval-gcn-muvim
plot-all-gcn:
	$(MAKE) plot-gcn-le2i
	$(MAKE) plot-gcn-caucafall
	$(MAKE) plot-gcn-urfd
	$(MAKE) plot-gcn-muvim

pipeline-le2i:
	$(MAKE) train-tcn-le2i
	$(MAKE) windows-eval-le2i
	$(MAKE) fit-ops-le2i
	$(MAKE) eval-le2i
	$(MAKE) plot-le2i
pipeline-urfd:
	$(MAKE) train-tcn-urfd
	$(MAKE) windows-eval-urfd
	$(MAKE) fit-ops-urfd
	$(MAKE) eval-urfd
	$(MAKE) plot-urfd
pipeline-caucafall:
	$(MAKE) train-tcn-caucafall
	$(MAKE) windows-eval-caucafall
	$(MAKE) fit-ops-caucafall
	$(MAKE) eval-caucafall
	$(MAKE) plot-caucafall
pipeline-muvim:
	$(MAKE) train-tcn-muvim
	$(MAKE) windows-eval-muvim
	$(MAKE) fit-ops-muvim
	$(MAKE) eval-muvim
	$(MAKE) plot-muvim
pipeline-all:
	$(MAKE) pipeline-le2i
	$(MAKE) pipeline-caucafall
	$(MAKE) pipeline-urfd
	$(MAKE) pipeline-muvim

pipeline-gcn-le2i:
	$(MAKE) train-gcn-le2i
	$(MAKE) windows-eval-le2i
	$(MAKE) fit-ops-gcn-le2i
	$(MAKE) eval-gcn-le2i
	$(MAKE) plot-gcn-le2i
pipeline-gcn-urfd:
	$(MAKE) train-gcn-urfd
	$(MAKE) windows-eval-urfd
	$(MAKE) fit-ops-gcn-urfd
	$(MAKE) eval-gcn-urfd
	$(MAKE) plot-gcn-urfd
pipeline-gcn-caucafall:
	$(MAKE) train-gcn-caucafall
	$(MAKE) windows-eval-caucafall
	$(MAKE) fit-ops-gcn-caucafall
	$(MAKE) eval-gcn-caucafall
	$(MAKE) plot-gcn-caucafall
pipeline-gcn-muvim:
	$(MAKE) train-gcn-muvim
	$(MAKE) windows-eval-muvim
	$(MAKE) fit-ops-gcn-muvim
	$(MAKE) eval-gcn-muvim
	$(MAKE) plot-gcn-muvim
pipeline-all-gcn:
	$(MAKE) pipeline-gcn-le2i
	$(MAKE) pipeline-gcn-caucafall
	$(MAKE) pipeline-gcn-urfd
	$(MAKE) pipeline-gcn-muvim