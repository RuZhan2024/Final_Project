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
RUN  := $(VENV) && $(PY)

# -------------------------
# Global knobs
# -------------------------
SPLIT_SEED ?= 33724876
TRAIN_FRAC ?= 0.80
VAL_FRAC   ?= 0.10
TEST_FRAC  ?= 0.10

# Windows (default: W32 S8)
WIN_W ?= 32
WIN_S ?= 8

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
GCN_MONITOR ?= F1
GCN_LABEL_SMOOTHING ?= 0.0
GCN_NUM_WORKERS ?= 0


# Threshold sweep (used inside train_gcn.py for val metric selection)
GCN_THR_MIN ?= 0.05
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
GCN_MONITOR_MUVIM ?= AP

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

OUT_GCN_LE2I  := $(OUT_DIR)/le2i_gcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)
OUT_GCN_URFD  := $(OUT_DIR)/urfd_gcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)
OUT_GCN_CAUC  := $(OUT_DIR)/caucafall_gcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)
OUT_GCN_MUVIM := $(OUT_DIR)/muvim_gcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)
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

CENTER ?= pelvis
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
	@echo "  make STRICT=0 fit-ops-gcn-le2i"
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

CAUCA_ANN_GLOB='data/raw/CAUCAFall/**/*.txt'
#   CAUCA_FALL_CLASS_ID=0 or 1  (depends on dataset class mapping)
CAUCA_ANN_GLOB ?=
CAUCA_FALL_CLASS_ID ?= 0
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
	  ANN="--ann_glob \"$(URFD_ANN_GLOB)\" --fall_class_id $(URFD_FALL_CLASS_ID) --min_run $(URFD_MIN_RUN) --gap_fill $(URFD_GAP_FILL)"; \
	fi; \
	eval "$(RUN) labels/make_urfd_labels.py --npz_dir \"$(POSE_URFD)\" --out_labels \"$(LABELS_URFD)\" --out_spans \"$(SPANS_URFD)\" $$ANN"

labels-caucafall: preprocess-caucafall-only
	@mkdir -p "$(LABELS_DIR)"
	@ANN=""; \
	if [ -n "$(CAUCA_ANN_GLOB)" ]; then \
	  ANN="--ann_glob \"$(CAUCA_ANN_GLOB)\" --fall_class_id $(CAUCA_FALL_CLASS_ID) --min_run $(CAUCA_MIN_RUN) --gap_fill $(CAUCA_GAP_FILL)"; \
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
	  --epochs "$(EPOCHS)" --batch "$(BATCH)" --lr "$(LR)" --seed "$(SPLIT_SEED)" \
	  --save_dir "$(OUT_TCN_LE2I)"

train-tcn-urfd:
	# [check] train-tcn-urfd
	if [ ! -d "$(WIN_URFD)/train" ] || [ ! -d "$(WIN_URFD)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_URFD)"; \
	  echo "      Run: make windows-urfd   (or pipeline-urfd-noextract)"; \
	  exit 2; \
	fi
	$(RUN) models/train_tcn.py --train_dir "$(WIN_URFD)/train" --val_dir "$(WIN_URFD)/val" \
	  --epochs "$(EPOCHS)" --batch "$(BATCH)" --lr "$(LR)" --seed "$(SPLIT_SEED)" \
	  --save_dir "$(OUT_TCN_URFD)"

train-tcn-caucafall:
	# [check] train-tcn-caucafall
	if [ ! -d "$(WIN_CAUC)/train" ] || [ ! -d "$(WIN_CAUC)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_CAUC)"; \
	  echo "      Run: make windows-caucafall   (or pipeline-caucafall-noextract)"; \
	  exit 2; \
	fi
	$(RUN) models/train_tcn.py --train_dir "$(WIN_CAUC)/train" --val_dir "$(WIN_CAUC)/val" \
	  --epochs "$(EPOCHS)" --batch "$(BATCH)" --lr "$(LR)" --seed "$(SPLIT_SEED)" \
	  --save_dir "$(OUT_TCN_CAUC)"

train-tcn-muvim:
	# [check] train-tcn-muvim
	if [ ! -d "$(WIN_MUVIM)/train" ] || [ ! -d "$(WIN_MUVIM)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_MUVIM)"; \
	  echo "      Run: make windows-muvim   (or pipeline-muvim-noextract)"; \
	  exit 2; \
	fi
	$(RUN) models/train_tcn.py --train_dir "$(WIN_MUVIM)/train" --val_dir "$(WIN_MUVIM)/val" \
	  --epochs "$(EPOCHS)" --batch "$(BATCH)" --lr "$(LR)" --seed "$(SPLIT_SEED)" \
	  --save_dir "$(OUT_TCN_MUVIM)"

train-gcn-le2i:
	# [check] train-gcn-le2i
	if [ ! -d "$(WIN_LE2I)/train" ] || [ ! -d "$(WIN_LE2I)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_LE2I)"; \
	  echo "      Run: make windows-le2i   (or pipeline-le2i-noextract)"; \
	  exit 2; \
	fi
	@mkdir -p "outputs/reports"
	$(RUN) models/train_gcn.py --train_dir "$(WIN_LE2I)/train" --val_dir "$(WIN_LE2I)/val" \
	  --epochs "$(EPOCHS_GCN)" --batch "$(BATCH_GCN)" --lr "$(LR_GCN_LE2I)" --seed "$(SPLIT_SEED)" \
	  --fps_default "$(FPS_LE2I)" \
	  $(TRAIN_FLAGS_GCN) \
	  --monitor "$(GCN_MONITOR_LE2I)" \
	  --thr_min "$(GCN_THR_MIN)" --thr_max "$(GCN_THR_MAX)" --thr_step "$(GCN_THR_STEP)" --min_epochs "$(GCN_MIN_EPOCHS)" \
	  --save_dir "$(OUT_GCN_LE2I)$(OUT_TAG)" \
	  --report_json "outputs/reports/train_le2i_gcn.json" --report_dataset_name le2i

train-gcn-urfd:
	# [check] train-gcn-urfd
	if [ ! -d "$(WIN_URFD)/train" ] || [ ! -d "$(WIN_URFD)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_URFD)"; \
	  echo "      Run: make windows-urfd   (or pipeline-urfd-noextract)"; \
	  exit 2; \
	fi
	@mkdir -p "outputs/reports"
	$(RUN) models/train_gcn.py --train_dir "$(WIN_URFD)/train" --val_dir "$(WIN_URFD)/val" \
	  --epochs "$(EPOCHS_GCN)" --batch "$(BATCH_GCN)" --lr "$(LR_GCN_URFD)" --seed "$(SPLIT_SEED)" \
	  --fps_default "$(FPS_URFD)" \
	  $(TRAIN_FLAGS_GCN) \
	  --monitor "$(GCN_MONITOR_URFD)" \
	  --thr_min "$(GCN_THR_MIN)" --thr_max "$(GCN_THR_MAX)" --thr_step "$(GCN_THR_STEP)" --min_epochs "$(GCN_MIN_EPOCHS)" \
	  --save_dir "$(OUT_GCN_URFD)$(OUT_TAG)" \
	  --report_json "outputs/reports/train_urfd_gcn.json" --report_dataset_name urfd

train-gcn-caucafall:
	# [check] train-gcn-caucafall
	if [ ! -d "$(WIN_CAUC)/train" ] || [ ! -d "$(WIN_CAUC)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_CAUC)"; \
	  echo "      Run: make windows-caucafall   (or pipeline-caucafall-noextract)"; \
	  exit 2; \
	fi
	@mkdir -p "outputs/reports"
	$(RUN) models/train_gcn.py --train_dir "$(WIN_CAUC)/train" --val_dir "$(WIN_CAUC)/val" \
	  --epochs "$(EPOCHS_GCN)" --batch "$(BATCH_GCN)" --lr "$(LR_GCN_CAUC)" --seed "$(SPLIT_SEED)" \
	  --fps_default "$(FPS_CAUC)" \
	  $(TRAIN_FLAGS_GCN) \
	  --monitor "$(GCN_MONITOR_CAUC)" \
	  --thr_min "$(GCN_THR_MIN)" --thr_max "$(GCN_THR_MAX)" --thr_step "$(GCN_THR_STEP)" --min_epochs "$(GCN_MIN_EPOCHS)" \
	  --save_dir "$(OUT_GCN_CAUC)$(OUT_TAG)" \
	  --report_json "outputs/reports/train_caucafall_gcn.json" --report_dataset_name caucafall

train-gcn-muvim:
	# [check] train-gcn-muvim
	if [ ! -d "$(WIN_MUVIM)/train" ] || [ ! -d "$(WIN_MUVIM)/val" ]; then \
	  echo "[err] windows missing under: $(WIN_MUVIM)"; \
	  echo "      Run: make windows-muvim   (or pipeline-muvim-noextract)"; \
	  exit 2; \
	fi
	@mkdir -p "outputs/reports"
	$(RUN) models/train_gcn.py --train_dir "$(WIN_MUVIM)/train" --val_dir "$(WIN_MUVIM)/val" \
	  --epochs "$(EPOCHS_GCN)" --batch "$(BATCH_GCN)" --lr "$(LR_GCN_MUVIM)" --seed "$(SPLIT_SEED)" \
	  --fps_default "$(FPS_MUVIM)" \
	  $(TRAIN_FLAGS_GCN) \
	  --monitor "$(GCN_MONITOR_MUVIM)" \
	  --dropout "$(GCN_DROPOUT_MUVIM)" --mask_joint_p "$(MASK_JOINT_P_MUVIM)" --mask_frame_p "$(MASK_FRAME_P_MUVIM)" \
	  --pos_weight "$(GCN_POS_WEIGHT_MUVIM)" $(GCN_BALANCED_SAMPLER_FLAG_MUVIM) \
	  --thr_min "$(GCN_THR_MIN)" --thr_max "$(GCN_THR_MAX)" --thr_step "$(GCN_THR_STEP_MUVIM)" --min_epochs "$(GCN_MIN_EPOCHS)" \
	  --save_dir "$(OUT_GCN_MUVIM)$(OUT_TAG)" \
	  --report_json "outputs/reports/train_muvim_gcn.json" --report_dataset_name muvim

pipeline-all-gcn: train-gcn-le2i train-gcn-urfd train-gcn-caucafall train-gcn-muvim eval-all-gcn plot-all-gcn

pipeline-all-gcn-noextract:
	$(MAKE) pipeline-le2i-noextract
	$(MAKE) pipeline-urfd-noextract
	$(MAKE) pipeline-caucafall-noextract
	$(MAKE) pipeline-muvim-noextract
	$(RUN) models/train_gcn.py --train_dir "$(WIN_LE2I)/train" --val_dir "$(WIN_LE2I)/val" --epochs "$(EPOCHS)" --batch "$(BATCH)" --lr "$(LR)" --seed "$(SPLIT_SEED)" --save_dir "$(OUT_GCN_LE2I)"
	$(RUN) models/train_gcn.py --train_dir "$(WIN_URFD)/train" --val_dir "$(WIN_URFD)/val" --epochs "$(EPOCHS)" --batch "$(BATCH)" --lr "$(LR)" --seed "$(SPLIT_SEED)" --save_dir "$(OUT_GCN_URFD)"
	$(RUN) models/train_gcn.py --train_dir "$(WIN_CAUC)/train" --val_dir "$(WIN_CAUC)/val" --epochs "$(EPOCHS)" --batch "$(BATCH)" --lr "$(LR)" --seed "$(SPLIT_SEED)" --save_dir "$(OUT_GCN_CAUC)"
	$(RUN) models/train_gcn.py --train_dir "$(WIN_MUVIM)/train" --val_dir "$(WIN_MUVIM)/val" --epochs "$(EPOCHS)" --batch "$(BATCH)" --lr "$(LR)" --seed "$(SPLIT_SEED)" --save_dir "$(OUT_GCN_MUVIM)"

# -------------------------
# Checks
# -------------------------
.PHONY: check-le2i check-urfd check-caucafall check-muvim
check-le2i:
	$(RUN) windows/check_windows.py --root "$(WIN_LE2I)"
check-urfd:
	$(RUN) windows/check_windows.py --root "$(WIN_URFD)"
check-caucafall:
	$(RUN) windows/check_windows.py --root "$(WIN_CAUC)"
check-muvim:
	$(RUN) windows/check_windows.py --root "$(WIN_MUVIM)"

# -------------------------
# Clean
# -------------------------
.PHONY: clean clean-outputs clean-reports clean-figs clean-windows
clean: clean-outputs

clean-outputs: clean-reports clean-figs

clean-reports:
	rm -rf "$(REPORTS_DIR)"

clean-figs:
	rm -rf "$(FIG_DIR)"

clean-windows:
	rm -rf "$(WIN_LE2I)" "$(WIN_URFD)" "$(WIN_CAUC)" "$(WIN_MUVIM)"


# =========================================================
# 8) EVAL + PLOT (uses eval/metrics.py + eval/plot_fa_recall.py)
# =========================================================


# If your GCN ckpt loading still complains about missing/unexpected keys,
# run with: make STRICT=0 fit-ops-gcn-le2i
STRICT ?= 1

# ----------------------
# Fit OPs (TCN)
# ----------------------
# ----------------------
# Runtime feature flags (must match training)
# ----------------------
CENTER ?= pelvis

# If your TCN was trained without motion/conf, keep these at 0.
TCN_USE_MOTION ?= 0
TCN_USE_CONF_CHANNEL ?= 0
TCN_MOTION_SCALE_BY_FPS ?= 0
TCN_CONF_GATE ?= 0.20
TCN_USE_PRECOMPUTED_MASK ?= 1

# Your current GCN training command uses these as 1.
GCN_USE_MOTION ?= 1
GCN_USE_CONF_CHANNEL ?= 1
GCN_MOTION_SCALE_BY_FPS ?= 1
GCN_CONF_GATE ?= 0.20
GCN_USE_PRECOMPUTED_MASK ?= 1

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
	# [check] train-tcn-le2i
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn $(FIT_FLAGS_TCN) --val_dir "$(WIN_LE2I)/val" --ckpt "$(CKPT_TCN_LE2I)" --out "$(OPS_TCN_LE2I)"

fit-ops-urfd: 
	# [check] train-tcn-urfd
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn $(FIT_FLAGS_TCN) --val_dir "$(WIN_URFD)/val" --ckpt "$(CKPT_TCN_URFD)" --out "$(OPS_TCN_URFD)"

fit-ops-caucafall: 
	# [check] train-tcn-caucafall
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn $(FIT_FLAGS_TCN) --val_dir "$(WIN_CAUC)/val" --ckpt "$(CKPT_TCN_CAUC)" --out "$(OPS_TCN_CAUC)"

fit-ops-muvim: 
	# [check] train-tcn-muvim
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch tcn $(FIT_FLAGS_TCN) --val_dir "$(WIN_MUVIM)/val" --ckpt "$(CKPT_TCN_MUVIM)" --out "$(OPS_TCN_MUVIM)"

# ----------------------
# Fit OPs (GCN)
# ----------------------
.PHONY: fit-ops-gcn-le2i fit-ops-gcn-urfd fit-ops-gcn-caucafall fit-ops-gcn-muvim

fit-ops-gcn-le2i: 
	# [check] train-gcn-le2i
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn $(FIT_FLAGS_GCN) --val_dir "$(WIN_LE2I)/val" --ckpt "$(CKPT_GCN_LE2I)" --out "$(OPS_GCN_LE2I)"

fit-ops-gcn-urfd: train-gcn-urfd
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn $(FIT_FLAGS_GCN) --val_dir "$(WIN_URFD)/val" --ckpt "$(CKPT_GCN_URFD)" --out "$(OPS_GCN_URFD)"

fit-ops-gcn-caucafall: train-gcn-caucafall
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn $(FIT_FLAGS_GCN) --val_dir "$(WIN_CAUC)/val" --ckpt "$(CKPT_GCN_CAUC)" --out "$(OPS_GCN_CAUC)"

fit-ops-gcn-muvim: train-gcn-muvim
	@mkdir -p "$(OPS_DIR)"
	$(RUN) eval/fit_ops.py --arch gcn $(FIT_FLAGS_GCN) --val_dir "$(WIN_MUVIM)/val" --ckpt "$(CKPT_GCN_MUVIM)" --out "$(OPS_GCN_MUVIM)"

# ----------------------
# Eval (TCN) -> writes reports/*.json
# ----------------------
.PHONY: eval-le2i eval-urfd eval-caucafall eval-muvim eval-le2i-on-urfd

eval-le2i: fit-ops-le2i
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --arch tcn $(EVAL_FLAGS_TCN) --eval_dir "$(WIN_LE2I)/test" --ckpt "$(CKPT_TCN_LE2I)" --ops "$(OPS_TCN_LE2I)" --dataset_name le2i --out_json "$(REPORTS_DIR)/le2i_tcn.json"

eval-urfd: fit-ops-urfd
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --arch tcn $(EVAL_FLAGS_TCN) --eval_dir "$(WIN_URFD)/test" --ckpt "$(CKPT_TCN_URFD)" --ops "$(OPS_TCN_URFD)" --dataset_name urfd --out_json "$(REPORTS_DIR)/urfd_tcn.json"

eval-caucafall: fit-ops-caucafall
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --arch tcn $(EVAL_FLAGS_TCN) --eval_dir "$(WIN_CAUC)/test" --ckpt "$(CKPT_TCN_CAUC)" --ops "$(OPS_TCN_CAUC)" --dataset_name caucafall --out_json "$(REPORTS_DIR)/caucafall_tcn.json"

eval-muvim: fit-ops-muvim
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --arch tcn $(EVAL_FLAGS_TCN) --eval_dir "$(WIN_MUVIM)/test" --ckpt "$(CKPT_TCN_MUVIM)" --ops "$(OPS_TCN_MUVIM)" --dataset_name muvim --out_json "$(REPORTS_DIR)/muvim_tcn.json"

# Cross (example): LE2i model evaluated on URFD test
eval-le2i-on-urfd: fit-ops-le2i
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --arch tcn $(EVAL_FLAGS_TCN) --eval_dir "$(WIN_URFD)/test" --ckpt "$(CKPT_TCN_LE2I)" --ops "$(OPS_TCN_LE2I)" --dataset_name le2i_on_urfd --out_json "$(REPORTS_DIR)/le2i_on_urfd_tcn.json"

# ----------------------
# Eval (GCN)
# ----------------------
.PHONY: eval-le2i-gcn eval-urfd-gcn eval-caucafall-gcn eval-muvim-gcn

eval-le2i-gcn: fit-ops-gcn-le2i
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --arch gcn $(EVAL_FLAGS_GCN) --eval_dir "$(WIN_LE2I)/test" --ckpt "$(CKPT_GCN_LE2I)" --ops "$(OPS_GCN_LE2I)" --dataset_name le2i --out_json "$(REPORTS_DIR)/le2i_gcn.json"

eval-urfd-gcn: fit-ops-gcn-urfd
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --arch gcn $(EVAL_FLAGS_GCN) --eval_dir "$(WIN_URFD)/test" --ckpt "$(CKPT_GCN_URFD)" --ops "$(OPS_GCN_URFD)" --dataset_name urfd --out_json "$(REPORTS_DIR)/urfd_gcn.json"

eval-caucafall-gcn: fit-ops-gcn-caucafall
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --arch gcn $(EVAL_FLAGS_GCN) --eval_dir "$(WIN_CAUC)/test" --ckpt "$(CKPT_GCN_CAUC)" --ops "$(OPS_GCN_CAUC)" --dataset_name caucafall --out_json "$(REPORTS_DIR)/caucafall_gcn.json"

eval-muvim-gcn: fit-ops-gcn-muvim
	@mkdir -p "$(REPORTS_DIR)"
	$(RUN) eval/metrics.py --arch gcn $(EVAL_FLAGS_GCN) --eval_dir "$(WIN_MUVIM)/test" --ckpt "$(CKPT_GCN_MUVIM)" --ops "$(OPS_GCN_MUVIM)" --dataset_name muvim --out_json "$(REPORTS_DIR)/muvim_gcn.json"

# Plot FA/24h vs Recall (from report json)
# ----------------------
plot-le2i:
	# [check] plot-le2i
	if [ ! -f "$(REPORTS_DIR)/le2i_tcn.json" ]; then \
	  echo "[err] missing report: $(REPORTS_DIR)/le2i_tcn.json"; \
	  echo "      Run: make eval-le2i"; \
	  exit 2; \
	fi
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/le2i_tcn.json" --title "LE2i (TCN) FA/24h vs Recall" --subtitle "W$(WIN_W) S$(WIN_S)" --out_fig "$(FIG_DIR)/le2i_tcn_fa_recall_W$(WIN_W)_S$(WIN_S).png"

plot-urfd:
	# [check] plot-urfd
	if [ ! -f "$(REPORTS_DIR)/urfd_tcn.json" ]; then \
	  echo "[err] missing report: $(REPORTS_DIR)/urfd_tcn.json"; \
	  echo "      Run: make eval-urfd"; \
	  exit 2; \
	fi
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/urfd_tcn.json" --title "URFD (TCN) FA/24h vs Recall" --subtitle "W$(WIN_W) S$(WIN_S)" --out_fig "$(FIG_DIR)/urfd_tcn_fa_recall_W$(WIN_W)_S$(WIN_S).png"

plot-caucafall:
	# [check] plot-caucafall
	if [ ! -f "$(REPORTS_DIR)/caucafall_tcn.json" ]; then \
	  echo "[err] missing report: $(REPORTS_DIR)/caucafall_tcn.json"; \
	  echo "      Run: make eval-caucafall"; \
	  exit 2; \
	fi
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/caucafall_tcn.json" --title "CAUCAFall (TCN) FA/24h vs Recall" --subtitle "W$(WIN_W) S$(WIN_S)" --out_fig "$(FIG_DIR)/caucafall_tcn_fa_recall_W$(WIN_W)_S$(WIN_S).png"

plot-muvim:
	# [check] plot-muvim
	if [ ! -f "$(REPORTS_DIR)/muvim_tcn.json" ]; then \
	  echo "[err] missing report: $(REPORTS_DIR)/muvim_tcn.json"; \
	  echo "      Run: make eval-muvim"; \
	  exit 2; \
	fi
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/muvim_tcn.json" --title "MUVIM (TCN) FA/24h vs Recall" --subtitle "W$(WIN_W) S$(WIN_S)" --out_fig "$(FIG_DIR)/muvim_tcn_fa_recall_W$(WIN_W)_S$(WIN_S).png"

plot-le2i-on-urfd: eval-le2i-on-urfd
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/le2i_on_urfd_tcn.json" --title "LE2i→URFD (TCN) FA/24h vs Recall" --subtitle "Cross-dataset" --out_fig "$(FIG_DIR)/le2i_on_urfd_tcn_fa_recall_W$(WIN_W)_S$(WIN_S).png"

# GCN plots
.PHONY: plot-le2i-gcn plot-urfd-gcn plot-caucafall-gcn plot-muvim-gcn

plot-le2i-gcn:
	# [check] plot-le2i-gcn
	if [ ! -f "$(REPORTS_DIR)/le2i_gcn.json" ]; then \
	  echo "[err] missing report: $(REPORTS_DIR)/le2i_gcn.json"; \
	  echo "      Run: make eval-le2i-gcn"; \
	  exit 2; \
	fi
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/le2i_gcn.json" --title "LE2i (GCN) FA/24h vs Recall" --subtitle "W$(WIN_W) S$(WIN_S)" --out_fig "$(FIG_DIR)/le2i_gcn_fa_recall_W$(WIN_W)_S$(WIN_S).png"

plot-urfd-gcn:
	# [check] plot-urfd-gcn
	if [ ! -f "$(REPORTS_DIR)/urfd_gcn.json" ]; then \
	  echo "[err] missing report: $(REPORTS_DIR)/urfd_gcn.json"; \
	  echo "      Run: make eval-urfd-gcn"; \
	  exit 2; \
	fi
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/urfd_gcn.json" --title "URFD (GCN) FA/24h vs Recall" --subtitle "W$(WIN_W) S$(WIN_S)" --out_fig "$(FIG_DIR)/urfd_gcn_fa_recall_W$(WIN_W)_S$(WIN_S).png"

plot-caucafall-gcn:
	# [check] plot-caucafall-gcn
	if [ ! -f "$(REPORTS_DIR)/caucafall_gcn.json" ]; then \
	  echo "[err] missing report: $(REPORTS_DIR)/caucafall_gcn.json"; \
	  echo "      Run: make eval-caucafall-gcn"; \
	  exit 2; \
	fi
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/caucafall_gcn.json" --title "CAUCAFall (GCN) FA/24h vs Recall" --subtitle "W$(WIN_W) S$(WIN_S)" --out_fig "$(FIG_DIR)/caucafall_gcn_fa_recall_W$(WIN_W)_S$(WIN_S).png"

plot-muvim-gcn:
	# [check] plot-muvim-gcn
	if [ ! -f "$(REPORTS_DIR)/muvim_gcn.json" ]; then \
	  echo "[err] missing report: $(REPORTS_DIR)/muvim_gcn.json"; \
	  echo "      Run: make eval-muvim-gcn"; \
	  exit 2; \
	fi
	@mkdir -p "$(FIG_DIR)"
	$(RUN) eval/plot_fa_recall.py --reports "$(REPORTS_DIR)/muvim_gcn.json" --title "MUVIM (GCN) FA/24h vs Recall" --subtitle "W$(WIN_W) S$(WIN_S)" --out_fig "$(FIG_DIR)/muvim_gcn_fa_recall_W$(WIN_W)_S$(WIN_S).png"

# Convenience groups
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
