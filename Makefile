# ------------------------------------------------------------
# Makefile — Skeleton-only Fall Detection (ML + Eval + Deploy-sim)
# ------------------------------------------------------------
#
# This Makefile orchestrates your OFFLINE pipeline:
#   raw videos/frames
#     -> pose_npz_raw        (pose extraction)
#     -> pose_npz            (preprocess: resample + One-Euro + masks)
#     -> labels/spans.json   (dataset labeling)
#     -> splits              (train/val/test lists)
#     -> windows_W*_S*       (window NPZ for training/eval)
#     -> train (TCN/GCN)
#     -> calibrate temperature
#     -> fit operating points (OP-1/2/3)
#     -> eval / replay / mining
#
# Optional utilities included:
#   - LE2i unlabeled window generation (FA-only stream)
#   - score FA/day on unlabeled LE2i windows
#   - deploy simulation runner (deploy/run_modes.py)
#
# ------------------------------------------------------------

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Prevent Make from deleting intermediate results when a command fails.
.SECONDARY:
.DELETE_ON_ERROR:

# -------------------------
# Python runner
# -------------------------
PY ?= python3
VENV_ACT ?= .venv/bin/activate

# RUN will activate venv if present and ensure imports work from repo root.
ifneq ("$(wildcard $(VENV_ACT))","")
RUN := source "$(VENV_ACT)" && PYTHONPATH="$(CURDIR)" $(PY) -u
else
RUN := PYTHONPATH="$(CURDIR)" $(PY) -u
endif

# -------------------------
# Datasets supported in this repo
# -------------------------
DATASETS := le2i urfd caucafall muvim

# -------------------------
# Top-level paths
# -------------------------
RAW_DIR    := data/raw
INTERIM    := data/interim
PROCESSED  := data/processed
OUT_DIR    := outputs
CFG_DIR    := configs

LABELS_DIR := $(CFG_DIR)/labels
SPLITS_DIR := $(CFG_DIR)/splits
OPS_DIR    := $(CFG_DIR)/ops

CAL_DIR    := $(OUT_DIR)/calibration
MET_DIR    := $(OUT_DIR)/metrics
MINED_DIR  := $(OUT_DIR)/mined
PLOTS_DIR  := $(OUT_DIR)/plots

# Selected operating point when a script supports it (op1/op2/op3)
OP ?= op2

# -------------------------
# Raw dataset roots (override if your paths differ)
# -------------------------
RAW_le2i      := $(RAW_DIR)/LE2i
RAW_urfd      := $(RAW_DIR)/UR_Fall
RAW_caucafall := $(RAW_DIR)/CAUCAFall
RAW_muvim     := $(RAW_DIR)/MUVIM

# Source FPS defaults (used only when metadata is missing)
FPS_le2i      ?= 25
FPS_urfd      ?= 30
FPS_caucafall ?= 23
FPS_muvim     ?= 30

# -------------------------
# Preprocess knobs (pose_npz_raw -> pose_npz)
# -------------------------
# Confidence gating threshold when constructing masks.
PREPROC_CONF_THR ?= 0.20

# Target deploy FPS (your "single time base").
PREPROC_TARGET_FPS ?= 30

# One-Euro smoothing settings
PREPROC_ONE_EURO ?= 1
PREPROC_ONE_EURO_MIN_CUTOFF ?= 1.0
PREPROC_ONE_EURO_BETA ?= 0.0
PREPROC_ONE_EURO_D_CUTOFF ?= 1.0

# Legacy smoothing (kept for backwards compatibility; usually keep 1 when One-Euro is enabled)
PREPROC_SMOOTH_K ?= 1
PREPROC_MAX_GAP  ?= 4

# -------------------------
# Window builder knobs (pose_npz -> windows)
# -------------------------
WIN_W ?= 48
WIN_S ?= 12

# Extra arguments for windows/make_windows.py
# Keep this as a single variable so you can override it in one place.
WIN_EXTRA ?= --strategy balanced --min_overlap_frames 1 --pos_per_span 80 --neg_ratio 3.0 \
            --max_neg_per_video 250 --max_windows_per_video_no_spans 120 --min_valid_frac 0.0 \
            --skip_existing --write_manifest

# -------------------------
# Unlabeled window pipeline knobs (LE2i only; optional)
# -------------------------
# Scene keywords used by labels/make_unlabeled_test_list.py (case-insensitive match).
LE2I_UNLABELED_SCENES ?= Office "Lecture room"
UNLAB_SUBSET ?= test_unlabeled

UNLAB_MAX_WINDOWS_PER_VIDEO ?= 400
UNLAB_CONF_GATE ?= 0.20
UNLAB_MIN_VALID_FRAC ?= 0.00
UNLAB_MIN_AVG_CONF ?= 0.00
UNLAB_USE_PRECOMPUTED_MASK ?= 1

# -------------------------
# Training knobs
# -------------------------
FORCE ?= 0               # FORCE=1 reruns even if outputs exist
SPLIT_SEED ?= 33724876

EPOCHS_TCN ?= 200
BATCH_TCN  ?= 64
LR_TCN     ?= 0.0005
PATIENCE_TCN ?= 25

EPOCHS_GCN ?= 200
BATCH_GCN  ?= 64
LR_GCN     ?= 0.0005
PATIENCE_GCN ?= 30
MIN_EPOCHS_GCN ?= 20

# ReduceLROnPlateau knobs
LR_PLATEAU_PATIENCE ?= 5
LR_PLATEAU_FACTOR   ?= 0.5
LR_PLATEAU_MIN_LR   ?= 1e-6

# Training objective
MONITOR ?= ap # ap | f1
LOSS ?= bce # bce | focal
FOCAL_ALPHA ?= 0.25
FOCAL_GAMMA ?= 2.0

# Use balanced sampler? (flag only)
BALANCED_SAMPLER ?= 1
BALANCED_SAMPLER_FLAG := $(if $(filter 1 true yes,$(BALANCED_SAMPLER)),--balanced_sampler,)

# EMA for more stable validation
EMA_DECAY  ?= 0.999
PREFER_EMA ?= 1

# Augmentation knobs
MASK_JOINT_P ?= 0.15
MASK_FRAME_P ?= 0.10
AUG_HFLIP_P ?= 0.50
AUG_JITTER_STD ?= 0.008
AUG_JITTER_CONF_SCALED ?= 1
AUG_OCC_P ?= 0.30
AUG_OCC_MIN ?= 3
AUG_OCC_MAX ?= 10
AUG_TIME_SHIFT ?= 1

# -------------------------
# Alert policy knobs for fit_ops (deployment-style)
# -------------------------
CONFIRM ?= 1
QUALITY_ADAPT ?= 1
QUALITY_MIN   ?= 0.25
QUALITY_BOOST ?= 0.15

# Override temperature manually when fitting ops (optional)
# Example: make fit-ops-tcn-le2i TEMP=1.2
TEMP ?=

# Fit-ops FA estimation
#   auto: use $(WIN_ds)/$(UNLAB_SUBSET) if present
#   none: don't pass --fa_dir
#   <path>: explicit folder
FITOPS_FA ?= auto

# Mining knobs
HARDNEG_SPLIT ?= val
NEARMISS_SPLIT ?= val
HARDNEG_MULT ?= 5

# Deploy-sim knobs
DEPLOY_CFG ?= $(CFG_DIR)/deploy_modes.yaml
DEPLOY_SPLIT ?= $(UNLAB_SUBSET)   # prefer unlabeled if available; falls back to test
DEPLOY_TIME_MODE ?= center

# ============================================================
# Derived per-dataset paths
# ============================================================
POSE_RAW_le2i      := $(INTERIM)/le2i/pose_npz_raw
POSE_RAW_urfd      := $(INTERIM)/urfd/pose_npz_raw
POSE_RAW_caucafall := $(INTERIM)/caucafall/pose_npz_raw
POSE_RAW_muvim     := $(INTERIM)/muvim/pose_npz_raw

POSE_le2i      := $(INTERIM)/le2i/pose_npz
POSE_urfd      := $(INTERIM)/urfd/pose_npz
POSE_caucafall := $(INTERIM)/caucafall/pose_npz
POSE_muvim     := $(INTERIM)/muvim/pose_npz

LABELS_le2i      := $(LABELS_DIR)/le2i.json
LABELS_urfd      := $(LABELS_DIR)/urfd.json
LABELS_caucafall := $(LABELS_DIR)/caucafall.json
LABELS_muvim     := $(LABELS_DIR)/muvim.json

SPANS_le2i      := $(LABELS_DIR)/le2i_spans.json
SPANS_urfd      := $(LABELS_DIR)/urfd_spans.json
SPANS_caucafall := $(LABELS_DIR)/caucafall_spans.json
SPANS_muvim     := $(LABELS_DIR)/muvim_spans.json

SPLIT_TRAIN_le2i      := $(SPLITS_DIR)/le2i_train.txt
SPLIT_TRAIN_urfd      := $(SPLITS_DIR)/urfd_train.txt
SPLIT_TRAIN_caucafall := $(SPLITS_DIR)/caucafall_train.txt
SPLIT_TRAIN_muvim     := $(SPLITS_DIR)/muvim_train.txt

SPLIT_VAL_le2i      := $(SPLITS_DIR)/le2i_val.txt
SPLIT_VAL_urfd      := $(SPLITS_DIR)/urfd_val.txt
SPLIT_VAL_caucafall := $(SPLITS_DIR)/caucafall_val.txt
SPLIT_VAL_muvim     := $(SPLITS_DIR)/muvim_val.txt

SPLIT_TEST_le2i      := $(SPLITS_DIR)/le2i_test.txt
SPLIT_TEST_urfd      := $(SPLITS_DIR)/urfd_test.txt
SPLIT_TEST_caucafall := $(SPLITS_DIR)/caucafall_test.txt
SPLIT_TEST_muvim     := $(SPLITS_DIR)/muvim_test.txt

# Optional: list of stems used to build unlabeled windows (LE2i helper)
SPLIT_UNLABELED_le2i := $(SPLITS_DIR)/le2i_unlabeled.txt

WIN_le2i      := $(PROCESSED)/le2i/windows_W$(WIN_W)_S$(WIN_S)
WIN_urfd      := $(PROCESSED)/urfd/windows_W$(WIN_W)_S$(WIN_S)
WIN_caucafall := $(PROCESSED)/caucafall/windows_W$(WIN_W)_S$(WIN_S)
WIN_muvim     := $(PROCESSED)/muvim/windows_W$(WIN_W)_S$(WIN_S)

OUT_TCN_le2i      := $(OUT_DIR)/le2i_tcn_W$(WIN_W)S$(WIN_S)
OUT_TCN_urfd      := $(OUT_DIR)/urfd_tcn_W$(WIN_W)S$(WIN_S)
OUT_TCN_caucafall := $(OUT_DIR)/caucafall_tcn_W$(WIN_W)S$(WIN_S)
OUT_TCN_muvim     := $(OUT_DIR)/muvim_tcn_W$(WIN_W)S$(WIN_S)

OUT_GCN_le2i      := $(OUT_DIR)/le2i_gcn_W$(WIN_W)S$(WIN_S)
OUT_GCN_urfd      := $(OUT_DIR)/urfd_gcn_W$(WIN_W)S$(WIN_S)
OUT_GCN_caucafall := $(OUT_DIR)/caucafall_gcn_W$(WIN_W)S$(WIN_S)
OUT_GCN_muvim     := $(OUT_DIR)/muvim_gcn_W$(WIN_W)S$(WIN_S)

CKPT_TCN_le2i      := $(OUT_TCN_le2i)/best.pt
CKPT_TCN_urfd      := $(OUT_TCN_urfd)/best.pt
CKPT_TCN_caucafall := $(OUT_TCN_caucafall)/best.pt
CKPT_TCN_muvim     := $(OUT_TCN_muvim)/best.pt

CKPT_GCN_le2i      := $(OUT_GCN_le2i)/best.pt
CKPT_GCN_urfd      := $(OUT_GCN_urfd)/best.pt
CKPT_GCN_caucafall := $(OUT_GCN_caucafall)/best.pt
CKPT_GCN_muvim     := $(OUT_GCN_muvim)/best.pt

CAL_TCN_le2i      := $(CAL_DIR)/tcn_le2i.yaml
CAL_TCN_urfd      := $(CAL_DIR)/tcn_urfd.yaml
CAL_TCN_caucafall := $(CAL_DIR)/tcn_caucafall.yaml
CAL_TCN_muvim     := $(CAL_DIR)/tcn_muvim.yaml

CAL_GCN_le2i      := $(CAL_DIR)/gcn_le2i.yaml
CAL_GCN_urfd      := $(CAL_DIR)/gcn_urfd.yaml
CAL_GCN_caucafall := $(CAL_DIR)/gcn_caucafall.yaml
CAL_GCN_muvim     := $(CAL_DIR)/gcn_muvim.yaml

OPS_TCN_le2i      := $(OPS_DIR)/tcn_le2i.yaml
OPS_TCN_urfd      := $(OPS_DIR)/tcn_urfd.yaml
OPS_TCN_caucafall := $(OPS_DIR)/tcn_caucafall.yaml
OPS_TCN_muvim     := $(OPS_DIR)/tcn_muvim.yaml

OPS_GCN_le2i      := $(OPS_DIR)/gcn_le2i.yaml
OPS_GCN_urfd      := $(OPS_DIR)/gcn_urfd.yaml
OPS_GCN_caucafall := $(OPS_DIR)/gcn_caucafall.yaml
OPS_GCN_muvim     := $(OPS_DIR)/gcn_muvim.yaml

# ============================================================
# Help
# ============================================================
.PHONY: help
help:
	@echo "Core pipeline (per dataset):"
	@echo "  make extract-<ds>            # pose from videos (ds: $(DATASETS))"
	@echo "  make extract-img-urfd        # pose from image sequences (URFD) [optional]"
	@echo "  make extract-img-caucafall   # pose from image sequences (CAUCAFall) [optional]"
	@echo "  make preprocess-<ds>         # resample + One-Euro + masks"
	@echo "  make labels-<ds>             # labels.json + spans.json"
	@echo "  make splits-<ds>             # train/val/test lists"
	@echo "  make windows-<ds>            # build windows_W*_S*"
	@echo "  make check-windows-<ds>      # quick schema checks"
	@echo ""
	@echo "Training + eval:"
	@echo "  make train-tcn-<ds> | train-gcn-<ds>"
	@echo "  make calibrate-tcn-<ds> | calibrate-gcn-<ds>"
	@echo "  make fit-ops-tcn-<ds> | fit-ops-gcn-<ds>"
	@echo "  make eval-tcn-<ds> | eval-gcn-<ds>"
	@echo "  make replay-tcn-<ds> | replay-gcn-<ds>"
	@echo ""
	@echo "Unlabeled (FA-only) + deploy-sim (optional):"
	@echo "  make unlabeled-le2i          # create $(UNLAB_SUBSET) windows for LE2i"
	@echo "  make score-unlabeled-tcn-le2i | score-unlabeled-gcn-le2i"
	@echo "  make deploy-tcn-<ds> | deploy-gcn-<ds> | deploy-dual-<ds>"
	@echo ""
	@echo "Mining + finetune:"
	@echo "  make mine-nearmiss-tcn-<ds> | mine-nearmiss-gcn-<ds>"
	@echo "  make mine-hardneg-tcn-<ds>  | mine-hardneg-gcn-<ds>"
	@echo "  make finetune-hardneg-tcn-<ds> | finetune-hardneg-gcn-<ds>"
	@echo ""
	@echo "Plots:"
	@echo "  make plot-ops-tcn-<ds> | plot-ops-gcn-<ds>"
	@echo ""
	@echo "Meta:"
	@echo "  make pipeline-<ds>           # extract->preprocess->labels->splits->windows"
	@echo "  make workflow-tcn-<ds>       # windows->train->calibrate->fit-ops->eval"
	@echo "  make workflow-gcn-<ds>"
	@echo ""
	@echo "Common overrides:"
	@echo "  WIN_W=48 WIN_S=12  LOSS=focal  EMA_DECAY=0.999  QUALITY_ADAPT=1 CONFIRM=1"
	@echo "  FORCE=1 (rerun even if outputs exist)"
	@echo "  FITOPS_FA=auto|none|<path> (use unlabeled windows for FA in fit_ops)"

# ============================================================
# Pose extraction
# ============================================================
.PHONY: extract-le2i extract-urfd extract-caucafall extract-muvim

extract-le2i:
	@mkdir -p "$(POSE_RAW_le2i)"
	$(RUN) pose/extract_2d.py \
	  --videos_glob "$(RAW_le2i)/**/Videos/*.avi" "$(RAW_le2i)/**/*.avi" "$(RAW_le2i)/**/Videos/*.mp4" "$(RAW_le2i)/**/*.mp4" \
	  --out_dir "$(POSE_RAW_le2i)" --fps_default "$(FPS_le2i)" --skip_existing || true

extract-urfd:
	@mkdir -p "$(POSE_RAW_urfd)"
	$(RUN) pose/extract_2d.py \
	  --videos_glob "$(RAW_urfd)/**/*.avi" "$(RAW_urfd)/**/*.mp4" \
	  --out_dir "$(POSE_RAW_urfd)" --fps_default "$(FPS_urfd)" --skip_existing || true

extract-caucafall:
	@mkdir -p "$(POSE_RAW_caucafall)"
	$(RUN) pose/extract_2d.py \
	  --videos_glob "$(RAW_caucafall)/**/*.avi" "$(RAW_caucafall)/**/*.mp4" \
	  --out_dir "$(POSE_RAW_caucafall)" --fps_default "$(FPS_caucafall)" --skip_existing || true

extract-muvim:
	@mkdir -p "$(POSE_RAW_muvim)"
	$(RUN) pose/extract_2d.py \
	  --videos_glob "$(RAW_muvim)/**/*.avi" "$(RAW_muvim)/**/*.mp4" \
	  --out_dir "$(POSE_RAW_muvim)" --fps_default "$(FPS_muvim)" --skip_existing || true

# Alternative extractors for image-sequence datasets (optional)
.PHONY: extract-img-urfd extract-img-caucafall

extract-img-urfd:
	@mkdir -p "$(POSE_RAW_urfd)"
	$(RUN) pose/extract_2d_from_images.py \
	  --dataset_code urfd \
	  --images_glob "$(RAW_urfd)/**/*.*" \
	  --out_dir "$(POSE_RAW_urfd)" --fps_default "$(FPS_urfd)" --skip_existing || true

extract-img-caucafall:
	@mkdir -p "$(POSE_RAW_caucafall)"
	$(RUN) pose/extract_2d_from_images.py \
	  --dataset_code caucafall \
	  --images_glob "$(RAW_caucafall)/**/*.*" \
	  --out_dir "$(POSE_RAW_caucafall)" --fps_default "$(FPS_caucafall)" --skip_existing || true

# ============================================================
# Preprocess (pose_npz_raw -> pose_npz)
# ============================================================
.PHONY: preprocess-% preprocess-only-%

preprocess-%: extract-%
	$(MAKE) preprocess-only-$*

preprocess-only-%:
	@mkdir -p "$(POSE_$*)" "$(POSE_RAW_$*)"
	@if [ ! -d "$(POSE_RAW_$*)" ]; then echo "[err] missing pose raw dir: $(POSE_RAW_$*)"; exit 2; fi
	$(RUN) pose/preprocess_pose_npz.py \
	  --in_dir "$(POSE_RAW_$*)" --out_dir "$(POSE_$*)" \
	  --fps_default "$(FPS_$*)" \
	  --target_fps "$(PREPROC_TARGET_FPS)" \
	  --one_euro "$(PREPROC_ONE_EURO)" \
	  --one_euro_min_cutoff "$(PREPROC_ONE_EURO_MIN_CUTOFF)" \
	  --one_euro_beta "$(PREPROC_ONE_EURO_BETA)" \
	  --one_euro_d_cutoff "$(PREPROC_ONE_EURO_D_CUTOFF)" \
	  --conf_thr "$(PREPROC_CONF_THR)" \
	  --smooth_k "$(PREPROC_SMOOTH_K)" \
	  --max_gap "$(PREPROC_MAX_GAP)" \
	  --skip_existing

# ============================================================
# Labels (pose_npz -> labels.json/spans.json)
# ============================================================
URFD_MIN_RUN ?= 2
URFD_GAP_FILL ?= 2

CAUC_MIN_RUN ?= 2
CAUC_GAP_FILL ?= 2

MUVIM_ZED_CSV ?= $(RAW_muvim)/ZED_RGB/ZED_RGB.csv

.PHONY: labels-le2i labels-urfd labels-caucafall labels-muvim

labels-le2i: preprocess-only-le2i
	@mkdir -p "$(LABELS_DIR)"
	$(RUN) labels/make_le2i_labels.py \
	  --npz_dir "$(POSE_le2i)" --raw_root "$(RAW_le2i)" \
	  --out_labels "$(LABELS_le2i)" --out_spans "$(SPANS_le2i)"

labels-urfd: preprocess-only-urfd
	@mkdir -p "$(LABELS_DIR)"
	$(RUN) labels/make_urfd_labels.py \
	  --raw_root "$(RAW_urfd)" --npz_dir "$(POSE_urfd)" \
	  --out_labels "$(LABELS_urfd)" --out_spans "$(SPANS_urfd)" \
	  --min_run "$(URFD_MIN_RUN)" --gap_fill "$(URFD_GAP_FILL)"

labels-caucafall: preprocess-only-caucafall
	@mkdir -p "$(LABELS_DIR)"
	$(RUN) labels/make_caucafall_labels_from_frames.py \
	  --raw_root "$(RAW_caucafall)" --npz_dir "$(POSE_caucafall)" \
	  --out_labels "$(LABELS_caucafall)" --out_spans "$(SPANS_caucafall)" \
	  --min_run "$(CAUC_MIN_RUN)" --gap_fill "$(CAUC_GAP_FILL)"

labels-muvim: preprocess-only-muvim
	@mkdir -p "$(LABELS_DIR)"
	$(RUN) labels/make_muvim_labels.py \
	  --npz_dir "$(POSE_muvim)" --zed_csv "$(MUVIM_ZED_CSV)" \
	  --out_labels "$(LABELS_muvim)" --out_spans "$(SPANS_muvim)"

# ============================================================
# Splits (labels.json -> train/val/test lists)
# ============================================================
.PHONY: splits-%

splits-%: labels-%
	@mkdir -p "$(SPLITS_DIR)"
	$(RUN) split/make_splits.py \
	  --labels_json "$(LABELS_$*)" \
	  --out_dir "$(SPLITS_DIR)" --prefix "$*" \
	  --seed "$(SPLIT_SEED)"

# ============================================================
# Windows (pose_npz + splits -> windows folder)
# ============================================================
.PHONY: windows-% check-windows-%

windows-%: splits-%
	@mkdir -p "$(WIN_$*)"
	@SP=""; \
	if [ -f "$(SPANS_$*)" ] && [ "$$(wc -c < "$(SPANS_$*)")" -gt 5 ]; then SP="--spans_json \"$(SPANS_$*)\""; fi; \
	eval "$(RUN) windows/make_windows.py \
	  --npz_dir \"$(POSE_$*)\" \
	  --labels_json \"$(LABELS_$*)\" $$SP \
	  --out_dir \"$(WIN_$*)\" \
	  --W \"$(WIN_W)\" --stride \"$(WIN_S)\" \
	  --fps_default \"$(FPS_$*)\" \
	  --train_list \"$(SPLIT_TRAIN_$*)\" \
	  --val_list \"$(SPLIT_VAL_$*)\" \
	  --test_list \"$(SPLIT_TEST_$*)\" \
	  $(WIN_EXTRA)"

check-windows-%:
	@if [ ! -d "$(WIN_$*)" ]; then echo "[err] missing windows: $(WIN_$*) (run make windows-$*)"; exit 2; fi
	$(RUN) windows/check_windows.py --root "$(WIN_$*)"

# ============================================================
# Unlabeled windows (LE2i only; optional)
# ============================================================
.PHONY: unlabeled-le2i unlabeled-list-le2i windows-unlabeled-le2i

# 1) Pick stems by scene keywords
unlabeled-list-le2i: preprocess-only-le2i
	@mkdir -p "$(SPLITS_DIR)"
	$(RUN) labels/make_unlabeled_test_list.py \
	  --npz_dir "$(POSE_le2i)" \
	  --out "$(SPLIT_UNLABELED_le2i)" \
	  --scenes $(LE2I_UNLABELED_SCENES)

# 2) Build unlabeled windows from the stems list
windows-unlabeled-le2i: unlabeled-list-le2i
	@mkdir -p "$(WIN_le2i)/$(UNLAB_SUBSET)"
	@MASKFLAG=""; if [ "$(UNLAB_USE_PRECOMPUTED_MASK)" = "1" ]; then MASKFLAG="--use_precomputed_mask"; fi; \
	$(RUN) windows/make_unlabeled_windows.py \
	  --npz_dir "$(POSE_le2i)" \
	  --stems_txt "$(SPLIT_UNLABELED_le2i)" \
	  --out_dir "$(WIN_le2i)" \
	  --subset "$(UNLAB_SUBSET)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" \
	  --fps_default "$(FPS_le2i)" \
	  --seed "$(SPLIT_SEED)" \
	  --max_windows_per_video "$(UNLAB_MAX_WINDOWS_PER_VIDEO)" \
	  $$MASKFLAG \
	  --conf_gate "$(UNLAB_CONF_GATE)" \
	  --min_valid_frac "$(UNLAB_MIN_VALID_FRAC)" \
	  --min_avg_conf "$(UNLAB_MIN_AVG_CONF)" \
	  --skip_existing

unlabeled-le2i: windows-unlabeled-le2i
	@:

# ============================================================
# Training
# ============================================================
define TRAIN_COMMON_FLAGS
--monitor "$(strip $(MONITOR))" $(BALANCED_SAMPLER_FLAG) \
--loss "$(strip $(LOSS))" --focal_alpha "$(FOCAL_ALPHA)" --focal_gamma "$(FOCAL_GAMMA)" \
--ema_decay "$(EMA_DECAY)" \
--mask_joint_p "$(MASK_JOINT_P)" --mask_frame_p "$(MASK_FRAME_P)" \
--aug_hflip_p "$(AUG_HFLIP_P)" --aug_jitter_std "$(AUG_JITTER_STD)" --aug_jitter_conf_scaled "$(AUG_JITTER_CONF_SCALED)" \
--aug_occ_p "$(AUG_OCC_P)" --aug_occ_min_len "$(AUG_OCC_MIN)" --aug_occ_max_len "$(AUG_OCC_MAX)" \
--aug_time_shift "$(AUG_TIME_SHIFT)"
endef

.PHONY: train-tcn-% train-gcn-%

train-tcn-%: windows-%
	@mkdir -p "$(OUT_TCN_$*)"
	@if [ -f "$(CKPT_TCN_$*)" ] && [ "$(FORCE)" != "1" ]; then echo "[skip] $(CKPT_TCN_$*) exists (FORCE=1 to retrain)"; exit 0; fi
	$(RUN) models/train_tcn.py \
	  --train_dir "$(WIN_$*)/train" --val_dir "$(WIN_$*)/val" \
	  --epochs "$(EPOCHS_TCN)" --patience "$(PATIENCE_TCN)" \
	  --batch "$(BATCH_TCN)" --lr "$(LR_TCN)" \
	  --seed "$(SPLIT_SEED)" \
	  --lr_plateau_patience "$(LR_PLATEAU_PATIENCE)" \
	  --lr_plateau_factor "$(LR_PLATEAU_FACTOR)" \
	  --lr_plateau_min_lr "$(LR_PLATEAU_MIN_LR)" \
	  --fps_default "$(FPS_$*)" \
	  --save_dir "$(OUT_TCN_$*)" \
	  $(TRAIN_COMMON_FLAGS)

train-gcn-%: windows-%
	@mkdir -p "$(OUT_GCN_$*)"
	@if [ -f "$(CKPT_GCN_$*)" ] && [ "$(FORCE)" != "1" ]; then echo "[skip] $(CKPT_GCN_$*) exists (FORCE=1 to retrain)"; exit 0; fi
	$(RUN) models/train_gcn.py \
	  --train_dir "$(WIN_$*)/train" --val_dir "$(WIN_$*)/val" \
	  --epochs "$(EPOCHS_GCN)" --patience "$(PATIENCE_GCN)" --min_epochs "$(MIN_EPOCHS_GCN)" \
	  --batch "$(BATCH_GCN)" --lr "$(LR_GCN)" \
	  --seed "$(SPLIT_SEED)" \
	  --lr_plateau_patience "$(LR_PLATEAU_PATIENCE)" \
	  --lr_plateau_factor "$(LR_PLATEAU_FACTOR)" \
	  --lr_plateau_min_lr "$(LR_PLATEAU_MIN_LR)" \
	  --fps_default "$(FPS_$*)" \
	  --save_dir "$(OUT_GCN_$*)" \
	  $(TRAIN_COMMON_FLAGS)

# ============================================================
# Temperature calibration
# ============================================================
.PHONY: calibrate-tcn-% calibrate-gcn-%

calibrate-tcn-%: train-tcn-%
	@mkdir -p "$(CAL_DIR)"
	$(RUN) eval/calibrate_temperature.py \
	  --arch tcn \
	  --val_dir "$(WIN_$*)/val" \
	  --ckpt "$(CKPT_TCN_$*)" \
	  --out_yaml "$(CAL_TCN_$*)" \
	  --prefer_ema "$(PREFER_EMA)"

calibrate-gcn-%: train-gcn-%
	@mkdir -p "$(CAL_DIR)"
	$(RUN) eval/calibrate_temperature.py \
	  --arch gcn \
	  --val_dir "$(WIN_$*)/val" \
	  --ckpt "$(CKPT_GCN_$*)" \
	  --out_yaml "$(CAL_GCN_$*)" \
	  --prefer_ema "$(PREFER_EMA)"

# ============================================================
# Fit operating points (OP-1/2/3)
# ============================================================
.PHONY: fit-ops-tcn-% fit-ops-gcn-%

fit-ops-tcn-%: calibrate-tcn-%
	@mkdir -p "$(OPS_DIR)"
	@TFLAG=""; if [ -n "$(TEMP)" ]; then TFLAG="--temperature $(TEMP)"; fi; \
	FAFLAG=""; \
	if [ "$(FITOPS_FA)" = "auto" ]; then \
	  if [ -d "$(WIN_$*)/$(UNLAB_SUBSET)" ]; then FAFLAG="--fa_dir \"$(WIN_$*)/$(UNLAB_SUBSET)\""; fi; \
	elif [ "$(FITOPS_FA)" = "none" ]; then \
	  FAFLAG=""; \
	else \
	  FAFLAG="--fa_dir \"$(FITOPS_FA)\""; \
	fi; \
	$(RUN) eval/fit_ops.py \
	  --arch tcn \
	  --val_dir "$(WIN_$*)/val" \
	  --ckpt "$(CKPT_TCN_$*)" \
	  --out "$(OPS_TCN_$*)" \
	  --deploy_fps "$(PREPROC_TARGET_FPS)" --deploy_w "$(WIN_W)" --deploy_s "$(WIN_S)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(CAL_TCN_$*)" $$TFLAG $$FAFLAG \
	  --confirm "$(CONFIRM)" \
	  --quality_adapt "$(QUALITY_ADAPT)" --quality_min "$(QUALITY_MIN)" --quality_boost "$(QUALITY_BOOST)"

fit-ops-gcn-%: calibrate-gcn-%
	@mkdir -p "$(OPS_DIR)"
	@TFLAG=""; if [ -n "$(TEMP)" ]; then TFLAG="--temperature $(TEMP)"; fi; \
	FAFLAG=""; \
	if [ "$(FITOPS_FA)" = "auto" ]; then \
	  if [ -d "$(WIN_$*)/$(UNLAB_SUBSET)" ]; then FAFLAG="--fa_dir \"$(WIN_$*)/$(UNLAB_SUBSET)\""; fi; \
	elif [ "$(FITOPS_FA)" = "none" ]; then \
	  FAFLAG=""; \
	else \
	  FAFLAG="--fa_dir \"$(FITOPS_FA)\""; \
	fi; \
	$(RUN) eval/fit_ops.py \
	  --arch gcn \
	  --val_dir "$(WIN_$*)/val" \
	  --ckpt "$(CKPT_GCN_$*)" \
	  --out "$(OPS_GCN_$*)" \
	  --deploy_fps "$(PREPROC_TARGET_FPS)" --deploy_w "$(WIN_W)" --deploy_s "$(WIN_S)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(CAL_GCN_$*)" $$TFLAG $$FAFLAG \
	  --confirm "$(CONFIRM)" \
	  --quality_adapt "$(QUALITY_ADAPT)" --quality_min "$(QUALITY_MIN)" --quality_boost "$(QUALITY_BOOST)"

# ============================================================
# Evaluation
# ============================================================
.PHONY: eval-tcn-% eval-gcn-% replay-tcn-% replay-gcn-%

eval-tcn-%: fit-ops-tcn-%
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/metrics.py \
	  --arch tcn \
	  --windows_dir "$(WIN_$*)/test" \
	  --ckpt "$(CKPT_TCN_$*)" \
	  --out_json "$(MET_DIR)/tcn_$*.json" \
	  --op "$(OP)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(CAL_TCN_$*)" \
	  --ops_yaml "$(OPS_TCN_$*)"

eval-gcn-%: fit-ops-gcn-%
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/metrics.py \
	  --arch gcn \
	  --windows_dir "$(WIN_$*)/test" \
	  --ckpt "$(CKPT_GCN_$*)" \
	  --out_json "$(MET_DIR)/gcn_$*.json" \
	  --op "$(OP)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(CAL_GCN_$*)" \
	  --ops_yaml "$(OPS_GCN_$*)"

replay-tcn-%: fit-ops-tcn-%
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/replay_eval.py \
	  --arch tcn \
	  --windows_dir "$(WIN_$*)/test" \
	  --ckpt "$(CKPT_TCN_$*)" \
	  --out_json "$(MET_DIR)/replay_tcn_$*.json" \
	  --op "$(OP)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(CAL_TCN_$*)" \
	  --ops_yaml "$(OPS_TCN_$*)"

replay-gcn-%: fit-ops-gcn-%
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/replay_eval.py \
	  --arch gcn \
	  --windows_dir "$(WIN_$*)/test" \
	  --ckpt "$(CKPT_GCN_$*)" \
	  --out_json "$(MET_DIR)/replay_gcn_$*.json" \
	  --op "$(OP)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(CAL_GCN_$*)" \
	  --ops_yaml "$(OPS_GCN_$*)"

# Unlabeled FA scoring (LE2i only)
# --------------------------------
# LE2i is the only dataset in this repo with a built-in helper for selecting
# "unlabeled test scenes" (Office/Lecture room). These windows are useful for:
#   - estimating FA/day on long normal-life footage
#   - providing a realistic FA stream when fitting ops (FITOPS_FA=auto)
.PHONY: score-unlabeled-tcn-le2i score-unlabeled-gcn-le2i

score-unlabeled-tcn-le2i: fit-ops-tcn-le2i unlabeled-le2i
	@mkdir -p "$(MET_DIR)"; \
	$(RUN) eval/score_unlabeled_alert_rate.py \
	  --arch tcn \
	  --windows_dir "$(WIN_le2i)/$(UNLAB_SUBSET)" \
	  --ckpt "$(CKPT_TCN_le2i)" \
	  --out_json "$(MET_DIR)/unlabeled_tcn_le2i.json" \
	  --op "$(OP)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(CAL_TCN_le2i)" \
	  --ops_yaml "$(OPS_TCN_le2i)"

score-unlabeled-gcn-le2i: fit-ops-gcn-le2i unlabeled-le2i
	@mkdir -p "$(MET_DIR)"; \
	$(RUN) eval/score_unlabeled_alert_rate.py \
	  --arch gcn \
	  --windows_dir "$(WIN_le2i)/$(UNLAB_SUBSET)" \
	  --ckpt "$(CKPT_GCN_le2i)" \
	  --out_json "$(MET_DIR)/unlabeled_gcn_le2i.json" \
	  --op "$(OP)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(CAL_GCN_le2i)" \
	  --ops_yaml "$(OPS_GCN_le2i)"

# ============================================================
# Plotting (from ops YAML)
# ============================================================
.PHONY: plot-ops-tcn-% plot-ops-gcn-%

plot-ops-tcn-%: fit-ops-tcn-%
	@mkdir -p "$(PLOTS_DIR)"
	$(RUN) eval/plot_fa_recall.py \
	  --ops_yaml "$(OPS_TCN_$*)" \
	  --out "$(PLOTS_DIR)/tcn_$*_recall_vs_fa.png" \
	  --out_f1 "$(PLOTS_DIR)/tcn_$*_f1_vs_tau.png" \
	  --log_x

plot-ops-gcn-%: fit-ops-gcn-%
	@mkdir -p "$(PLOTS_DIR)"
	$(RUN) eval/plot_fa_recall.py \
	  --ops_yaml "$(OPS_GCN_$*)" \
	  --out "$(PLOTS_DIR)/gcn_$*_recall_vs_fa.png" \
	  --out_f1 "$(PLOTS_DIR)/gcn_$*_f1_vs_tau.png" \
	  --log_x

# ============================================================
# Mining + finetune
# ============================================================
.PHONY: mine-nearmiss-tcn-% mine-nearmiss-gcn-% mine-hardneg-tcn-% mine-hardneg-gcn-% \
        finetune-hardneg-tcn-% finetune-hardneg-gcn-%

mine-nearmiss-tcn-%: calibrate-tcn-%
	@mkdir -p "$(MINED_DIR)"
	$(RUN) eval/mine_near_miss_negatives.py \
	  --arch tcn \
	  --windows_dir "$(WIN_$*)/$(NEARMISS_SPLIT)" \
	  --ckpt "$(CKPT_TCN_$*)" \
	  --out_txt "$(MINED_DIR)/nearmiss_tcn_$*.txt" \
	  --out_csv "$(MINED_DIR)/nearmiss_tcn_$*.csv" \
	  --out_jsonl "$(MINED_DIR)/nearmiss_tcn_$*.jsonl" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(CAL_TCN_$*)" \
	  --ops_yaml "$(OPS_TCN_$*)" --op "$(OP)"

mine-nearmiss-gcn-%: calibrate-gcn-%
	@mkdir -p "$(MINED_DIR)"
	$(RUN) eval/mine_near_miss_negatives.py \
	  --arch gcn \
	  --windows_dir "$(WIN_$*)/$(NEARMISS_SPLIT)" \
	  --ckpt "$(CKPT_GCN_$*)" \
	  --out_txt "$(MINED_DIR)/nearmiss_gcn_$*.txt" \
	  --out_csv "$(MINED_DIR)/nearmiss_gcn_$*.csv" \
	  --out_jsonl "$(MINED_DIR)/nearmiss_gcn_$*.jsonl" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(CAL_GCN_$*)" \
	  --ops_yaml "$(OPS_GCN_$*)" --op "$(OP)"

mine-hardneg-tcn-%: calibrate-tcn-%
	@mkdir -p "$(MINED_DIR)"
	$(RUN) eval/mine_hard_negatives.py \
	  --arch tcn \
	  --windows_dir "$(WIN_$*)/$(HARDNEG_SPLIT)" \
	  --ckpt "$(CKPT_TCN_$*)" \
	  --out_txt "$(MINED_DIR)/hardneg_tcn_$*.txt" \
	  --out_csv "$(MINED_DIR)/hardneg_tcn_$*.csv" \
	  --out_jsonl "$(MINED_DIR)/hardneg_tcn_$*.jsonl" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(CAL_TCN_$*)"

mine-hardneg-gcn-%: calibrate-gcn-%
	@mkdir -p "$(MINED_DIR)"
	$(RUN) eval/mine_hard_negatives.py \
	  --arch gcn \
	  --windows_dir "$(WIN_$*)/$(HARDNEG_SPLIT)" \
	  --ckpt "$(CKPT_GCN_$*)" \
	  --out_txt "$(MINED_DIR)/hardneg_gcn_$*.txt" \
	  --out_csv "$(MINED_DIR)/hardneg_gcn_$*.csv" \
	  --out_jsonl "$(MINED_DIR)/hardneg_gcn_$*.jsonl" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(CAL_GCN_$*)"

finetune-hardneg-tcn-%: mine-hardneg-tcn-%
	@OUT="$(OUT_TCN_$*)_hardneg"; \
	if [ -f "$$OUT/best.pt" ] && [ "$(FORCE)" != "1" ]; then echo "[skip] $$OUT/best.pt exists (FORCE=1 to rerun)"; exit 0; fi; \
	mkdir -p "$$OUT"; \
	$(RUN) models/train_tcn.py \
	  --train_dir "$(WIN_$*)/train" --val_dir "$(WIN_$*)/val" \
	  --fps_default "$(FPS_$*)" \
	  --epochs "$(EPOCHS_TCN)" --patience "$(PATIENCE_TCN)" \
	  --batch "$(BATCH_TCN)" --lr "$(LR_TCN)" --seed "$(SPLIT_SEED)" \
	  --resume "$(CKPT_TCN_$*)" \
	  --hard_neg_list "$(MINED_DIR)/hardneg_tcn_$*.txt" --hard_neg_mult "$(HARDNEG_MULT)" \
	  --save_dir "$$OUT" \
	  $(TRAIN_COMMON_FLAGS)

finetune-hardneg-gcn-%: mine-hardneg-gcn-%
	@OUT="$(OUT_GCN_$*)_hardneg"; \
	if [ -f "$$OUT/best.pt" ] && [ "$(FORCE)" != "1" ]; then echo "[skip] $$OUT/best.pt exists (FORCE=1 to rerun)"; exit 0; fi; \
	mkdir -p "$$OUT"; \
	$(RUN) models/train_gcn.py \
	  --train_dir "$(WIN_$*)/train" --val_dir "$(WIN_$*)/val" \
	  --fps_default "$(FPS_$*)" \
	  --epochs "$(EPOCHS_GCN)" --patience "$(PATIENCE_GCN)" --min_epochs "$(MIN_EPOCHS_GCN)" \
	  --batch "$(BATCH_GCN)" --lr "$(LR_GCN)" --seed "$(SPLIT_SEED)" \
	  --resume "$(CKPT_GCN_$*)" \
	  --hard_neg_list "$(MINED_DIR)/hardneg_gcn_$*.txt" --hard_neg_mult "$(HARDNEG_MULT)" \
	  --save_dir "$$OUT" \
	  $(TRAIN_COMMON_FLAGS)

# ============================================================
# Deploy simulation runner (optional)
# ============================================================
.PHONY: deploy-tcn-% deploy-gcn-% deploy-dual-%

deploy-tcn-%: fit-ops-tcn-%
	@SPL="$(DEPLOY_SPLIT)"; \
	if [ ! -d "$(WIN_$*)/$$SPL" ]; then SPL="test"; fi; \
	$(RUN) deploy/run_modes.py \
	  --mode tcn \
	  --win_dir "$(WIN_$*)/$$SPL" \
	  --ckpt_tcn "$(CKPT_TCN_$*)" \
	  --cfg "$(DEPLOY_CFG)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --time_mode "$(DEPLOY_TIME_MODE)"

deploy-gcn-%: fit-ops-gcn-%
	@SPL="$(DEPLOY_SPLIT)"; \
	if [ ! -d "$(WIN_$*)/$$SPL" ]; then SPL="test"; fi; \
	$(RUN) deploy/run_modes.py \
	  --mode gcn \
	  --win_dir "$(WIN_$*)/$$SPL" \
	  --ckpt_gcn "$(CKPT_GCN_$*)" \
	  --cfg "$(DEPLOY_CFG)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --time_mode "$(DEPLOY_TIME_MODE)"

deploy-dual-%: fit-ops-tcn-% fit-ops-gcn-%
	@SPL="$(DEPLOY_SPLIT)"; \
	if [ ! -d "$(WIN_$*)/$$SPL" ]; then SPL="test"; fi; \
	$(RUN) deploy/run_modes.py \
	  --mode dual \
	  --win_dir "$(WIN_$*)/$$SPL" \
	  --ckpt_tcn "$(CKPT_TCN_$*)" \
	  --ckpt_gcn "$(CKPT_GCN_$*)" \
	  --cfg "$(DEPLOY_CFG)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --time_mode "$(DEPLOY_TIME_MODE)"

# ============================================================
# Meta targets
# ============================================================
.PHONY: pipeline-% workflow-tcn-% workflow-gcn-% pipeline-all workflow-all

pipeline-%: extract-% preprocess-% labels-% splits-% windows-%
	@:

workflow-tcn-%: windows-% train-tcn-% calibrate-tcn-% fit-ops-tcn-% eval-tcn-%
	@:

workflow-gcn-%: windows-% train-gcn-% calibrate-gcn-% fit-ops-gcn-% eval-gcn-%
	@:

pipeline-all:
	@for d in $(DATASETS); do $(MAKE) pipeline-$$d; done

workflow-all:
	@for d in $(DATASETS); do $(MAKE) workflow-tcn-$$d; $(MAKE) workflow-gcn-$$d; done

# ============================================================
# Cleaning helpers
# ============================================================
.PHONY: clean-metrics clean-ops clean-cal clean-mined clean-plots clean-windows clean-all

clean-metrics:
	@rm -rf "$(MET_DIR)" && echo "[ok] removed $(MET_DIR)"

clean-ops:
	@rm -rf "$(OPS_DIR)" && echo "[ok] removed $(OPS_DIR)"

clean-cal:
	@rm -rf "$(CAL_DIR)" && echo "[ok] removed $(CAL_DIR)"

clean-mined:
	@rm -rf "$(MINED_DIR)" && echo "[ok] removed $(MINED_DIR)"

clean-plots:
	@rm -rf "$(PLOTS_DIR)" && echo "[ok] removed $(PLOTS_DIR)"

clean-windows:
	@rm -rf "$(PROCESSED)" && echo "[ok] removed $(PROCESSED)"

clean-all: clean-metrics clean-ops clean-cal clean-mined clean-plots
	@rm -rf "$(OUT_DIR)" && echo "[ok] removed $(OUT_DIR)"
