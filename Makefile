# ------------------------------------------------------------
# Makefile — Skeleton-only Fall Detection (ML + Eval + Deploy-sim)
# Core + Nice-to-have targets, refactored to reduce duplication.
# (Core + nice targets; dataset-specific extract + preprocess; optional UR_Fall prep helpers.)
# ------------------------------------------------------------

# extract (raw video/images → pose_npz_raw)

# preprocess (clean/resample/normalize → pose_npz)

# labels (make labels.json + spans.json)

# splits (train/val/test lists of sequences)

# windows (build windows_W.._S.. from pose_npz + spans + splits)

# train (produce best.pt)

# calibrate (optional but recommended: temperature scaling / reliability calibration on val)

# fit thresholds / ops (choose deployment thresholds, cooldown, OP-1/2/3 params on val)

# plot / report (PR curve, ROC, calibration curve, ops plots, confusion vs threshold)

SHELL := /bin/bash
.DEFAULT_GOAL := help

THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))

.SECONDARY:
.DELETE_ON_ERROR:

# ============================================================
# Python runner
# ============================================================
PY ?= python3
VENV_ACT ?= .venv/bin/activate

ifneq ("$(wildcard $(VENV_ACT))","")
RUN := source "$(VENV_ACT)" && PYTHONPATH="$(CURDIR)" $(PY) -u
else
RUN := PYTHONPATH="$(CURDIR)" $(PY) -u
endif

# ============================================================
# Datasets supported
# ============================================================
DATASETS := le2i urfd caucafall muvim

# ============================================================
# Paths
# ============================================================
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

# ============================================================
# Per-dataset roots (override if your paths differ)
# ============================================================
RAW_le2i      := $(RAW_DIR)/LE2i
RAW_urfd      := $(RAW_DIR)/UR_Fall
RAW_urfd_clip := $(RAW_DIR)/UR_Fall_clips
RAW_caucafall := $(RAW_DIR)/CAUCAFall
RAW_muvim     := $(RAW_DIR)/MUVIM

# Source FPS defaults (used when raw/extracted metadata is missing)
FPS_le2i      ?= 25
FPS_urfd      ?= 30
FPS_caucafall ?= 23
FPS_muvim     ?= 30

# ============================================================
# Dataset formats (extraction)
# ============================================================
# le2i      : videos (.avi)
# urfd      : UR_Fall image sequences (.jpg/.jpeg/.png) under RAW_urfd_clip
# caucafall : image sequences (.png)
# muvim     : image sequences (.png) under ZED_RGB
URFD_SEQUENCE_ID_DEPTH ?= 2
CAUCA_SEQUENCE_ID_DEPTH ?= 2
MUVIM_SEQUENCE_ID_DEPTH ?= 2

# Restrict globs to the actual dataset formats to avoid picking up labels/CSVs/etc.
URFD_IMAGES_GLOBS  ?= "$(RAW_urfd_clip)/**/*.jpg" "$(RAW_urfd_clip)/**/*.jpeg" "$(RAW_urfd_clip)/**/*.png"
CAUCA_IMAGES_GLOB  ?= "$(RAW_caucafall)/**/*.png"
MUVIM_IMAGES_GLOB  ?= "$(RAW_muvim)/ZED_RGB/**/*.png"

# ============================================================
# Preprocess (pose_npz_raw -> pose_npz)
# CLI: pose/preprocess_pose_npz.py (new-only decoupled flags)
# ============================================================
PREPROC_TARGET_FPS ?= 30
POST_FPS ?= $(PREPROC_TARGET_FPS)

# Decoupled thresholds
# - conf_gate: validity mask for smoothing/filling
# - fill_conf_thr: confidence assigned to filled points (if fill_conf=thr)
# - norm_conf_gate: frames below this are excluded from normalization math
PREPROC_CONF_GATE      ?= 0.20
PREPROC_FILL_CONF_THR  ?= 0.20
PREPROC_NORM_CONF_GATE ?= 0.10

PREPROC_CONF_GATE_CLEAN      = $(strip $(PREPROC_CONF_GATE))
PREPROC_FILL_CONF_THR_CLEAN  = $(strip $(PREPROC_FILL_CONF_THR))
PREPROC_NORM_CONF_GATE_CLEAN = $(strip $(PREPROC_NORM_CONF_GATE))

# Smoothing
PREPROC_ONE_EURO ?= 1
PREPROC_ONE_EURO_MIN_CUTOFF ?= 1.0
PREPROC_ONE_EURO_BETA ?= 0.0
PREPROC_ONE_EURO_D_CUTOFF ?= 1.0

PREPROC_SMOOTH_K ?= 1
PREPROC_MAX_GAP  ?= 4
# Gap fill confidence policy: keep|thr|min_neighbors|linear
PREPROC_FILL_CONF ?= thr

# Frame-level gating (preprocess wrapper)
PREPROC_MIN_VALID_RATIO ?= 0.25
# 1 => pass --invalidate_bad_frames
PREPROC_INVALIDATE_BAD_FRAMES ?= 1

# Normalisation / rotation knobs (wrapper-level)
# normalize: none|torso|shoulder
PREPROC_NORMALIZE ?= torso
# rotate: none|shoulders
PREPROC_ROTATE ?= none
# pelvis_fill: nearest|zero
PREPROC_PELVIS_FILL ?= nearest

# Cleaned variants (strip whitespace in case of overrides)
PREPROC_FILL_CONF_CLEAN = $(strip $(PREPROC_FILL_CONF))
PREPROC_NORMALIZE_CLEAN = $(strip $(PREPROC_NORMALIZE))
PREPROC_ROTATE_CLEAN = $(strip $(PREPROC_ROTATE))
PREPROC_PELVIS_FILL_CLEAN = $(strip $(PREPROC_PELVIS_FILL))
PREPROC_INVALIDATE_BAD_FRAMES_CLEAN = $(strip $(PREPROC_INVALIDATE_BAD_FRAMES))
# Windows
# ============================================================
WIN_W ?= 48
WIN_S ?= 12

WIN_EXTRA ?= --strategy balanced --min_overlap_frames 1 --pos_per_span 80 --neg_ratio 3.0 \
            --max_neg_per_video 250 --max_windows_per_video_no_spans 120 --min_valid_frac 0.0 \
            --skip_existing --write_manifest

# ============================================================
# Unlabeled windows (LE2i; optional)
# ============================================================
LE2I_UNLABELED_SCENES ?= Office "Lecture room"
UNLAB_SUBSET ?= test_unlabeled
UNLAB_MAX_WINDOWS_PER_VIDEO ?= 400
UNLAB_CONF_GATE ?= 0.20
UNLAB_MIN_VALID_FRAC ?= 0.00
UNLAB_MIN_AVG_CONF ?= 0.00
UNLAB_USE_PRECOMPUTED_MASK ?= 1

# ============================================================
# Training knobs
# ============================================================
FORCE ?= 0

# Stamp files to mark completed stages (so .PHONY targets can skip fast)
STAMP_EXTRACT := .done_extract
STAMP_PREPROC := .done_preprocess
STAMP_WINDOWS := .done_windows

SPLIT_SEED ?= 33724876

# Keep your existing names (TCN/GCN), but also expose arch-keyed aliases for templates.
EPOCHS_TCN ?= 200
BATCH_TCN  ?= 64
LR_TCN     ?= 0.0005
PATIENCE_TCN ?= 25

EPOCHS_GCN ?= 200
BATCH_GCN  ?= 64
LR_GCN     ?= 0.0005
PATIENCE_GCN ?= 30
MIN_EPOCHS_GCN ?= 20

# Arch aliases used internally by templates (override if you want)
EPOCHS_tcn ?= $(EPOCHS_TCN)
BATCH_tcn  ?= $(BATCH_TCN)
LR_tcn     ?= $(LR_TCN)
PATIENCE_tcn ?= $(PATIENCE_TCN)
MIN_EPOCHS_tcn ?= 0

EPOCHS_gcn ?= $(EPOCHS_GCN)
BATCH_gcn  ?= $(BATCH_GCN)
LR_gcn     ?= $(LR_GCN)
PATIENCE_gcn ?= $(PATIENCE_GCN)
MIN_EPOCHS_gcn ?= $(MIN_EPOCHS_GCN)

LR_PLATEAU_PATIENCE ?= 5
LR_PLATEAU_FACTOR   ?= 0.5
LR_PLATEAU_MIN_LR   ?= 1e-6

MONITOR ?= ap
LOSS ?= bce
FOCAL_ALPHA ?= 0.25
FOCAL_GAMMA ?= 2.0

BALANCED_SAMPLER ?= 1
BALANCED_SAMPLER_FLAG := $(if $(filter 1 true yes,$(BALANCED_SAMPLER)),--balanced_sampler,)

EMA_DECAY  ?= 0.999
PREFER_EMA ?= 1

MASK_JOINT_P ?= 0.15
MASK_FRAME_P ?= 0.10
AUG_HFLIP_P ?= 0.50
AUG_JITTER_STD ?= 0.008
AUG_JITTER_CONF_SCALED ?= 1
AUG_OCC_P ?= 0.30
AUG_OCC_MIN ?= 3
AUG_OCC_MAX ?= 10
AUG_TIME_SHIFT ?= 1

# ============================================================
# fit_ops knobs
# ============================================================
OP ?= op2
OPS ?= op1 op2 op3

CONFIRM ?= 1
QUALITY_ADAPT ?= 1
QUALITY_MIN   ?= 0.25
QUALITY_BOOST ?= 0.15
QUALITY_BOOST_LOW ?= 0.10

# Confirmation knobs (passed to fit_ops.py; stored in ops YAML for eval/replay)
# Balanced defaults (recommended for CAUCAFall; override per run if needed):
#   make ... CONFIRM_S=3.0 CONFIRM_MIN_LYING=0.50 CONFIRM_MAX_MOTION=0.12 CONFIRM_REQUIRE_LOW=1
CONFIRM_S ?= 3.0
CONFIRM_MIN_LYING ?= 0.50
CONFIRM_MAX_MOTION ?= 0.12
CONFIRM_REQUIRE_LOW ?= 1

# Extra raw flags for eval/fit_ops.py (advanced)
FITOPS_EXTRA ?=


TEMP ?=
# FA reference directory for fit_ops: auto | none | <path>
FITOPS_FA ?= auto
FITOPS_FA_CLEAN = $(strip $(FITOPS_FA))

# ============================================================
# Mining knobs (optional)
# ============================================================
HARDNEG_SPLIT ?= val
NEARMISS_SPLIT ?= val
HARDNEG_MULT ?= 5

# ============================================================
# Deploy sim knobs (optional)
# ============================================================
DEPLOY_CFG ?= $(CFG_DIR)/deploy_modes.yaml
DEPLOY_SPLIT ?= $(UNLAB_SUBSET)
DEPLOY_TIME_MODE ?= center
SKIP_IF_NO_VIDEOS ?= 0

# ============================================================
# Helpers: paths
# ============================================================
pose_raw   = $(INTERIM)/$1/pose_npz_raw
pose_dir   = $(INTERIM)/$1/pose_npz


labels_json = $(LABELS_DIR)/$1.json
spans_json  = $(LABELS_DIR)/$1_spans.json

split_train = $(SPLITS_DIR)/$1_train.txt
split_val   = $(SPLITS_DIR)/$1_val.txt
split_test  = $(SPLITS_DIR)/$1_test.txt
split_unlabeled_le2i = $(SPLITS_DIR)/le2i_unlabeled.txt

win_dir    = $(PROCESSED)/$1/windows_W$(WIN_W)_S$(WIN_S)

out_dir    = $(OUT_DIR)/$1_$2_W$(WIN_W)S$(WIN_S)
ckpt       = $(call out_dir,$1,$2)/best.pt

hardneg_out_dir = $(call out_dir,$1,$2)_hardneg
hardneg_ckpt    = $(call hardneg_out_dir,$1,$2)/best.pt

cal_yaml   = $(CAL_DIR)/$2_$1.yaml
ops_yaml   = $(OPS_DIR)/$2_$1.yaml

# ============================================================
# Help
# ============================================================
.PHONY: help print-config
help:
	@echo "Core pipeline (per dataset):"
	@echo "  make extract-<ds>            # pose from videos/images (ds: $(DATASETS))"
	@echo "  make preprocess-<ds>         # resample + smoothing + gap fill"
	@echo "  make labels-<ds>             # labels.json + spans.json"
	@echo "  make splits-<ds>             # train/val/test lists"
	@echo "  make windows-<ds>            # build windows_W*_S*"
	@echo "  make check-windows-<ds>      # quick schema checks (nice)"
	@echo ""
	@echo "Training + eval:"
	@echo "  make train-tcn-<ds> | train-gcn-<ds>"
	@echo "  make calibrate-tcn-<ds> | calibrate-gcn-<ds>"
	@echo "  make fit-ops-tcn-<ds> | fit-ops-gcn-<ds>"
	@echo "  make eval-tcn-<ds> | eval-gcn-<ds>        (OP=$(OP))"
	@echo "  make replay-tcn-<ds> | replay-gcn-<ds>    (OP=$(OP))"
	@echo ""
	@echo "Nice-to-have:"
	@echo "  make plot-ops-tcn-<ds> | plot-ops-gcn-<ds>"
	@echo "  make mine-hardneg-tcn-<ds> | mine-hardneg-gcn-<ds>"
	@echo "  make mine-nearmiss-tcn-<ds> | mine-nearmiss-gcn-<ds> (OP=$(OP))"
	@echo "  make finetune-hardneg-tcn-<ds> | finetune-hardneg-gcn-<ds>"
	@echo "  make deploy-tcn-<ds> | deploy-gcn-<ds> | deploy-dual-<ds>"
	@echo ""
	@echo "LE2i unlabeled (nice):"
	@echo "  make unlabeled-le2i"
	@echo "  make score-unlabeled-tcn-le2i | score-unlabeled-gcn-le2i"
	@echo ""
	@echo "Meta:"
	@echo "  make pipeline-<ds>           # extract->preprocess->labels->splits->windows"
	@echo "  make workflow-tcn-<ds>       # pipeline->train->calibrate->fit-ops->eval"
	@echo "  make workflow-gcn-<ds>"
	@echo "  make workflow-full-<ds>      # tcn+gcn+dual + mining + deploy (nice)"
	@echo ""
	@echo "Inspect config:"
	@echo "  make print-config"

print-config:
	@echo "WIN_W=$(WIN_W) WIN_S=$(WIN_S) POST_FPS=$(POST_FPS) PREPROC_TARGET_FPS=$(PREPROC_TARGET_FPS)"
	@echo "PREPROC_CONF_GATE=$(PREPROC_CONF_GATE) PREPROC_FILL_CONF_THR=$(PREPROC_FILL_CONF_THR) PREPROC_NORM_CONF_GATE=$(PREPROC_NORM_CONF_GATE) PREPROC_MAX_GAP=$(PREPROC_MAX_GAP) PREPROC_FILL_CONF=$(PREPROC_FILL_CONF_CLEAN)"
	@echo "LOSS=$(LOSS) MONITOR=$(MONITOR) EMA_DECAY=$(EMA_DECAY) BALANCED_SAMPLER=$(BALANCED_SAMPLER)"
	@echo "CONFIRM=$(CONFIRM) CONFIRM_S=$(CONFIRM_S) CONFIRM_MIN_LYING=$(CONFIRM_MIN_LYING) CONFIRM_MAX_MOTION=$(CONFIRM_MAX_MOTION) CONFIRM_REQUIRE_LOW=$(CONFIRM_REQUIRE_LOW)"
	@echo "Image globs: URFD=$(URFD_IMAGES_GLOBS)"
	@echo "            CAUCA=$(CAUCA_IMAGES_GLOB)"
	@echo "            MUVIM=$(MUVIM_IMAGES_GLOB)"

# ============================================================
# Pose extraction
# ============================================================
.PHONY: extract-le2i extract-urfd extract-caucafall extract-muvim

# LE2i is videos (.avi)
extract-le2i:
	@mkdir -p "$(call pose_raw,le2i)"
	@if [ "$(FORCE)" != "1" ] && [ -f "$(call pose_raw,le2i)/$(STAMP_EXTRACT)" ]; then \
	  echo "[skip] extract-le2i (stamp exists: $(call pose_raw,le2i)/$(STAMP_EXTRACT))"; \
	elif [ "$(FORCE)" != "1" ] && [ -n "$$(find "$(call pose_raw,le2i)" -type f -name '*.npz' -print -quit 2>/dev/null)" ]; then \
	  echo "[skip] extract-le2i (npz exists) -> stamping"; \
	  touch "$(call pose_raw,le2i)/$(STAMP_EXTRACT)"; \
	else \
	  $(RUN) pose/extract_2d.py \
	    --videos_glob "$(RAW_le2i)/**/Videos/*.avi" "$(RAW_le2i)/**/*.avi" \
	    --out_dir "$(call pose_raw,le2i)" \
	    --fps_default "$(FPS_le2i)" \
	    --skip_existing || true; \
	  touch "$(call pose_raw,le2i)/$(STAMP_EXTRACT)"; \
	fi

# UR_Fall (dataset code: urfd) is image sequences (.jpg/.png) — use the prepared clips folder
extract-urfd:
	@mkdir -p "$(call pose_raw,urfd)"
	@if [ "$(FORCE)" != "1" ] && [ -f "$(call pose_raw,urfd)/$(STAMP_EXTRACT)" ]; then \
	  echo "[skip] extract-urfd (stamp exists: $(call pose_raw,urfd)/$(STAMP_EXTRACT))"; \
	elif [ "$(FORCE)" != "1" ] && [ -n "$$(find "$(call pose_raw,urfd)" -type f -name '*.npz' -print -quit 2>/dev/null)" ]; then \
	  echo "[skip] extract-urfd (npz exists) -> stamping"; \
	  touch "$(call pose_raw,urfd)/$(STAMP_EXTRACT)"; \
	else \
	  $(RUN) pose/extract_2d_from_images.py \
	    --dataset urfd \
	    --images_glob $(URFD_IMAGES_GLOBS) \
	    --sequence_id_depth "$(URFD_SEQUENCE_ID_DEPTH)" \
	    --out_dir "$(call pose_raw,urfd)" \
	    --fps "$(FPS_urfd)" \
	    --skip_existing || true; \
	  touch "$(call pose_raw,urfd)/$(STAMP_EXTRACT)"; \
	fi

# CAUCAFall is image sequences (.png)
extract-caucafall:
	@mkdir -p "$(call pose_raw,caucafall)"
	@if [ "$(FORCE)" != "1" ] && [ -f "$(call pose_raw,caucafall)/$(STAMP_EXTRACT)" ]; then \
	  echo "[skip] extract-caucafall (stamp exists: $(call pose_raw,caucafall)/$(STAMP_EXTRACT))"; \
	elif [ "$(FORCE)" != "1" ] && [ -n "$$(find "$(call pose_raw,caucafall)" -type f -name '*.npz' -print -quit 2>/dev/null)" ]; then \
	  echo "[skip] extract-caucafall (npz exists) -> stamping"; \
	  touch "$(call pose_raw,caucafall)/$(STAMP_EXTRACT)"; \
	else \
	  $(RUN) pose/extract_2d_from_images.py \
	    --dataset caucafall \
	    --images_glob $(CAUCA_IMAGES_GLOB) \
	    --sequence_id_depth "$(CAUCA_SEQUENCE_ID_DEPTH)" \
	    --out_dir "$(call pose_raw,caucafall)" \
	    --fps "$(FPS_caucafall)" \
	    --skip_existing || true; \
	  touch "$(call pose_raw,caucafall)/$(STAMP_EXTRACT)"; \
	fi

# MUVIM is image sequences (.png) under ZED_RGB
extract-muvim:
	@mkdir -p "$(call pose_raw,muvim)"
	@if [ "$(FORCE)" != "1" ] && [ -f "$(call pose_raw,muvim)/$(STAMP_EXTRACT)" ]; then \
	  echo "[skip] extract-muvim (stamp exists: $(call pose_raw,muvim)/$(STAMP_EXTRACT))"; \
	elif [ "$(FORCE)" != "1" ] && [ -n "$$(find "$(call pose_raw,muvim)" -type f -name '*.npz' -print -quit 2>/dev/null)" ]; then \
	  echo "[skip] extract-muvim (npz exists) -> stamping"; \
	  touch "$(call pose_raw,muvim)/$(STAMP_EXTRACT)"; \
	else \
	  $(RUN) pose/extract_2d_from_images.py \
	    --dataset muvim \
	    --images_glob $(MUVIM_IMAGES_GLOB) \
	    --sequence_id_depth "$(MUVIM_SEQUENCE_ID_DEPTH)" \
	    --out_dir "$(call pose_raw,muvim)" \
	    --fps "$(FPS_muvim)" \
	    --skip_existing || true; \
	  touch "$(call pose_raw,muvim)/$(STAMP_EXTRACT)"; \
	fi

# ============================================================
# Preprocess (pose_npz_raw -> pose_npz)
# ============================================================
# ============================================================
# Preprocess targets
# ============================================================
.PHONY: preprocess-% preprocess-only-%
preprocess-%: extract-% preprocess-only-%
	@:

preprocess-only-%:
	@mkdir -p "$(call pose_dir,$*)"
	@if [ "$(FORCE)" != "1" ] && [ -f "$(call pose_dir,$*)/$(STAMP_PREPROC)" ]; then \
	  echo "[skip] preprocess-$* (stamp exists: $(call pose_dir,$*)/$(STAMP_PREPROC))"; \
	  exit 0; \
	fi
	@if [ ! -d "$(call pose_raw,$*)" ]; then echo "[err] missing pose raw dir: $(call pose_raw,$*)"; exit 2; fi
	@if [ -z "$$(find "$(call pose_raw,$*)" -type f -name '*.npz' -print -quit 2>/dev/null)" ]; then \
	  echo "[err] no .npz files under: $(call pose_raw,$*)"; \
	  echo "      - check RAW_$* path: $(RAW_$*)"; \
	  echo "      - run: make extract-$*"; \
	  exit 2; \
	fi
	@RAWN="$$(find "$(call pose_raw,$*)" -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')"; \
	OUTN="$$(find "$(call pose_dir,$*)" -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')"; \
	if [ "$(FORCE)" != "1" ] && [ "$$RAWN" -gt 0 ] && [ "$$OUTN" -ge "$$RAWN" ]; then \
	  echo "[skip] preprocess-$* (already cleaned: $$OUTN/$$RAWN) -> stamping"; \
	  touch "$(call pose_dir,$*)/$(STAMP_PREPROC)"; \
	  exit 0; \
	fi
	@INV=""; if [ "$(PREPROC_INVALIDATE_BAD_FRAMES_CLEAN)" = "1" ]; then INV="--invalidate_bad_frames"; fi; \
	$(RUN) pose/preprocess_pose_npz.py \
	  --in_dir "$(call pose_raw,$*)" \
	  --out_dir "$(call pose_dir,$*)" \
	  --fps_default "$(FPS_$*)" \
	  --target_fps "$(PREPROC_TARGET_FPS)" \
	  --conf_gate "$(PREPROC_CONF_GATE_CLEAN)" \
	  --fill_conf_thr "$(PREPROC_FILL_CONF_THR_CLEAN)" \
	  --norm_conf_gate "$(PREPROC_NORM_CONF_GATE_CLEAN)" \
	  --max_gap "$(PREPROC_MAX_GAP)" \
	  --fill_conf "$(PREPROC_FILL_CONF_CLEAN)" \
	  --one_euro "$(PREPROC_ONE_EURO)" \
	  --one_euro_min_cutoff "$(PREPROC_ONE_EURO_MIN_CUTOFF)" \
	  --one_euro_beta "$(PREPROC_ONE_EURO_BETA)" \
	  --one_euro_d_cutoff "$(PREPROC_ONE_EURO_D_CUTOFF)" \
	  --smooth_k "$(PREPROC_SMOOTH_K)" \
	  --normalize "$(PREPROC_NORMALIZE_CLEAN)" \
	  --rotate "$(PREPROC_ROTATE_CLEAN)" \
	  --pelvis_fill "$(PREPROC_PELVIS_FILL_CLEAN)" \
	  --min_valid_ratio "$(PREPROC_MIN_VALID_RATIO)" \
	  $$INV \
	  --skip_existing
	@touch "$(call pose_dir,$*)/$(STAMP_PREPROC)"


# ============================================================
# Labels
# ============================================================
URFD_MIN_RUN ?= 2
URFD_GAP_FILL ?= 2

CAUC_MIN_RUN ?= 2
CAUC_GAP_FILL ?= 2

MUVIM_ZED_CSV ?= $(RAW_muvim)/ZED_RGB/ZED_RGB.csv

.PHONY: labels-le2i labels-urfd labels-caucafall labels-muvim

labels-le2i: preprocess-le2i
	@mkdir -p "$(LABELS_DIR)"
	@if [ "$(FORCE)" != "1" ] && [ -s "$(call labels_json,le2i)" ] && [ -s "$(call spans_json,le2i)" ]; then \
	  echo "[skip] labels-le2i (exists: $(call labels_json,le2i))"; \
	else \
	  $(RUN) labels/make_le2i_labels.py \
	    --npz_dir "$(call pose_dir,le2i)" \
	    --raw_root "$(RAW_le2i)" \
	    --out_labels "$(call labels_json,le2i)" \
	    --out_spans "$(call spans_json,le2i)"; \
	fi

labels-urfd: preprocess-urfd
	@mkdir -p "$(LABELS_DIR)"
	@if [ "$(FORCE)" != "1" ] && [ -s "$(call labels_json,urfd)" ] && [ -s "$(call spans_json,urfd)" ]; then \
	  echo "[skip] labels-urfd (exists: $(call labels_json,urfd))"; \
	else \
	  $(RUN) labels/make_urfd_labels.py \
	    --raw_root "$(RAW_urfd)" \
	    --npz_dir "$(call pose_dir,urfd)" \
	    --out_labels "$(call labels_json,urfd)" \
	    --out_spans "$(call spans_json,urfd)" \
	    --min_run "$(URFD_MIN_RUN)" \
	    --gap_fill "$(URFD_GAP_FILL)"; \
	fi

labels-caucafall: preprocess-caucafall
	@mkdir -p "$(LABELS_DIR)"
	@if [ "$(FORCE)" != "1" ] && [ -s "$(call labels_json,caucafall)" ] && [ -s "$(call spans_json,caucafall)" ]; then \
	  echo "[skip] labels-caucafall (exists: $(call labels_json,caucafall))"; \
	else \
	  $(RUN) labels/make_caucafall_labels_from_frames.py \
	    --raw_root "$(RAW_caucafall)" \
	    --npz_dir "$(call pose_dir,caucafall)" \
	    --out_labels "$(call labels_json,caucafall)" \
	    --out_spans "$(call spans_json,caucafall)" \
	    --min_run "$(CAUC_MIN_RUN)" \
	    --gap_fill "$(CAUC_GAP_FILL)"; \
	fi

labels-muvim: preprocess-muvim
	@mkdir -p "$(LABELS_DIR)"
	@if [ "$(FORCE)" != "1" ] && [ -s "$(call labels_json,muvim)" ] && [ -s "$(call spans_json,muvim)" ]; then \
	  echo "[skip] labels-muvim (exists: $(call labels_json,muvim))"; \
	else \
	  $(RUN) labels/make_muvim_labels.py \
	    --npz_dir "$(call pose_dir,muvim)" \
	    --zed_csv "$(MUVIM_ZED_CSV)" \
	    --out_labels "$(call labels_json,muvim)" \
	    --out_spans "$(call spans_json,muvim)"; \
	fi

# ============================================================
# Splits
# ============================================================
.PHONY: splits-%
splits-%: labels-%
	@mkdir -p "$(SPLITS_DIR)"
	@if [ "$(FORCE)" != "1" ] && [ -s "$(call split_train,$*)" ] && [ -s "$(call split_val,$*)" ] && [ -s "$(call split_test,$*)" ]; then \
	  echo "[skip] splits-$* (exists under $(SPLITS_DIR))"; \
	else \
	  $(RUN) split/make_splits.py \
	    --labels_json "$(call labels_json,$*)" \
	    --out_dir "$(SPLITS_DIR)" \
	    --prefix "$*" \
	    --seed "$(SPLIT_SEED)"; \
	fi


# ============================================================
# Windows
# ============================================================
.PHONY: windows-% check-windows-%
windows-%: splits-%
	@mkdir -p "$(call win_dir,$*)"
	@if [ "$(FORCE)" != "1" ] && [ -f "$(call win_dir,$*)/$(STAMP_WINDOWS)" ]; then \
	  echo "[skip] windows-$* (stamp exists: $(call win_dir,$*)/$(STAMP_WINDOWS))"; \
	elif [ "$(FORCE)" != "1" ] && \
	     [ -n "$$(find "$(call win_dir,$*)/train" -type f -name '*.npz' -print -quit 2>/dev/null)" ] && \
	     [ -n "$$(find "$(call win_dir,$*)/val"  -type f -name '*.npz' -print -quit 2>/dev/null)" ] && \
	     [ -n "$$(find "$(call win_dir,$*)/test" -type f -name '*.npz' -print -quit 2>/dev/null)" ]; then \
	  echo "[skip] windows-$* (already built) -> stamping"; \
	  touch "$(call win_dir,$*)/$(STAMP_WINDOWS)"; \
	else \
	  SP=""; \
	  if [ -f "$(call spans_json,$*)" ] && [ "$$(wc -c < "$(call spans_json,$*)")" -gt 5 ]; then SP="--spans_json \"$(call spans_json,$*)\""; fi; \
	  eval '$(RUN) windows/make_windows.py \
	    --npz_dir "$(call pose_dir,$*)" \
	    --labels_json "$(call labels_json,$*)" '$$SP' \
	    --out_dir "$(call win_dir,$*)" \
	    --W "$(WIN_W)" --stride "$(WIN_S)" \
	    --fps_default "$(POST_FPS)" \
	    --train_list "$(call split_train,$*)" \
	    --val_list "$(call split_val,$*)" \
	    --test_list "$(call split_test,$*)" \
	    $(WIN_EXTRA)' && \
	  touch "$(call win_dir,$*)/$(STAMP_WINDOWS)"; \
	fi


check-windows-%:
	@if [ ! -d "$(call win_dir,$*)" ]; then echo "[err] missing windows: $(call win_dir,$*) (run make windows-$*)"; exit 2; fi
	$(RUN) windows/check_windows.py --root "$(call win_dir,$*)"

.PHONY: clean-stamps-%
clean-stamps-%:
	@rm -f "$(call pose_raw,$*)/$(STAMP_EXTRACT)" "$(call pose_dir,$*)/$(STAMP_PREPROC)" "$(call win_dir,$*)/$(STAMP_WINDOWS)"
	@echo "[ok] removed stamps for $* (set FORCE=1 to rebuild without stamps)"


# ============================================================
# Unlabeled windows (LE2i only; optional)
# ============================================================
.PHONY: unlabeled-list-le2i windows-unlabeled-le2i unlabeled-le2i

unlabeled-list-le2i: preprocess-le2i
	@mkdir -p "$(SPLITS_DIR)"
	$(RUN) labels/make_unlabeled_test_list.py \
	  --npz_dir "$(call pose_dir,le2i)" \
	  --out "$(split_unlabeled_le2i)" \
	  --scenes $(LE2I_UNLABELED_SCENES)

windows-unlabeled-le2i: unlabeled-list-le2i
	@mkdir -p "$(call win_dir,le2i)/$(UNLAB_SUBSET)"
	@MASKFLAG=""; if [ "$(UNLAB_USE_PRECOMPUTED_MASK)" = "1" ]; then MASKFLAG="--use_precomputed_mask"; fi; \
	$(RUN) windows/make_unlabeled_windows.py \
	  --npz_dir "$(call pose_dir,le2i)" \
	  --stems_txt "$(split_unlabeled_le2i)" \
	  --out_dir "$(call win_dir,le2i)" \
	  --subset "$(UNLAB_SUBSET)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" \
	  --fps_default "$(POST_FPS)" \
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
# Training / Calibration / fit_ops / eval / replay (templated)
# ============================================================
ARCHS := tcn gcn
TRAIN_COMMON_FLAGS = \
  --monitor "$(MONITOR)" $(BALANCED_SAMPLER_FLAG) \
  --loss "$(LOSS)" --focal_alpha "$(FOCAL_ALPHA)" --focal_gamma "$(FOCAL_GAMMA)" \
  --ema_decay "$(EMA_DECAY)" \
  --mask_joint_p "$(MASK_JOINT_P)" --mask_frame_p "$(MASK_FRAME_P)" \
  --aug_hflip_p "$(AUG_HFLIP_P)" \
  --aug_jitter_std "$(AUG_JITTER_STD)" --aug_jitter_conf_scaled "$(AUG_JITTER_CONF_SCALED)" \
  --aug_occ_p "$(AUG_OCC_P)" --aug_occ_min_len "$(AUG_OCC_MIN)" --aug_occ_max_len "$(AUG_OCC_MAX)" \
  --aug_time_shift "$(AUG_TIME_SHIFT)"

TRAIN_EXTRA_tcn :=
TRAIN_EXTRA_gcn := --min_epochs "$(MIN_EPOCHS_gcn)"

define TRAIN_RULE
.PHONY: train-$(1)-% train-$(1)-only-%
train-$(1)-%: windows-% train-$(1)-only-%
	@:

train-$(1)-only-%:
	@mkdir -p "$(call out_dir,$$*,$(1))"
	@if [ ! -d "$(call win_dir,$$*)/train" ] || [ ! -d "$(call win_dir,$$*)/val" ]; then \
	  echo "[err] missing train/val windows under $(call win_dir,$$*) (run make windows-$$*)"; exit 2; \
	fi
	@if [ -f "$(call ckpt,$$*,$(1))" ] && [ "$(FORCE)" != "1" ]; then \
	  echo "[skip] $(call ckpt,$$*,$(1)) exists (FORCE=1 to retrain)"; exit 0; \
	fi
	$(RUN) models/train_$(1).py \
	  --train_dir "$(call win_dir,$$*)/train" \
	  --val_dir "$(call win_dir,$$*)/val" \
	  --epochs "$(EPOCHS_$(1))" \
	  --patience "$(PATIENCE_$(1))" \
	  $(TRAIN_EXTRA_$(1)) \
	  --batch "$(BATCH_$(1))" \
	  --lr "$(LR_$(1))" \
	  --seed "$(SPLIT_SEED)" \
	  --lr_plateau_patience "$(LR_PLATEAU_PATIENCE)" \
	  --lr_plateau_factor "$(LR_PLATEAU_FACTOR)" \
	  --lr_plateau_min_lr "$(LR_PLATEAU_MIN_LR)" \
	  --fps_default "$(POST_FPS)" \
	  --save_dir "$(call out_dir,$$*,$(1))" \
	  $(TRAIN_COMMON_FLAGS)
endef

define CAL_RULE
.PHONY: calibrate-$(1)-% calibrate-$(1)-only-%
calibrate-$(1)-%: train-$(1)-% calibrate-$(1)-only-%
	@:

calibrate-$(1)-only-%:
	@mkdir -p "$(CAL_DIR)"
	@if [ "$(FORCE)" != "1" ] && [ -s "$(call cal_yaml,$$*,$(1))" ]; then \
	  echo "[skip] calibrate-$(1)-$$* (exists: $(call cal_yaml,$$*,$(1)))"; \
	else \
	  $(RUN) eval/calibrate_temperature.py \
	    --arch $(1) \
	    --val_dir "$(call win_dir,$$*)/val" \
	    --ckpt "$(call ckpt,$$*,$(1))" \
	    --out_yaml "$(call cal_yaml,$$*,$(1))" \
	    --prefer_ema "$(PREFER_EMA)"; \
	fi
endef

define FITOPS_RULE
.PHONY: fit-ops-$(1)-% fit-ops-$(1)-only-%
fit-ops-$(1)-%: calibrate-$(1)-% fit-ops-$(1)-only-%
	@:

fit-ops-$(1)-only-%:
	@mkdir -p "$(OPS_DIR)"
	@if [ "$(FORCE)" != "1" ] && [ -s "$(call ops_yaml,$$*,$(1))" ]; then \
	  echo "[skip] fit-ops-$(1)-$$* (exists: $(call ops_yaml,$$*,$(1)))"; \
	else \
	  FA_DIR=""; \
	if [ "$(FITOPS_FA_CLEAN)" = "auto" ]; then \
	  if [ -d "$(call win_dir,$$*)/$(UNLAB_SUBSET)" ] && [ -n "$$(find "$(call win_dir,$$*)/$(UNLAB_SUBSET)" -type f -name '*.npz' -print -quit 2>/dev/null)" ]; then \
	    FA_DIR="$(call win_dir,$$*)/$(UNLAB_SUBSET)"; \
	  fi; \
	elif [ "$(FITOPS_FA_CLEAN)" = "none" ]; then \
	  FA_DIR=""; \
	else \
	  FA_DIR="$(FITOPS_FA_CLEAN)"; \
	fi; \
	TFLAG=""; if [ -n "$$$$TEMP" ]; then TFLAG="--temperature $$$$TEMP"; fi; \
	FAFLAG=""; if [ -n "$$$$FA_DIR" ]; then FAFLAG="--fa_dir=$$$$FA_DIR"; fi; \
	$(RUN) eval/fit_ops.py \
	  --arch $(1) \
	  --val_dir "$(call win_dir,$$*)/val" \
	  --ckpt "$(call ckpt,$$*,$(1))" \
	  --out "$(call ops_yaml,$$*,$(1))" \
	  --deploy_fps "$(PREPROC_TARGET_FPS)" --deploy_w "$(WIN_W)" --deploy_s "$(WIN_S)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(call cal_yaml,$$*,$(1))" $$$$TFLAG $$$$FAFLAG \
	  --confirm "$(CONFIRM)" \
	  --confirm_s "$(CONFIRM_S)" \
	  --confirm_min_lying "$(CONFIRM_MIN_LYING)" \
	  --confirm_max_motion "$(CONFIRM_MAX_MOTION)" \
	  --confirm_require_low "$(CONFIRM_REQUIRE_LOW)" \
	  --quality_adapt "$(QUALITY_ADAPT)" --quality_min "$(QUALITY_MIN)" --quality_boost "$(QUALITY_BOOST)" --quality_boost_low "$(QUALITY_BOOST_LOW)" \
	  $(FITOPS_EXTRA); \
	fi
endef

define EVAL_RULE
.PHONY: eval-$(1)-% eval-$(1)-only-%
eval-$(1)-%: fit-ops-$(1)-% eval-$(1)-only-%
	@:

eval-$(1)-only-%:
	@mkdir -p "$(MET_DIR)"
	@if [ -f "$(MET_DIR)/$(1)_$$*_$(OP).json" ] && [ "$(FORCE)" != "1" ]; then \
	  echo "[skip] $(MET_DIR)/$(1)_$$*_$(OP).json exists (FORCE=1 to re-eval)"; \
	else \
	  $(RUN) eval/metrics.py \
	    --arch $(1) \
	    --windows_dir "$(call win_dir,$$*)/test" \
	    --ckpt "$(call ckpt,$$*,$(1))" \
	    --out_json "$(MET_DIR)/$(1)_$$*_$(OP).json" \
	    --op "$(OP)" \
	    --prefer_ema "$(PREFER_EMA)" \
	    --calibration_yaml "$(call cal_yaml,$$*,$(1))" \
	    --ops_yaml "$(call ops_yaml,$$*,$(1))"; \
	  echo " [ok] wrote: $(MET_DIR)/$(1)_$$*_$(OP).json"; \
	fi
endef

define REPLAY_RULE
.PHONY: replay-$(1)-% replay-$(1)-only-%
replay-$(1)-%: fit-ops-$(1)-% replay-$(1)-only-%
	@:

replay-$(1)-only-%:
	@mkdir -p "$(MET_DIR)"
	@if [ -f "$(MET_DIR)/replay_$(1)_$$*_$(OP).json" ] && [ "$(FORCE)" != "1" ]; then \
	  echo "[skip] $(MET_DIR)/replay_$(1)_$$*_$(OP).json exists (FORCE=1 to replay again)"; \
	else \
	  $(RUN) eval/replay_eval.py \
	    --arch $(1) \
	    --windows_dir "$(call win_dir,$$*)/test" \
	    --ckpt "$(call ckpt,$$*,$(1))" \
	    --out_json "$(MET_DIR)/replay_$(1)_$$*_$(OP).json" \
	    --op "$(OP)" \
	    --prefer_ema "$(PREFER_EMA)" \
	    --calibration_yaml "$(call cal_yaml,$$*,$(1))" \
	    --ops_yaml "$(call ops_yaml,$$*,$(1))"; \
	  echo " [ok] wrote: $(MET_DIR)/replay_$(1)_$$*_$(OP).json"; \
	fi
endef

define PLOT_RULE
.PHONY: plot-ops-$(1)-% plot-ops-$(1)-only-%
plot-ops-$(1)-%: fit-ops-$(1)-% plot-ops-$(1)-only-%
	@:

plot-ops-$(1)-only-%:
	@mkdir -p "$(PLOTS_DIR)"
	$(RUN) eval/plot_fa_recall.py \
	  --ops_yaml "$(call ops_yaml,$$*,$(1))" \
	  --out "$(PLOTS_DIR)/$(1)_$$*_recall_vs_fa.png" \
	  --out_f1 "$(PLOTS_DIR)/$(1)_$$*_f1_vs_tau.png" \
	  --log_x
endef

define MINE_NEARMISS_RULE
.PHONY: mine-nearmiss-$(1)-% mine-nearmiss-$(1)-only-%
mine-nearmiss-$(1)-%: fit-ops-$(1)-% mine-nearmiss-$(1)-only-%
	@:

mine-nearmiss-$(1)-only-%:
	@mkdir -p "$(MINED_DIR)"
	$(RUN) eval/mine_near_miss_negatives.py \
	  --arch $(1) \
	  --windows_dir "$(call win_dir,$$*)/$(NEARMISS_SPLIT)" \
	  --ckpt "$(call ckpt,$$*,$(1))" \
	  --out_txt "$(MINED_DIR)/nearmiss_$(1)_$$*.txt" \
	  --out_csv "$(MINED_DIR)/nearmiss_$(1)_$$*.csv" \
	  --out_jsonl "$(MINED_DIR)/nearmiss_$(1)_$$*.jsonl" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(call cal_yaml,$$*,$(1))" \
	  --ops_yaml "$(call ops_yaml,$$*,$(1))" \
	  --op "$(OP)"
endef

define MINE_HARDNEG_RULE
.PHONY: mine-hardneg-$(1)-% mine-hardneg-$(1)-only-%
mine-hardneg-$(1)-%: calibrate-$(1)-% mine-hardneg-$(1)-only-%
	@:

mine-hardneg-$(1)-only-%:
	@mkdir -p "$(MINED_DIR)"
	$(RUN) eval/mine_hard_negatives.py \
	  --arch $(1) \
	  --windows_dir "$(call win_dir,$$*)/$(HARDNEG_SPLIT)" \
	  --ckpt "$(call ckpt,$$*,$(1))" \
	  --out_txt "$(MINED_DIR)/hardneg_$(1)_$$*.txt" \
	  --out_csv "$(MINED_DIR)/hardneg_$(1)_$$*.csv" \
	  --out_jsonl "$(MINED_DIR)/hardneg_$(1)_$$*.jsonl" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(call cal_yaml,$$*,$(1))"
endef

define FINETUNE_HARDNEG_RULE
.PHONY: finetune-hardneg-$(1)-% finetune-hardneg-$(1)-only-%
finetune-hardneg-$(1)-%: mine-hardneg-$(1)-% finetune-hardneg-$(1)-only-%
	@:

finetune-hardneg-$(1)-only-%:
	@OUT="$(call hardneg_out_dir,$$*,$(1))"; \
	if [ -f "$$OUT/best.pt" ] && [ "$(FORCE)" != "1" ]; then echo "[skip] $$OUT/best.pt exists (FORCE=1 to rerun)"; exit 0; fi; \
	mkdir -p "$$OUT"; \
	$(RUN) models/train_$(1).py \
	  --train_dir "$(call win_dir,$$*)/train" \
	  --val_dir "$(call win_dir,$$*)/val" \
	  --fps_default "$(POST_FPS)" \
	  --epochs "$(EPOCHS_$(1))" \
	  --patience "$(PATIENCE_$(1))" \
	  $(TRAIN_EXTRA_$(1)) \
	  --batch "$(BATCH_$(1))" \
	  --lr "$(LR_$(1))" \
	  --seed "$(SPLIT_SEED)" \
	  --resume "$(call ckpt,$$*,$(1))" \
	  --hard_neg_list "$(MINED_DIR)/hardneg_$(1)_$$*.txt" \
	  --hard_neg_mult "$(HARDNEG_MULT)" \
	  --save_dir "$$OUT" \
	  $(TRAIN_COMMON_FLAGS)
endef

$(foreach A,$(ARCHS),$(eval $(call TRAIN_RULE,$(A))))
$(foreach A,$(ARCHS),$(eval $(call CAL_RULE,$(A))))
$(foreach A,$(ARCHS),$(eval $(call FITOPS_RULE,$(A))))
$(foreach A,$(ARCHS),$(eval $(call EVAL_RULE,$(A))))
$(foreach A,$(ARCHS),$(eval $(call REPLAY_RULE,$(A))))
$(foreach A,$(ARCHS),$(eval $(call PLOT_RULE,$(A))))
$(foreach A,$(ARCHS),$(eval $(call MINE_NEARMISS_RULE,$(A))))
$(foreach A,$(ARCHS),$(eval $(call MINE_HARDNEG_RULE,$(A))))
$(foreach A,$(ARCHS),$(eval $(call FINETUNE_HARDNEG_RULE,$(A))))

# ============================================================
# Unlabeled FA scoring (LE2i only; nice)
# ============================================================
.PHONY: score-unlabeled-tcn-le2i score-unlabeled-gcn-le2i
score-unlabeled-tcn-le2i: fit-ops-tcn-le2i unlabeled-le2i
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/score_unlabeled_alert_rate.py \
	  --arch tcn \
	  --windows_dir "$(call win_dir,le2i)/$(UNLAB_SUBSET)" \
	  --ckpt "$(call ckpt,le2i,tcn)" \
	  --out_json "$(MET_DIR)/unlabeled_tcn_le2i.json" \
	  --op "$(OP)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(call cal_yaml,le2i,tcn)" \
	  --ops_yaml "$(call ops_yaml,le2i,tcn)"

score-unlabeled-gcn-le2i: fit-ops-gcn-le2i unlabeled-le2i
	@mkdir -p "$(MET_DIR)"
	$(RUN) eval/score_unlabeled_alert_rate.py \
	  --arch gcn \
	  --windows_dir "$(call win_dir,le2i)/$(UNLAB_SUBSET)" \
	  --ckpt "$(call ckpt,le2i,gcn)" \
	  --out_json "$(MET_DIR)/unlabeled_gcn_le2i.json" \
	  --op "$(OP)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --calibration_yaml "$(call cal_yaml,le2i,gcn)" \
	  --ops_yaml "$(call ops_yaml,le2i,gcn)"

# ============================================================
# Deploy simulation runner (optional)
# ============================================================
$(DEPLOY_CFG):
	@mkdir -p "$(CFG_DIR)"
	@if [ -f "$@" ]; then :; \
	elif [ -f "deploy_modes.yaml" ]; then \
	  echo "[info] copying deploy_modes.yaml -> $@"; \
	  cp "deploy_modes.yaml" "$@"; \
	else \
	  echo "[err] missing deploy config. Put deploy_modes.yaml in repo root OR $(CFG_DIR)/deploy_modes.yaml"; \
	  exit 2; \
	fi

.PHONY: deploy-tcn-% deploy-gcn-% deploy-dual-% deploy-tcn-hardneg-% deploy-gcn-hardneg-% deploy-dual-hardneg-%

deploy-tcn-%: fit-ops-tcn-% $(DEPLOY_CFG)
	@SPL="$(DEPLOY_SPLIT)"; \
	if [ ! -d "$(call win_dir,$*)/$$SPL" ]; then SPL="test"; fi; \
	$(RUN) deploy/run_modes.py \
	  --mode tcn \
	  --win_dir "$(call win_dir,$*)/$$SPL" \
	  --ckpt_tcn "$(call ckpt,$*,tcn)" \
	  --cfg "$(DEPLOY_CFG)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --time_mode "$(DEPLOY_TIME_MODE)"

deploy-gcn-%: fit-ops-gcn-% $(DEPLOY_CFG)
	@SPL="$(DEPLOY_SPLIT)"; \
	if [ ! -d "$(call win_dir,$*)/$$SPL" ]; then SPL="test"; fi; \
	$(RUN) deploy/run_modes.py \
	  --mode gcn \
	  --win_dir "$(call win_dir,$*)/$$SPL" \
	  --ckpt_gcn "$(call ckpt,$*,gcn)" \
	  --cfg "$(DEPLOY_CFG)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --time_mode "$(DEPLOY_TIME_MODE)"

deploy-dual-%: fit-ops-tcn-% fit-ops-gcn-% $(DEPLOY_CFG)
	@SPL="$(DEPLOY_SPLIT)"; \
	if [ ! -d "$(call win_dir,$*)/$$SPL" ]; then SPL="test"; fi; \
	$(RUN) deploy/run_modes.py \
	  --mode dual \
	  --win_dir "$(call win_dir,$*)/$$SPL" \
	  --ckpt_tcn "$(call ckpt,$*,tcn)" \
	  --ckpt_gcn "$(call ckpt,$*,gcn)" \
	  --cfg "$(DEPLOY_CFG)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --time_mode "$(DEPLOY_TIME_MODE)"

deploy-tcn-hardneg-%: finetune-hardneg-tcn-% $(DEPLOY_CFG)
	@SPL="$(DEPLOY_SPLIT)"; \
	if [ ! -d "$(call win_dir,$*)/$$SPL" ]; then SPL="test"; fi; \
	CKPT="$(call hardneg_ckpt,$*,tcn)"; \
	$(RUN) deploy/run_modes.py \
	  --mode tcn \
	  --win_dir "$(call win_dir,$*)/$$SPL" \
	  --ckpt_tcn "$$CKPT" \
	  --cfg "$(DEPLOY_CFG)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --time_mode "$(DEPLOY_TIME_MODE)"

deploy-gcn-hardneg-%: finetune-hardneg-gcn-% $(DEPLOY_CFG)
	@SPL="$(DEPLOY_SPLIT)"; \
	if [ ! -d "$(call win_dir,$*)/$$SPL" ]; then SPL="test"; fi; \
	CKPT="$(call hardneg_ckpt,$*,gcn)"; \
	$(RUN) deploy/run_modes.py \
	  --mode gcn \
	  --win_dir "$(call win_dir,$*)/$$SPL" \
	  --ckpt_gcn "$$CKPT" \
	  --cfg "$(DEPLOY_CFG)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --time_mode "$(DEPLOY_TIME_MODE)"

deploy-dual-hardneg-%: finetune-hardneg-tcn-% finetune-hardneg-gcn-% $(DEPLOY_CFG)
	@SPL="$(DEPLOY_SPLIT)"; \
	if [ ! -d "$(call win_dir,$*)/$$SPL" ]; then SPL="test"; fi; \
	CKPT_TCN="$(call hardneg_ckpt,$*,tcn)"; \
	CKPT_GCN="$(call hardneg_ckpt,$*,gcn)"; \
	$(RUN) deploy/run_modes.py \
	  --mode dual \
	  --win_dir "$(call win_dir,$*)/$$SPL" \
	  --ckpt_tcn "$$CKPT_TCN" \
	  --ckpt_gcn "$$CKPT_GCN" \
	  --cfg "$(DEPLOY_CFG)" \
	  --prefer_ema "$(PREFER_EMA)" \
	  --time_mode "$(DEPLOY_TIME_MODE)"

# ============================================================
# Meta targets (nice)
# ============================================================
.PHONY: pipeline-% workflow-tcn-% workflow-gcn-% \
        eval-all-tcn-% eval-all-gcn-% replay-all-tcn-% replay-all-gcn-% \
        workflow-full-tcn-% workflow-full-gcn-% workflow-full-%

pipeline-%: windows-%
	@:

workflow-tcn-%: eval-tcn-%
	@:

workflow-gcn-%: eval-gcn-%
	@:

eval-all-tcn-%: fit-ops-tcn-%
	@for o in $(OPS); do $(MAKE) -f "$(THIS_MAKEFILE)" eval-tcn-$* OP=$$o; done

eval-all-gcn-%: fit-ops-gcn-%
	@for o in $(OPS); do $(MAKE) -f "$(THIS_MAKEFILE)" eval-gcn-$* OP=$$o; done

replay-all-tcn-%: fit-ops-tcn-%
	@for o in $(OPS); do $(MAKE) -f "$(THIS_MAKEFILE)" replay-tcn-$* OP=$$o; done

replay-all-gcn-%: fit-ops-gcn-%
	@for o in $(OPS); do $(MAKE) -f "$(THIS_MAKEFILE)" replay-gcn-$* OP=$$o; done

workflow-full-tcn-%: eval-all-tcn-% replay-all-tcn-% mine-nearmiss-tcn-% finetune-hardneg-tcn-% deploy-tcn-% deploy-tcn-hardneg-%
	@:

workflow-full-gcn-%: eval-all-gcn-% replay-all-gcn-% mine-nearmiss-gcn-% finetune-hardneg-gcn-% deploy-gcn-% deploy-gcn-hardneg-%
	@:

workflow-full-%:
	@if [ "$(SKIP_IF_NO_VIDEOS)" = "1" ]; then \
	  if [ -z "$$(find "$(RAW_$*)" -type f \\( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' \\) -print -quit 2>/dev/null)" ] && \
	     [ -z "$$(find "$(RAW_$*)" -type f \\( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \\) -print -quit 2>/dev/null)" ]; then \
	    echo "[skip] $*: no videos/images found under $(RAW_$*)."; \
	    exit 0; \
	  fi; \
	fi
	@$(MAKE) -f "$(THIS_MAKEFILE)" workflow-full-tcn-$*
	@$(MAKE) -f "$(THIS_MAKEFILE)" workflow-full-gcn-$*
	@$(MAKE) -f "$(THIS_MAKEFILE)" deploy-dual-$*
	@$(MAKE) -f "$(THIS_MAKEFILE)" deploy-dual-hardneg-$*






# -------------------------
# UR-Fall prep (optional)
# -------------------------
URFALL_RAW   ?= data/raw/UR_Fall
URFALL_SEQ   ?= data/raw/UR_Fall_seq
URFALL_CLIPS ?= data/raw/UR_Fall_clips
URFALL_DRYRUN ?= 0

.PHONY: urfall-strip-rf urfall-group-seq urfall-merge-clips urfall-prep

urfall-strip-rf:
	$(RUN) tools/urfall_strip_rf.py \
	  --root "$(URFALL_RAW)" \
	  $(if $(filter 1,$(URFALL_DRYRUN)),--dry_run,)

urfall-group-seq: urfall-strip-rf
	$(RUN) tools/urfall_group_by_prefix.py \
	  --in_root "$(URFALL_RAW)" \
	  --out_root "$(URFALL_SEQ)"

urfall-merge-clips: urfall-group-seq
	$(RUN) tools/urfall_merge_seq_splits.py \
	  --in_root "$(URFALL_SEQ)" \
	  --out_root "$(URFALL_CLIPS)"

# One-shot
urfall-prep: urfall-strip-rf urfall-group-seq urfall-merge-clips
	@echo "[OK] UR-Fall prepared at: $(URFALL_CLIPS)"

