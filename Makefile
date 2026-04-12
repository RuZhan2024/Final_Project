# Fall detection v2 — refactored Makefile (DRY, DAG-correct, -j friendly)
# Datasets: le2i, urfd, caucafall, muvim
# Models:   tcn, gcn

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
.SECONDEXPANSION:
.SECONDARY:   # keep intermediate DAG nodes (incl. stamp files)

.DEFAULT_GOAL := help

# -------------------------
# Runtime / Python
# -------------------------
PY      ?= python3
VENV    ?= source ".venv/bin/activate"
PYTHONPATH := $(CURDIR)/ml/src:$(CURDIR)
export PYTHONPATH
RUN := $(VENV) && PYTHONPATH="$(PYTHONPATH)" $(PY)

# -------------------------
# Inventory
# -------------------------
DATASETS := le2i urfd caucafall muvim


# Guard: fail fast if a pattern target is accidentally called with a non-dataset stem (e.g. eval-le2i, gcn-le2i).
REQUIRE_DATASET = $(if $(filter $(1),$(DATASETS)),,$(error Unknown dataset '$(1)'. Valid: $(DATASETS)))


# -------------------------
# Directories
# -------------------------
RAW_DIR    ?= data/raw
INTERIM    ?= data/interim
PROCESSED  ?= data/processed

OUT_DIR    ?= outputs
CFG_DIR    ?= configs
LABELS_DIR ?= $(CFG_DIR)/labels
SPLITS_DIR ?= $(CFG_DIR)/splits
OPS_DIR    ?= $(CFG_DIR)/ops

CAL_DIR    ?= $(OUT_DIR)/calibration
MET_DIR    ?= $(OUT_DIR)/metrics
PLOT_DIR   ?= artifacts/figures/plots

STAMP_DIR  ?= .make

# -------------------------
# Locked reproducibility profiles (caucafall)
# -------------------------
LOCK_TCN_CAUC_TRAIN_DIR ?= $(OUT_DIR)/repro/caucafall_tcn_r1_augreg
LOCK_GCN_CAUC_TRAIN_DIR ?= $(OUT_DIR)/repro/caucafall_gcn_r2_recallpush_b
LOCK_GCN_CAUC_RESUME    ?= $(OUT_DIR)/caucafall_gcn_W$(WIN_W)S$(WIN_S)/best.pt
LOCK_GCN_CAUC_HNEG_LIST ?= $(OUT_DIR)/hardneg/gcn_caucafall_train_p50.txt

LOCK_TCN_CAUC_CKPT ?= $(OUT_DIR)/caucafall_tcn_W$(WIN_W)S$(WIN_S)_r1_ctrl/best.pt
LOCK_GCN_CAUC_CKPT ?= $(OUT_DIR)/caucafall_gcn_W$(WIN_W)S$(WIN_S)_r2_recallpush_b/best.pt
LOCK_TCN_CAUC_OPS  ?= $(OPS_DIR)/tcn_caucafall_r1_ctrl.yaml
LOCK_GCN_CAUC_OPS  ?= $(OPS_DIR)/gcn_caucafall_locked.yaml
LOCK_TCN_CAUC_MET  ?= $(MET_DIR)/tcn_caucafall_locked.json
LOCK_GCN_CAUC_MET  ?= $(MET_DIR)/gcn_caucafall_locked.json

# Optional LE2i paper-comparison profile (scene-scoped start-guard diagnostic)
LOCK_GCN_LE2I_TRAIN_DIR   ?= $(OUT_DIR)/le2i_gcn_W$(WIN_W)S$(WIN_S)_opt33_r8_dataside_noise
LOCK_GCN_LE2I_TRAIN_RESUME ?= $(OUT_DIR)/le2i_gcn_W$(WIN_W)S$(WIN_S)_opt33_r4_recallpush_promoted/best.pt
LOCK_GCN_LE2I_PAPER_CKPT ?= $(OUT_DIR)/le2i_gcn_W$(WIN_W)S$(WIN_S)_opt33_r8_dataside_noise/best.pt
LOCK_GCN_LE2I_PAPER_OPS  ?= $(OPS_DIR)/gcn_le2i_paper_profile.yaml
LOCK_GCN_LE2I_PAPER_MET  ?= $(MET_DIR)/gcn_le2i_opt33_r8_dataside_noise_paperops.json
LOCK_GCN_LE2I_DEPLOY_MET ?= $(MET_DIR)/gcn_le2i_deploy_locked.json

# Optional MUVIM quick locked profile (best-known quick baseline)
LOCK_TCN_MUVIM_CKPT ?= $(OUT_DIR)/muvim_tcn_W$(WIN_W)S$(WIN_S)_quick/best.pt
LOCK_GCN_MUVIM_CKPT ?= $(OUT_DIR)/muvim_gcn_W$(WIN_W)S$(WIN_S)_quick/best.pt
LOCK_TCN_MUVIM_OPS  ?= $(OPS_DIR)/tcn_muvim_quick.yaml
LOCK_GCN_MUVIM_OPS  ?= $(OPS_DIR)/gcn_muvim_quick.yaml
LOCK_TCN_MUVIM_MET  ?= $(MET_DIR)/tcn_muvim_locked.json
LOCK_GCN_MUVIM_MET  ?= $(MET_DIR)/gcn_muvim_locked.json

# Standard per-dataset layout (functions)
pose_raw_dir = $(INTERIM)/$(1)/pose_npz_raw
pose_dir     = $(INTERIM)/$(1)/pose_npz

labels_json  = $(LABELS_DIR)/$(1).json
spans_json   = $(LABELS_DIR)/$(1)_spans.json

split_train  = $(SPLITS_DIR)/$(1)_train.txt
split_val    = $(SPLITS_DIR)/$(1)_val.txt
split_test   = $(SPLITS_DIR)/$(1)_test.txt

# NEW: unlabeled test stems (for "unlabeled evaluation" flows)
split_unlabeled = $(SPLITS_DIR)/$(1)_unlabeled.txt

win_dir      = $(PROCESSED)/$(1)/windows_W$(WIN_W)_S$(WIN_S)
win_eval_dir = $(PROCESSED)/$(1)/windows_eval_W$(WIN_W)_S$(WIN_S)
fa_win_dir   = $(PROCESSED)/$(1)/fa_windows_W$(WIN_W)_S$(WIN_S)

# NEW: unlabeled windows output root
win_unlabeled_dir = $(PROCESSED)/$(1)/windows_unlabeled_W$(WIN_W)_S$(WIN_S)

# Helper: dataset override fallback (VAR_ds overrides VAR)
#   $(call get,LR_TCN,$*) => $(LR_TCN_le2i) if set, else $(LR_TCN)
get = $(or $(strip $($(1)_$(2))),$(strip $($(1))))

# -------------------------
# Global knobs
# -------------------------
SPLIT_SEED ?= 33724876
TRAIN_FRAC ?= 0.80
VAL_FRAC   ?= 0.10
TEST_FRAC  ?= 0.10

WIN_W ?= 48
WIN_S ?= 12
WIN_CLEAN      ?= 1
WIN_EVAL_CLEAN ?= 0

# Pose preprocess knobs
CONF_THR    ?= 0.20
SMOOTH_K    ?= 5
MAX_GAP     ?= 4
NORM_MODE   ?= torso
PELVIS_FILL ?= nearest

# If 1, pose extraction failures won't fail the build
ALLOW_EXTRACT_FAIL ?= 0
EXTRACT_FAIL_GUARD := $(if $(filter 1,$(strip $(ALLOW_EXTRACT_FAIL))),|| true,)

# Suffix for experiment isolation
OUT_TAG ?=

# -------------------------
# Raw dataset roots + FPS
# -------------------------
RAW_le2i      ?= $(RAW_DIR)/LE2i
RAW_urfd      ?= $(RAW_DIR)/UR_Fall_clips
RAW_caucafall ?= $(RAW_DIR)/CAUCAFall
RAW_muvim     ?= $(RAW_DIR)/MUVIM

FPS_le2i      ?= 25
FPS_urfd      ?= 30
FPS_caucafall ?= 23
FPS_muvim     ?= 30

# Image extractor knobs
CAUC_IMAGES_GLOB ?= "$(RAW_caucafall)/**/*.jpg" "$(RAW_caucafall)/**/*.png"
SEQUENCE_ID_DEPTH ?= 2

# -------------------------
# Labels / spans knobs
# -------------------------
USE_PER_FRAME_ACTION_TXT ?= 1

URFD_ANN_GLOB       ?=
URFD_FALL_CLASS_ID  ?= 1
URFD_MIN_RUN        ?= 3
URFD_GAP_FILL       ?= 1

CAUCA_ANN_GLOB      ?= $(RAW_caucafall)/**/*.txt
CAUCA_FALL_CLASS_ID ?= 0
CAUCA_MIN_RUN       ?= 3
CAUCA_GAP_FILL      ?= 2
CAUCA_REQUIRE_SPANS ?= 1

CAUCA_SPLIT_GROUP_MODE ?= caucafall_subject
CAUCA_SPLIT_BALANCE_BY ?= groups
ALLOW_CAUC_NON_SUBJECT_SPLIT ?= 0

MUVIM_ZED_CSV ?= $(RAW_muvim)/ZED_RGB/ZED_RGB.csv

# NEW: unlabeled scene keywords (make_unlabeled_test_list.py requires --scenes)
# Use a generic fallback (.) to select "all" if a dataset doesn't need scenes.
UNLABELED_SCENES ?= .
UNLABELED_SCENES_le2i ?= Office "Lecture room"
UNLABELED_SCENES_urfd ?=
UNLABELED_SCENES_caucafall ?=
UNLABELED_SCENES_muvim ?=

# Unlabeled window generation knobs (kept minimal)
UNLABELED_SUBSET ?= test_unlabeled
UNLABELED_MAX_WINDOWS_PER_VIDEO ?= 400
UNLABELED_MIN_VALID_FRAC ?= 0.00
UNLABELED_MIN_AVG_CONF ?= 0.00
UNLABELED_SKIP_EXISTING ?= 1

# -------------------------
# Window generation knobs
# -------------------------
WIN_EXTRA ?= --strategy balanced --min_overlap_frames 1 --pos_per_span 20 --neg_ratio 2.0 --max_neg_per_video 250 --max_windows_per_video_no_spans 120 --min_valid_frac 0.0 --spans_end_exclusive

WIN_STRATEGY_EVAL ?= all
CONF_GATE ?= 0.20
USE_PRECOMPUTED_MASK ?= 1
WIN_EVAL_EXTRA ?= --strategy "$(WIN_STRATEGY_EVAL)" --min_overlap_frames 1 --min_valid_frac 0.0 --conf_gate "$(CONF_GATE)" $(if $(filter 1,$(strip $(USE_PRECOMPUTED_MASK))),--use_precomputed_mask,) --seed "$(SPLIT_SEED)" --skip_existing

# Optional adapter-driven sequence loading for windows/make_windows.py.
# Defaults are OFF to preserve stable behavior.
ADAPTER_USE ?= 0
ADAPTER_URFALL_TARGET_FPS ?= 25.0
ADAPTER_JOINT_LAYOUT ?= mp33
ADAPTER_DATASET_le2i ?= le2i
ADAPTER_DATASET_urfd ?= urfd
ADAPTER_DATASET_caucafall ?= caucafall
ADAPTER_DATASET_muvim ?= muvim
ADAPTER_FLAGS = $(if $(filter 1,$(strip $(ADAPTER_USE))),--adapter_dataset "$(call get,ADAPTER_DATASET,$*)" --adapter_urfall_target_fps "$(call get,ADAPTER_URFALL_TARGET_FPS,$*)" --adapter_joint_layout "$(ADAPTER_JOINT_LAYOUT)",)

WIN_EXTRA_caucafall      += --require_spans "$(CAUCA_REQUIRE_SPANS)"
WIN_EVAL_EXTRA_caucafall += --require_spans "$(CAUCA_REQUIRE_SPANS)"
WIN_EXTRA_muvim          += --fallback_if_no_span skip_fall
WIN_EVAL_EXTRA_muvim     += --fallback_if_no_span skip_fall   # polished: inherit fallback in eval windows too

# -------------------------
# FA windows
# -------------------------
FA_MODE ?= symlink
FA_ONLY_NEG_VIDEOS ?= 0
FA_SPLIT ?= val

# -------------------------
# Feature config
# -------------------------
CENTER ?= pelvis
FEAT_USE_MOTION ?= 1
FEAT_USE_CONF_CHANNEL ?= 1
FEAT_USE_BONE ?= 1
FEAT_USE_BONE_LEN ?= 1
FEAT_MOTION_SCALE_BY_FPS ?= 1
FEAT_CONF_GATE ?= 0.20
FEAT_USE_PRECOMPUTED_MASK ?= 1

FEAT_FLAGS_TCN = \
  --center "$(CENTER)" \
  --use_motion "$(FEAT_USE_MOTION)" \
  --use_conf_channel "$(FEAT_USE_CONF_CHANNEL)" \
  --use_bone "$(FEAT_USE_BONE)" \
  --use_bone_length "$(FEAT_USE_BONE_LEN)" \
  --motion_scale_by_fps "$(FEAT_MOTION_SCALE_BY_FPS)" \
  --conf_gate "$(FEAT_CONF_GATE)" \
  --use_precomputed_mask "$(FEAT_USE_PRECOMPUTED_MASK)"

FEAT_USE_ANGLES ?= 0
FEAT_INCLUDE_CENTERED ?= 1
FEAT_INCLUDE_ABS ?= 1
FEAT_INCLUDE_VEL ?= 1

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
# Fit-ops feature flags must match eval/fit_ops.py argparse (subset of runtime flags).
FITOPS_FEAT_FLAGS = \
  --center "$(CENTER)" \
  --use_motion "$(FEAT_USE_MOTION)" \
  --use_conf_channel "$(FEAT_USE_CONF_CHANNEL)" \
  --use_bone "$(FEAT_USE_BONE)" \
  --use_bone_length "$(FEAT_USE_BONE_LEN)"


RUNTIME_FEAT_FLAGS = \
  --center "$(CENTER)" \
  --use_motion "$(FEAT_USE_MOTION)" \
  --use_conf_channel "$(FEAT_USE_CONF_CHANNEL)" \
  --use_bone "$(FEAT_USE_BONE)" \
  --use_bone_length "$(FEAT_USE_BONE_LEN)"

# -------------------------
# Training knobs
# -------------------------
EPOCHS ?= 200
BATCH  ?= 128
LR     ?= 1e-3

LR_TCN ?= $(LR)
LR_TCN_muvim ?= 3e-4

TCN_HIDDEN ?= 128
TCN_NUM_BLOCKS ?= 4
TCN_KERNEL ?= 3
TCN_USE_TSM ?= 0
TCN_TSM_FOLD_DIV ?= 8
TCN_PATIENCE ?= 30
TCN_GRAD_CLIP ?= 1.0

TCN_DROPOUT ?= 0.4
TCN_DROPOUT_muvim ?= 0.20

TCN_POS_WEIGHT ?= auto
TCN_POS_WEIGHT_muvim ?= none

TCN_BALANCED_SAMPLER ?= 0
TCN_BALANCED_SAMPLER_muvim ?= 1

TCN_MASK_JOINT_P ?= 0.12
TCN_MASK_JOINT_P_muvim ?= 0.05
TCN_MASK_FRAME_P ?= 0.08
TCN_MASK_FRAME_P_muvim ?= 0.03
TCN_WEIGHT_DECAY ?= 0.0001
TCN_LABEL_SMOOTHING ?= 0.0

TCN_MONITOR ?= ap
GCN_MONITOR ?= ap

TCN_THR_MIN  ?= 0.05
TCN_THR_MAX  ?= 0.95
TCN_THR_STEP ?= 0.01

TCN_LOSS ?= bce
TCN_FOCAL_ALPHA ?= 0.25
TCN_FOCAL_GAMMA ?= 2.0
TCN_RESUME ?=
TCN_HARD_NEG_LIST ?=
TCN_HARD_NEG_MULT ?= 1


TCN_HARD_NEG_PREFIXES ?=
TCN_HARD_NEG_PREFIX_MULT ?= 1

EPOCHS_GCN ?= $(EPOCHS)
BATCH_GCN  ?= $(BATCH)
LR_GCN     ?= $(LR)
LR_GCN_muvim ?= 3e-4

GCN_HIDDEN ?= 96
GCN_NUM_BLOCKS ?= 6
GCN_TEMPORAL_KERNEL ?= 9
GCN_BASE_CHANNELS ?= 48

GCN_DROPOUT ?= 0.35
GCN_DROPOUT_muvim ?= 0.20

GCN_POS_WEIGHT ?= auto
GCN_POS_WEIGHT_muvim ?= none

GCN_BALANCED_SAMPLER ?= 0
GCN_BALANCED_SAMPLER_muvim ?= 1

GCN_TWO_STREAM ?= 1
GCN_FUSE ?= concat
GCN_USE_ADAPTIVE_ADJ ?= 0
GCN_ADAPTIVE_ADJ_EMBED ?= 16

GCN_GRAD_CLIP ?= 1.0
GCN_PATIENCE  ?= 30
GCN_MIN_EPOCHS ?= 5

MASK_JOINT_P ?= 0.15
MASK_JOINT_P_muvim ?= 0.05
MASK_FRAME_P ?= 0.10
MASK_FRAME_P_muvim ?= 0.02

GCN_THR_MIN ?= 0.01
GCN_THR_MAX ?= 0.95
GCN_THR_STEP ?= 0.05
GCN_THR_STEP_muvim ?= 0.01

GCN_LOSS ?= bce
GCN_FOCAL_ALPHA ?= 0.25
GCN_FOCAL_GAMMA ?= 2.0
GCN_RESUME ?=
GCN_HARD_NEG_LIST ?=
GCN_HARD_NEG_MULT ?= 1

# -------------------------
# fit_ops knobs
# -------------------------
FIT_THR_MIN  ?= 0.01
FIT_THR_MAX  ?= 0.95
FIT_THR_STEP ?= 0.01
FIT_TAU_LOW_RATIO ?= 0.78

FIT_TIME_MODE ?= center
FIT_MERGE_GAP_S ?= 1.0
FIT_OVERLAP_SLACK_S ?= 0.5

FIT_OP1_RECALL ?= 0.95
FIT_OP3_FA24H  ?= 1.0
FIT_OP2_OBJECTIVE ?= f1
FIT_COST_FN ?= 5.0
FIT_COST_FP ?= 1.0
FITOPS_ALLOW_DEGENERATE ?= 0
FITOPS_EMIT_ABSOLUTE_PATHS ?= 0

ALERT_EMA_ALPHA ?= 0.20
ALERT_K ?= 2
ALERT_N ?= 3
ALERT_COOLDOWN_S ?= 30
ALERT_CONFIRM ?= 1
ALERT_CONFIRM_S ?= 2.0
ALERT_CONFIRM_MIN_LYING ?= 0.65
ALERT_CONFIRM_MAX_MOTION ?= 0.08
ALERT_CONFIRM_REQUIRE_LOW ?= 1
ALERT_START_GUARD_MAX_LYING ?= -1
ALERT_START_GUARD_PREFIXES ?=

FITOPS_POLICY_FLAGS = \
  --ema_alpha "$(ALERT_EMA_ALPHA)" --k "$(ALERT_K)" --n "$(ALERT_N)" --cooldown_s "$(ALERT_COOLDOWN_S)" \
	--tau_low_ratio "$(FIT_TAU_LOW_RATIO)" \
	--confirm "$(ALERT_CONFIRM)" --confirm_s "$(ALERT_CONFIRM_S)" \
	--confirm_min_lying "$(ALERT_CONFIRM_MIN_LYING)" --confirm_max_motion "$(ALERT_CONFIRM_MAX_MOTION)" \
	--confirm_require_low "$(ALERT_CONFIRM_REQUIRE_LOW)" \
	--start_guard_max_lying "$(ALERT_START_GUARD_MAX_LYING)" \
	--start_guard_prefixes "$(ALERT_START_GUARD_PREFIXES)"

FITOPS_SWEEP_FLAGS = \
  --thr_min "$(FIT_THR_MIN)" --thr_max "$(FIT_THR_MAX)" --thr_step "$(FIT_THR_STEP)" \
  --time_mode "$(strip $(FIT_TIME_MODE))" --merge_gap_s "$(strip $(FIT_MERGE_GAP_S))" --overlap_slack_s "$(strip $(FIT_OVERLAP_SLACK_S))" \
  --op1_recall "$(strip $(FIT_OP1_RECALL))" --op3_fa24h "$(strip $(FIT_OP3_FA24H))" \
  --op2_objective "$(strip $(FIT_OP2_OBJECTIVE))" --cost_fn "$(strip $(FIT_COST_FN))" --cost_fp "$(strip $(FIT_COST_FP))"

FITOPS_GUARD_FLAGS = \
  --allow_degenerate_sweep "$(FITOPS_ALLOW_DEGENERATE)" \
  --emit_absolute_paths "$(FITOPS_EMIT_ABSOLUTE_PATHS)"

FITOPS_PICKER ?= conservative
FITOPS_TIE_BREAK ?= max_thr
FITOPS_TIE_EPS ?= 1e-3
FITOPS_SAVE_SWEEP_JSON ?= 1
FITOPS_PICKER_FLAGS = \
  --ops_picker "$(strip $(FITOPS_PICKER))" \
  --op_tie_break "$(strip $(FITOPS_TIE_BREAK))" \
  --tie_eps "$(strip $(FITOPS_TIE_EPS))" \
  --save_sweep_json "$(strip $(FITOPS_SAVE_SWEEP_JSON))"

FITOPS_MIN_TAU_HIGH ?= 0.20
FITOPS_MIN_TAU_HIGH_CAUC ?= 0.40

# Back-compat hooks (optional): allow old env vars to override without breaking defaults
FITOPS_MIN_TAU_HIGH_le2i      ?= $(or $(strip $(FITOPS_MIN_TAU_HIGH_LE2I)),$(FITOPS_MIN_TAU_HIGH))
FITOPS_MIN_TAU_HIGH_urfd      ?= $(or $(strip $(FITOPS_MIN_TAU_HIGH_URFD)),$(FITOPS_MIN_TAU_HIGH))
FITOPS_MIN_TAU_HIGH_caucafall ?= $(or $(strip $(FITOPS_MIN_TAU_HIGH_CAUC)),$(FITOPS_MIN_TAU_HIGH))
FITOPS_MIN_TAU_HIGH_muvim     ?= $(or $(strip $(FITOPS_MIN_TAU_HIGH_MUVIM)),$(FITOPS_MIN_TAU_HIGH))

FITOPS_USE_FA ?= 0

# Optional per-dataset override of FA dir; otherwise computed from fa_win_dir/$FA_SPLIT
FITOPS_FA_DIR_le2i      ?= $(strip $(FITOPS_FA_DIR_LE2I))
FITOPS_FA_DIR_urfd      ?= $(strip $(FITOPS_FA_DIR_URFD))
FITOPS_FA_DIR_caucafall ?= $(strip $(FITOPS_FA_DIR_CAUC))
FITOPS_FA_DIR_muvim     ?= $(strip $(FITOPS_FA_DIR_MUVIM))

FITOPS_FA_DIR_EFF = $(or $(strip $(FITOPS_FA_DIR_$*)),$(call fa_win_dir,$*)/$(FA_SPLIT))
FITOPS_FA_ARG = $(if $(filter 1,$(strip $(FITOPS_USE_FA))),--fa_dir "$(FITOPS_FA_DIR_EFF)",)

# -------------------------
# metrics sweep
# -------------------------
METR_THR_MIN  ?= 0.001
METR_THR_MAX  ?= 0.95
METR_THR_STEP ?= 0.01
METRICS_SWEEP_FLAGS = --thr_min "$(METR_THR_MIN)" --thr_max "$(METR_THR_MAX)" --thr_step "$(METR_THR_STEP)"

# -------------------------
# unlabeled eval knobs
# -------------------------
UNLABELED_MET_DIR ?= artifacts/reports/hneg_cycle
UNLABELED_SUBSET_EVAL ?= $(UNLABELED_SUBSET)
UNLABELED_CKPT ?=
UNLABELED_OPS_YAML ?=

# -------------------------
# Server
# -------------------------
SERVER_HOST ?= 127.0.0.1
SERVER_PORT ?= 8000

# -------------------------
# Cleaning
# -------------------------
CLEAN_OUT ?= 0   # set to 1 to also remove outputs/

# -------------------------
# Phonies
# -------------------------
.PHONY: help bootstrap-dev up dev stop-dev compose-up compose-down release-check release-manifest serve-dev check-windows pipeline-all pipeline-all-gcn pipeline-all-noextract pipeline-all-gcn-noextract \
        eval-all plot-all eval-all-gcn plot-all-gcn clean clean-stamps

# debug targets (pattern targets should not be declared .PHONY; mark concrete dataset aliases instead)
.PHONY: $(addprefix debug-,$(DATASETS))

# -------------------------
# Utility targets
# -------------------------
bootstrap-dev up:
	@bash ops/scripts/bootstrap_dev.sh

dev:
	@bash ops/scripts/start_fullstack.sh

stop-dev:
	@backend_pids="$$(lsof -ti tcp:8000 2>/dev/null || true)"; \
	frontend_pids="$$(lsof -ti tcp:3000 2>/dev/null || true)"; \
	if [ -z "$$backend_pids$$frontend_pids" ]; then \
	  echo "[dev] no local frontend/backend processes found on ports 3000/8000"; \
	  exit 0; \
	fi; \
	if [ -n "$$backend_pids" ]; then \
	  echo "$$backend_pids" | xargs kill -9; \
	  echo "[dev] stopped backend on port 8000"; \
	fi; \
	if [ -n "$$frontend_pids" ]; then \
	  echo "$$frontend_pids" | xargs kill -9; \
	  echo "[dev] stopped frontend on port 3000"; \
	fi

compose-up:
	@command -v docker >/dev/null 2>&1 || { \
	  echo "docker is not installed or not on PATH. Install and start Docker Desktop first."; \
	  exit 1; \
	}
	@docker compose up

compose-down:
	@command -v docker >/dev/null 2>&1 || { \
	  echo "docker is not installed or not on PATH. Install and start Docker Desktop first."; \
	  exit 1; \
	}
	@docker compose down

release-check:
	@bash ops/scripts/release_doctor.sh

release-manifest:
	@bash ops/scripts/release_manifest.sh

serve-dev:
	$(RUN) -m uvicorn applications.backend.app:app --host "$(SERVER_HOST)" --port "$(SERVER_PORT)" --reload

clean-stamps:
	@echo "[clean] removing stamps: $(STAMP_DIR)/"
	@rm -rf "$(STAMP_DIR)"

clean: clean-stamps
	@echo "[clean] done (stamps removed)"
	@if [ "$(strip $(CLEAN_OUT))" = "1" ]; then \
	  echo "[clean] removing outputs: $(OUT_DIR)/"; \
	  rm -rf "$(OUT_DIR)"; \
	else \
	  echo "[clean] outputs preserved (set CLEAN_OUT=1 to remove $(OUT_DIR)/)"; \
	fi

debug-%:
	@echo "==== debug-$* ===="
	@echo "DATASET=$*"
	@echo "RAW=$(RAW_$*)"
	@echo "FPS=$(FPS_$*)"
	@echo "POSE_RAW_DIR=$(call pose_raw_dir,$*)"
	@echo "POSE_DIR=$(call pose_dir,$*)"
	@echo "LABELS=$(call labels_json,$*)"
	@echo "SPANS=$(call spans_json,$*)"
	@echo "SPLITS: train=$(call split_train,$*) val=$(call split_val,$*) test=$(call split_test,$*)"
	@echo "UNLABELED_SPLIT=$(call split_unlabeled,$*)"
	@echo "WIN_DIR=$(call win_dir,$*)"
	@echo "WIN_EVAL_DIR=$(call win_eval_dir,$*)"
	@echo "WIN_UNLABELED_DIR=$(call win_unlabeled_dir,$*)"
	@echo "FA_WIN_DIR=$(call fa_win_dir,$*)"
	@echo "LR_TCN(resolved)=$(call get,LR_TCN,$*)"
	@echo "LR_GCN(resolved)=$(call get,LR_GCN,$*)"
	@echo "TCN_MONITOR(resolved)=$(call get,TCN_MONITOR,$*)"
	@echo "GCN_MONITOR(resolved)=$(call get,GCN_MONITOR,$*)"
	@echo "TCN_USE_TSM=$(TCN_USE_TSM) TCN_TSM_FOLD_DIV=$(TCN_TSM_FOLD_DIV)"
	@echo "GCN_USE_ADAPTIVE_ADJ=$(GCN_USE_ADAPTIVE_ADJ) GCN_ADAPTIVE_ADJ_EMBED=$(GCN_ADAPTIVE_ADJ_EMBED)"
	@echo "TCN_DROPOUT(resolved)=$(call get,TCN_DROPOUT,$*)"
	@echo "TCN_WEIGHT_DECAY=$(TCN_WEIGHT_DECAY) TCN_LABEL_SMOOTHING=$(TCN_LABEL_SMOOTHING)"
	@echo "GCN_DROPOUT(resolved)=$(call get,GCN_DROPOUT,$*)"
	@echo "SPLIT_MODE=$(if $(filter caucafall,$*),$(CAUCA_SPLIT_GROUP_MODE),n/a)"
	@echo "FITOPS_MIN_TAU_HIGH(resolved)=$(call get,FITOPS_MIN_TAU_HIGH,$*)"
	@echo "FITOPS_USE_FA=$(FITOPS_USE_FA)  FA_SPLIT=$(FA_SPLIT)"
	@echo "FITOPS_FA_DIR_EFF=$(or $(strip $(FITOPS_FA_DIR_$*)),$(call fa_win_dir,$*)/$(FA_SPLIT))"
	@echo "UNLABELED_SCENES(resolved)=$(call get,UNLABELED_SCENES,$*)"
	@echo "ADAPTER_USE=$(ADAPTER_USE) ADAPTER_DATASET(resolved)=$(call get,ADAPTER_DATASET,$*) ADAPTER_URFALL_TARGET_FPS(resolved)=$(call get,ADAPTER_URFALL_TARGET_FPS,$*)"
	@echo "=================="

help:
	@echo ""
	@echo "Fall Detection v2 — targets (datasets: $(DATASETS))"
	@echo ""
	@echo "App dev:"
	@echo "  make bootstrap-dev      (install missing deps, then start backend + frontend)"
	@echo "  make up                 (alias of bootstrap-dev)"
	@echo "  make dev                (start backend + frontend with one command)"
	@echo "  make stop-dev           (stop local backend + frontend on ports 8000/3000)"
	@echo "  make compose-up         (start frontend + backend + MySQL via docker compose)"
	@echo "  make compose-down       (stop docker compose services)"
	@echo "  make release-manifest   (print the current delivery release subset)"
	@echo "  make release-check      (static delivery checks)"
	@echo "  make serve-dev          (start backend only)"
	@echo ""
	@echo "Data prep:"
	@echo "  make pipeline-data-<ds>       (extract→preprocess→labels→splits→windows)"
	@echo "  make pipeline-<ds>-noextract  (preprocess→labels→splits→windows; assumes pose_npz_raw exists)"
	@echo ""
	@echo "Training:"
	@echo "  make train-tcn-<ds> | train-gcn-<ds>"
	@echo "  make train-best-tcn-caucafall | train-best-gcn-caucafall | train-best-caucafall"
	@echo "  make train-best-gcn-le2i-paper"
	@echo ""
	@echo "Eval windows + FA windows:"
	@echo "  make windows-eval-<ds>"
	@echo "  make fa-windows-<ds> [FA_SPLIT=val|train]"
	@echo ""
	@echo "Unlabeled test (stems + windows):"
	@echo "  make splits-unlabeled-<ds>"
	@echo "  make windows-unlabeled-<ds>"
	@echo ""
	@echo "OP fitting / evaluation:"
	@echo "  make fit-ops-<ds> | fit-ops-gcn-<ds> [FITOPS_USE_FA=1]"
	@echo "  make fit-ops-gcn-caucafall-force-ckpt (explicit ckpt-only recalibration alias)"
	@echo "  make repro-best-tcn-caucafall | repro-best-gcn-caucafall | repro-best-caucafall"
	@echo "  make repro-best-tcn-muvim | repro-best-gcn-muvim | repro-best-muvim"
	@echo "  make repro-deploy-gcn-le2i (locked LE2i GCN deploy profile)"
	@echo "  make apply-deploy-ops-le2i-gcn (promote LE2i GCN paper-profile ops to canonical)"
	@echo "  make apply-locked-ops-caucafall (promote locked ops to canonical deploy files)"
	@echo "  knobs: FITOPS_ALLOW_DEGENERATE=0|1 FITOPS_EMIT_ABSOLUTE_PATHS=0|1"
	@echo "  make eval-<ds>    | eval-gcn-<ds>"
	@echo "  make eval-unlabeled-<ds> | eval-unlabeled-gcn-<ds>"
	@echo "  make plot-<ds>    | plot-gcn-<ds>"
	@echo "  make plot-confmat-<ds> | plot-confmat-gcn-<ds>"
	@echo "  make plot-failure-<ds> | plot-failure-gcn-<ds>"
	@echo "  make plot-balance-<ds>"
	@echo ""
	@echo "Adapter windows mode (optional):"
	@echo "  make windows-<ds> ADAPTER_USE=1"
	@echo "  make windows-eval-<ds> ADAPTER_USE=1"
	@echo "  knobs: ADAPTER_DATASET_<ds>, ADAPTER_URFALL_TARGET_FPS"
	@echo ""
	@echo "Debug:"
	@echo "  make debug-<ds>   (print resolved vars for dataset)"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean-stamps (remove $(STAMP_DIR)/ only)"
	@echo "  make clean [CLEAN_OUT=1] (also remove outputs/ if requested)"
	@echo ""
	@echo "Full pipelines (from raw by default):"
	@echo "  make pipeline-<ds>        (TCN: train+fit_ops+eval+plot)"
	@echo "  make pipeline-gcn-<ds>    (GCN: train+fit_ops+eval+plot)"
	@echo ""
	@echo "Single-command auto pipelines (adapter enforced):"
	@echo "  make pipeline-auto-tcn-<ds>  (windows -> windows-eval -> [fa-windows if FITOPS_USE_FA=1] -> train -> fit-ops/eval -> plot)"
	@echo "  make pipeline-auto-gcn-<ds>  (windows -> windows-eval -> [fa-windows if FITOPS_USE_FA=1] -> train -> fit-ops/eval -> plot)"
	@echo "  knob: AUTO_DO_EXTRACT=0|1 (default 0; set 1 to force raw extraction)"
	@echo "  make pipeline-all | pipeline-all-gcn"
	@echo ""
	@echo "Audit gates:"
	@echo "  make audit-smoke"
	@echo "  make audit-ci     (CI-safe: smoke+static+runtime-imports+pytest)"
	@echo "  make audit-static"
	@echo "  make audit-runtime-imports"
	@echo "  make audit-api-contract"
	@echo "  make audit-api-v1-parity"
	@echo "  make audit-api-smoke"
	@echo "  make audit-integration-contract"
	@echo "  make audit-ops-sanity"
	@echo "  make audit-artifact-bundle"
	@echo "  make audit-promoted-profiles"
	@echo "  make audit-numeric [DATASETS='le2i,caucafall']"
	@echo "  make audit-temporal [DATASETS='le2i,caucafall']"
	@echo "  make audit-parity-le2i MODEL=tcn|gcn"
	@echo "  make audit-parity-le2i-strict MODEL=tcn|gcn  (requires performance targets)"
	@echo "  make baseline-capture-le2i MODEL=tcn|gcn     (populate performance baseline from metrics)"
	@echo "  make audit-all MODEL=tcn|gcn                 (runs smoke+static+runtime-imports+numeric+temporal+strict parity)"
	@echo "  make profile-infer PROFILE=cpu_local DS=le2i MODEL=tcn"
	@echo ""

# ============================================================
# DAG nodes (stamps + artifacts)
# ============================================================

# ---------- Extract ----------
extract-%: $(STAMP_DIR)/extract/%.stamp
	@:
$(STAMP_DIR)/extract/%.stamp:
	@mkdir -p "$(@D)" "$(call pose_raw_dir,$*)"
	$(EXTRACT_CMD_$*)
	@touch "$@"

define EXTRACT_CMD_le2i
$(RUN) ops/scripts/extract_pose_videos.py \
  --videos_glob "$(RAW_le2i)/**/Videos/*.avi" "$(RAW_le2i)/**/*.avi" "$(RAW_le2i)/**/Videos/*.mp4" "$(RAW_le2i)/**/*.mp4" \
  --out_dir "$(call pose_raw_dir,le2i)" \
  --fps_default "$(FPS_le2i)" \
  --skip_existing $(EXTRACT_FAIL_GUARD)
endef

define EXTRACT_CMD_urfd
$(RUN) ops/scripts/extract_pose_images.py \
  --images_glob "$(RAW_urfd)/*/*.jpg" "$(RAW_urfd)/*/*.png" \
  --sequence_id_depth "$(SEQUENCE_ID_DEPTH)" \
  --out_dir "$(call pose_raw_dir,urfd)" \
  --dataset urfd \
  --fps "$(FPS_urfd)" \
  --skip_existing
endef

define EXTRACT_CMD_caucafall
$(RUN) ops/scripts/extract_pose_images.py \
  --images_glob $(CAUC_IMAGES_GLOB) \
  --sequence_id_depth "$(SEQUENCE_ID_DEPTH)" \
  --out_dir "$(call pose_raw_dir,caucafall)" \
  --dataset caucafall \
  --fps "$(FPS_caucafall)" \
  --skip_existing
endef

define EXTRACT_CMD_muvim
$(RUN) ops/scripts/extract_pose_images.py \
  --images_glob \
    "$(RAW_muvim)/ZED_RGB/**/*.jpg" "$(RAW_muvim)/ZED_RGB/**/*.png" \
    "$(RAW_muvim)/Fall/**/*.jpg" "$(RAW_muvim)/Fall/**/*.png" \
    "$(RAW_muvim)/NonFall/**/*.jpg" "$(RAW_muvim)/NonFall/**/*.png" \
  --sequence_id_depth "2" \
  --out_dir "$(call pose_raw_dir,muvim)" \
  --dataset muvim \
  --fps "$(FPS_muvim)" \
  --skip_existing
endef

# ---------- Preprocess ----------
DO_EXTRACT ?= 0

preprocess-%: DO_EXTRACT=1
preprocess-%: $(STAMP_DIR)/pose/%.stamp
	@:

preprocess-only-%: DO_EXTRACT=0
preprocess-only-%: $(STAMP_DIR)/pose/%.stamp
	@:

$(STAMP_DIR)/pose/%.stamp: $$(if $$(filter 1,$$(DO_EXTRACT)),$(STAMP_DIR)/extract/$$*.stamp,)
	$(call REQUIRE_DATASET,$*)
	@mkdir -p "$(@D)" "$(call pose_dir,$*)"
	$(RUN) ops/scripts/preprocess_pose.py \
	  --in_dir  "$(call pose_raw_dir,$*)" \
	  --out_dir "$(call pose_dir,$*)" \
	  --recursive --skip_existing \
	  --conf_thr "$(CONF_THR)" --smooth_k "$(SMOOTH_K)" --max_gap "$(MAX_GAP)" \
	  --normalize "$(NORM_MODE)" --pelvis_fill "$(PELVIS_FILL)"
	@touch "$@"

# ---------- Labels ----------
labels-%: $(STAMP_DIR)/labels/%.stamp
	@:
$(STAMP_DIR)/labels/%.stamp: $(STAMP_DIR)/pose/%.stamp
	@mkdir -p "$(@D)" "$(LABELS_DIR)"
	$(LABEL_CMD_$*)
	@touch "$@"

URFD_ANN_ARGS = $(if $(strip $(URFD_ANN_GLOB)),--ann_glob "$(URFD_ANN_GLOB)" --use_per_frame_action_txt "$(USE_PER_FRAME_ACTION_TXT)" --fall_class_id "$(URFD_FALL_CLASS_ID)" --min_run "$(URFD_MIN_RUN)" --gap_fill "$(URFD_GAP_FILL)",)
CAUCA_SPAN_ARGS = $(if $(filter 1,$(strip $(USE_PER_FRAME_ACTION_TXT))),--use_per_frame_action_txt 1 --fall_class_id "$(CAUCA_FALL_CLASS_ID)" --min_run "$(CAUCA_MIN_RUN)" --gap_fill "$(CAUCA_GAP_FILL)" --frame_label_mode auto --clamp_to_npz_len,)

define LABEL_CMD_le2i
$(RUN) ops/scripts/make_labels_le2i.py \
  --npz_dir "$(call pose_dir,le2i)" \
  --raw_root "$(RAW_le2i)" \
  --out_labels "$(call labels_json,le2i)" \
  --out_spans  "$(call spans_json,le2i)"
endef

define LABEL_CMD_urfd
$(RUN) ops/scripts/make_labels_urfall.py \
  --npz_dir "$(call pose_dir,urfd)" \
  --out_labels "$(call labels_json,urfd)" \
  --out_spans  "$(call spans_json,urfd)" \
  $(URFD_ANN_ARGS)
endef

define LABEL_CMD_caucafall
$(RUN) ops/scripts/make_labels_caucafall.py \
  --raw_root "$(RAW_caucafall)" \
  --npz_dir "$(call pose_dir,caucafall)" \
  --out_labels "$(call labels_json,caucafall)" \
  --out_spans  "$(call spans_json,caucafall)" \
  $(CAUCA_SPAN_ARGS) \
  --verbose
endef

MUVIM_ZED_ARGS = $(if $(strip $(MUVIM_ZED_CSV)),--zed_csv "$(MUVIM_ZED_CSV)",)

define LABEL_CMD_muvim
$(RUN) ops/scripts/make_labels_muvim.py \
  --npz_dir "$(call pose_dir,muvim)" \
  $(MUVIM_ZED_ARGS) \
  --out_labels "$(call labels_json,muvim)" \
  --out_spans  "$(call spans_json,muvim)" \
  --stop_inclusive
endef

# ---------- Split lists ----------
splits-%: $(STAMP_DIR)/splits/%.stamp
	@:

SPLIT_GUARD_caucafall = $(if $(and $(filter-out 1,$(strip $(ALLOW_CAUC_NON_SUBJECT_SPLIT))),$(filter-out caucafall_subject,$(strip $(CAUCA_SPLIT_GROUP_MODE)))),\
  $(error CAUCAFall requires subject-independent splits: CAUCA_SPLIT_GROUP_MODE=caucafall_subject (override with ALLOW_CAUC_NON_SUBJECT_SPLIT=1)),)

SPLIT_EXTRA_caucafall = --group_mode "$(CAUCA_SPLIT_GROUP_MODE)" --balance_by "$(CAUCA_SPLIT_BALANCE_BY)"
SPLIT_EXTRA_le2i =
SPLIT_EXTRA_urfd =
SPLIT_EXTRA_muvim =

$(STAMP_DIR)/splits/%.stamp: $(STAMP_DIR)/labels/%.stamp
	@mkdir -p "$(@D)" "$(SPLITS_DIR)"
	$(SPLIT_GUARD_$*)
	$(RUN) ops/scripts/make_splits.py \
	  --labels_json "$(call labels_json,$*)" \
	  --out_dir "$(SPLITS_DIR)" --prefix "$*" \
	  $(SPLIT_EXTRA_$*) \
	  --train "$(TRAIN_FRAC)" --val "$(VAL_FRAC)" --test "$(TEST_FRAC)" \
	  --seed "$(SPLIT_SEED)" \
	  --summary_json "$(SPLITS_DIR)/$*_split_summary.json"
	@touch "$@"

# ---------- Unlabeled split list (NEW) ----------
# IMPORTANT: This is a *different* target name ("splits-unlabeled-%") so it cannot be mistaken for "splits-%".
splits-unlabeled-%: $(STAMP_DIR)/splits_unlabeled/%.stamp
	@:

$(STAMP_DIR)/splits_unlabeled/%.stamp: $(STAMP_DIR)/pose/%.stamp
	@mkdir -p "$(@D)" "$(SPLITS_DIR)"
	$(RUN) ops/scripts/make_unlabeled_test_list.py \
	  --npz_dir "$(call pose_dir,$*)" \
	  --out "$(call split_unlabeled,$*)" \
	  --scenes $(call get,UNLABELED_SCENES,$*)
	@touch "$@"

# ---------- Spans sanity check ----------
check-spans-%: $(STAMP_DIR)/check_spans/%.stamp
	@:
$(STAMP_DIR)/check_spans/%.stamp: $(STAMP_DIR)/labels/%.stamp
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/check_spans.py --labels_json "$(call labels_json,$*)" --spans_json "$(call spans_json,$*)" --require_nonempty 1
	@touch "$@"

WINDOW_PREREQ_caucafall = $(STAMP_DIR)/check_spans/caucafall.stamp
WINDOW_PREREQ_le2i =
WINDOW_PREREQ_urfd =
WINDOW_PREREQ_muvim =

# ---------- Dataset-scoped front-door targets (avoid pattern collisions)
# Without these, targets like `windows-eval-caucafall` can accidentally match `windows-%`
# with stem `eval-caucafall`, causing dataset-guard failures.
.PHONY: $(addprefix windows-eval-,$(DATASETS)) $(addprefix splits-unlabeled-,$(DATASETS)) $(addprefix windows-unlabeled-,$(DATASETS))

# Explicit dataset lists → correct stems (le2i/urfd/caucafall/muvim), never `eval-<ds>` or `unlabeled-<ds>`.
$(addprefix windows-eval-,$(DATASETS)): windows-eval-%: $(STAMP_DIR)/windows_eval/%.stamp
$(addprefix splits-unlabeled-,$(DATASETS)): splits-unlabeled-%: $(STAMP_DIR)/splits_unlabeled/%.stamp
$(addprefix windows-unlabeled-,$(DATASETS)): windows-unlabeled-%: $(STAMP_DIR)/windows_unlabeled/%.stamp

# ---------- Training windows ----------
windows-%: $(STAMP_DIR)/windows/%.stamp
	@:
$(STAMP_DIR)/windows/%.stamp: $(STAMP_DIR)/splits/%.stamp $$(WINDOW_PREREQ_$$*)
	@mkdir -p "$(@D)" "$(call win_dir,$*)"
	@if [ "$(WIN_CLEAN)" = "1" ]; then rm -rf "$(call win_dir,$*)/train" "$(call win_dir,$*)/val" "$(call win_dir,$*)/test" || true; fi
	$(RUN) ops/scripts/make_windows.py \
	  --npz_dir "$(call pose_dir,$*)" \
	  --labels_json "$(call labels_json,$*)" \
	  --spans_json  "$(call spans_json,$*)" \
	  --out_dir "$(call win_dir,$*)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_$*)" \
	  $(ADAPTER_FLAGS) \
	  --train_list "$(call split_train,$*)" --val_list "$(call split_val,$*)" --test_list "$(call split_test,$*)" \
	  $(WIN_EXTRA) $(WIN_EXTRA_$*)
	@touch "$@"

# ---------- Eval windows ----------
# This rule must exist so "windows-eval-<ds>" does NOT fall back to "windows-% (stem=eval-<ds>)".
windows-eval-%: $(STAMP_DIR)/windows_eval/%.stamp
	@:
$(STAMP_DIR)/windows_eval/%.stamp: $(STAMP_DIR)/splits/%.stamp $$(WINDOW_PREREQ_$$*)
	@mkdir -p "$(@D)" "$(call win_eval_dir,$*)"
	@if [ "$(WIN_EVAL_CLEAN)" = "1" ]; then rm -rf "$(call win_eval_dir,$*)/train" "$(call win_eval_dir,$*)/val" "$(call win_eval_dir,$*)/test" || true; fi
	$(RUN) ops/scripts/make_windows.py \
	  --npz_dir "$(call pose_dir,$*)" \
	  --labels_json "$(call labels_json,$*)" \
	  --spans_json  "$(call spans_json,$*)" \
	  --out_dir "$(call win_eval_dir,$*)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_$*)" \
	  $(ADAPTER_FLAGS) \
	  --train_list "$(call split_train,$*)" --val_list "$(call split_val,$*)" --test_list "$(call split_test,$*)" \
	  $(WIN_EVAL_EXTRA) $(WIN_EVAL_EXTRA_$*)
	@touch "$@"

# ---------- Unlabeled windows (NEW) ----------
windows-unlabeled-%: $(STAMP_DIR)/windows_unlabeled/%.stamp
	@:

$(STAMP_DIR)/windows_unlabeled/%.stamp: $(STAMP_DIR)/splits_unlabeled/%.stamp
	@mkdir -p "$(@D)" "$(call win_unlabeled_dir,$*)"
	$(RUN) ops/scripts/make_unlabeled_windows.py \
	  --npz_dir "$(call pose_dir,$*)" \
	  --stems_txt "$(call split_unlabeled,$*)" \
	  --out_dir "$(call win_unlabeled_dir,$*)" \
	  --W "$(WIN_W)" --stride "$(WIN_S)" --fps_default "$(FPS_$*)" \
	  $(ADAPTER_FLAGS) \
	  --subset "$(UNLABELED_SUBSET)" --seed "$(SPLIT_SEED)" \
	  --max_windows_per_video "$(UNLABELED_MAX_WINDOWS_PER_VIDEO)" \
	  --conf_gate "$(CONF_GATE)" \
	  $(if $(filter 1,$(strip $(USE_PRECOMPUTED_MASK))),--use_precomputed_mask,) \
	  --min_valid_frac "$(UNLABELED_MIN_VALID_FRAC)" --min_avg_conf "$(UNLABELED_MIN_AVG_CONF)" \
	  $(if $(filter 1,$(strip $(UNLABELED_SKIP_EXISTING))),--skip_existing,)
	@touch "$@"

# ---------- FA windows ----------
fa-windows-%: $(STAMP_DIR)/fa_windows/%.stamp
	@:
$(STAMP_DIR)/fa_windows/%.stamp: $(STAMP_DIR)/windows_eval/%.stamp
	@mkdir -p "$(@D)" "$(call fa_win_dir,$*)"
	$(RUN) ops/scripts/make_fa_windows.py \
	  --in_root  "$(call win_eval_dir,$*)" \
	  --out_root "$(call fa_win_dir,$*)" \
	  --split "$(FA_SPLIT)" \
	  --mode "$(strip $(FA_MODE))" \
	  --only_neg_videos "$(FA_ONLY_NEG_VIDEOS)"
	@touch "$@"

# ============================================================
# Train artifacts
# ============================================================
.PHONY: train-best-tcn-caucafall train-best-gcn-caucafall train-best-caucafall train-best-gcn-le2i-paper

train-best-tcn-caucafall: $(LOCK_TCN_CAUC_TRAIN_DIR)/best.pt
	@echo "[ok] locked TCN training finished: $(LOCK_TCN_CAUC_TRAIN_DIR)/best.pt"

train-best-gcn-caucafall: $(LOCK_GCN_CAUC_TRAIN_DIR)/best.pt
	@echo "[ok] locked GCN training finished: $(LOCK_GCN_CAUC_TRAIN_DIR)/best.pt"

train-best-caucafall: train-best-tcn-caucafall train-best-gcn-caucafall
	@echo "[ok] locked TCN+GCN training finished for caucafall."

train-best-gcn-le2i-paper: $(LOCK_GCN_LE2I_TRAIN_DIR)/best.pt
	@echo "[ok] locked LE2i GCN training finished: $(LOCK_GCN_LE2I_TRAIN_DIR)/best.pt"

$(LOCK_TCN_CAUC_TRAIN_DIR)/best.pt: $(STAMP_DIR)/windows/caucafall.stamp
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/train_tcn.py --train_dir "$(call win_dir,caucafall)/train" --val_dir "$(call win_dir,caucafall)/val" \
	  --epochs "200" --batch "128" --lr "0.001" --seed "$(SPLIT_SEED)" --fps_default "$(FPS_caucafall)" \
	  --center "pelvis" --use_motion "1" --use_conf_channel "1" --use_bone "1" --use_bone_length "1" --motion_scale_by_fps "1" --conf_gate "0.20" --use_precomputed_mask "1" \
	  --loss "bce" --focal_alpha "0.25" --focal_gamma "2.0" \
	  --hidden "128" --num_blocks "4" --kernel "3" \
	  --use_tsm "0" --tsm_fold_div "8" \
	  --grad_clip "1.0" --patience "30" --min_epochs "0" \
	  --thr_min "0.05" --thr_max "0.95" --thr_step "0.01" \
	  --monitor "ap" \
	  --dropout "0.40" \
	  --mask_joint_p "0.12" --mask_frame_p "0.08" \
	  --weight_decay "1e-4" --label_smoothing "0.0" \
	  --pos_weight "auto" \
	  --num_workers "0" --deterministic "1" \
	  --save_dir "$(LOCK_TCN_CAUC_TRAIN_DIR)"
	@test -f "$@"

$(LOCK_GCN_CAUC_TRAIN_DIR)/best.pt: $(STAMP_DIR)/windows/caucafall.stamp
	@mkdir -p "$(@D)"
	@test -f "$(LOCK_GCN_CAUC_RESUME)" || (echo "[ERR] missing locked GCN resume ckpt: $(LOCK_GCN_CAUC_RESUME)" && exit 1)
	@test -f "$(LOCK_GCN_CAUC_HNEG_LIST)" || (echo "[ERR] missing locked GCN hard-neg list: $(LOCK_GCN_CAUC_HNEG_LIST)" && exit 1)
	$(RUN) ops/scripts/train_gcn.py --train_dir "$(call win_dir,caucafall)/train" --val_dir "$(call win_dir,caucafall)/val" \
	  --epochs "80" --batch "128" --lr "0.0003" --seed "$(SPLIT_SEED)" --fps_default "$(FPS_caucafall)" \
	  --center "pelvis" \
	  --use_motion "1" --use_conf "1" --use_bone "1" --use_bonelen "1" --motion_scale_by_fps "1" --conf_gate "0.20" --use_precomputed_mask "1" \
	  --loss "bce" --focal_alpha "0.25" --focal_gamma "2.0" \
	  --hidden "96" --dropout "0.22" \
	  --num_blocks "6" --temporal_kernel "9" --base_channels "48" \
	  --two_stream "1" --fuse "concat" --use_adaptive_adj "0" --adaptive_adj_embed "16" \
	  --grad_clip "1.0" --patience "18" --min_epochs "10" \
	  --thr_min "0.01" --thr_max "0.95" --thr_step "0.05" \
	  --monitor "ap" \
	  --mask_joint_p "0.05" --mask_frame_p "0.03" \
	  --weight_decay "2e-4" --label_smoothing "0.0" \
	  --pos_weight "auto" \
	  --resume "$(LOCK_GCN_CAUC_RESUME)" \
	  --hard_neg_list "$(LOCK_GCN_CAUC_HNEG_LIST)" --hard_neg_mult "1" \
	  --num_workers "0" --prefetch_factor "2" --persistent_workers "1" --deterministic "1" \
	  --save_dir "$(LOCK_GCN_CAUC_TRAIN_DIR)"
	@test -f "$@"

$(LOCK_GCN_LE2I_TRAIN_DIR)/best.pt: $(STAMP_DIR)/windows/le2i.stamp
	@mkdir -p "$(@D)"
	@test -f "$(LOCK_GCN_LE2I_TRAIN_RESUME)" || (echo "[ERR] missing locked LE2i GCN resume ckpt: $(LOCK_GCN_LE2I_TRAIN_RESUME)" && exit 1)
	$(RUN) ops/scripts/train_gcn.py --train_dir "$(call win_dir,le2i)/train" --val_dir "$(call win_dir,le2i)/val" \
	  --epochs "45" --min_epochs "8" --batch "128" --lr "3e-4" --seed "$(SPLIT_SEED)" --fps_default "$(FPS_le2i)" \
	  --center "pelvis" \
	  --use_motion "1" --use_conf "1" --use_bone "1" --use_bonelen "1" \
	  --loss "bce" --monitor "ap" --balanced_sampler --pos_weight "none" \
	  --dropout "0.18" \
	  --num_blocks "6" --temporal_kernel "9" --base_channels "48" \
	  --two_stream "1" --fuse "concat" --use_adaptive_adj "0" --adaptive_adj_embed "16" \
	  --mask_joint_p "0.00" --mask_frame_p "0.00" \
	  --x_noise_std "0.01" --x_quant_step "0.002" --temporal_dropout_p "0.00" \
	  --grad_clip "1.0" --patience "8" \
	  --thr_min "0.01" --thr_max "0.95" --thr_step "0.05" \
	  --resume "$(LOCK_GCN_LE2I_TRAIN_RESUME)" \
	  --use_ema "1" --ema_decay "0.999" \
	  --hidden "96" \
	  --num_workers "0" --prefetch_factor "2" --persistent_workers "1" --deterministic "1" \
	  --save_dir "$(LOCK_GCN_LE2I_TRAIN_DIR)"
	@test -f "$@"

train-tcn-%: $(OUT_DIR)/%_tcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt
	@:
train-gcn-%: $(OUT_DIR)/%_gcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt
	@:

# NOTE: fixed evaluation-order bug by inlining $(call get,...) inside Make (no bash vars for flags)
$(OUT_DIR)/%_tcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt: $(STAMP_DIR)/windows/%.stamp
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/train_tcn.py --train_dir "$(call win_dir,$*)/train" --val_dir "$(call win_dir,$*)/val" \
	  --epochs "$(EPOCHS)" --batch "$(BATCH)" --lr "$(call get,LR_TCN,$*)" --seed "$(SPLIT_SEED)" --fps_default "$(FPS_$*)" \
	  $(FEAT_FLAGS_TCN) \
	  $(if $(strip $(TCN_RESUME)),--resume "$(strip $(TCN_RESUME))",) \
	  $(if $(strip $(TCN_HARD_NEG_LIST)),--hard_neg_list "$(strip $(TCN_HARD_NEG_LIST))",) \
	  --hard_neg_mult "$(TCN_HARD_NEG_MULT)" \
	  --hard_neg_prefixes "$(TCN_HARD_NEG_PREFIXES)" \
	  --hard_neg_prefix_mult "$(TCN_HARD_NEG_PREFIX_MULT)" \
	  --loss "$(TCN_LOSS)" --focal_alpha "$(TCN_FOCAL_ALPHA)" --focal_gamma "$(TCN_FOCAL_GAMMA)" \
	  --hidden "$(TCN_HIDDEN)" --num_blocks "$(TCN_NUM_BLOCKS)" --kernel "$(TCN_KERNEL)" \
	  --use_tsm "$(TCN_USE_TSM)" --tsm_fold_div "$(TCN_TSM_FOLD_DIV)" \
	  --grad_clip "$(TCN_GRAD_CLIP)" --patience "$(TCN_PATIENCE)" \
	  --thr_min "$(TCN_THR_MIN)" --thr_max "$(TCN_THR_MAX)" --thr_step "$(TCN_THR_STEP)" \
	  --monitor "$(call get,TCN_MONITOR,$*)" \
	  --dropout "$(call get,TCN_DROPOUT,$*)" \
	  --mask_joint_p "$(call get,TCN_MASK_JOINT_P,$*)" --mask_frame_p "$(call get,TCN_MASK_FRAME_P,$*)" \
	  --weight_decay "$(TCN_WEIGHT_DECAY)" --label_smoothing "$(TCN_LABEL_SMOOTHING)" \
	  --pos_weight "$(call get,TCN_POS_WEIGHT,$*)" $(if $(filter 1,$(call get,TCN_BALANCED_SAMPLER,$*)),--balanced_sampler,) \
	  --save_dir "$(OUT_DIR)/$*_tcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)"
	@test -f "$@"

$(OUT_DIR)/%_gcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt: $(STAMP_DIR)/windows/%.stamp
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/train_gcn.py --train_dir "$(call win_dir,$*)/train" --val_dir "$(call win_dir,$*)/val" \
	  --epochs "$(EPOCHS_GCN)" --batch "$(BATCH_GCN)" --lr "$(call get,LR_GCN,$*)" --seed "$(SPLIT_SEED)" --fps_default "$(FPS_$*)" \
	  $(FEAT_FLAGS_GCN) \
	  $(if $(strip $(GCN_RESUME)),--resume "$(strip $(GCN_RESUME))",) \
	  $(if $(strip $(GCN_HARD_NEG_LIST)),--hard_neg_list "$(strip $(GCN_HARD_NEG_LIST))",) \
	  --hard_neg_mult "$(GCN_HARD_NEG_MULT)" \
	  --loss "$(GCN_LOSS)" --focal_alpha "$(GCN_FOCAL_ALPHA)" --focal_gamma "$(GCN_FOCAL_GAMMA)" \
	  --hidden "$(GCN_HIDDEN)" \
	  --num_blocks "$(GCN_NUM_BLOCKS)" --temporal_kernel "$(GCN_TEMPORAL_KERNEL)" --base_channels "$(GCN_BASE_CHANNELS)" \
	  --two_stream "$(GCN_TWO_STREAM)" --fuse "$(GCN_FUSE)" \
	  --use_adaptive_adj "$(GCN_USE_ADAPTIVE_ADJ)" --adaptive_adj_embed "$(GCN_ADAPTIVE_ADJ_EMBED)" \
	  --grad_clip "$(GCN_GRAD_CLIP)" --patience "$(GCN_PATIENCE)" --min_epochs "$(GCN_MIN_EPOCHS)" \
	  --monitor "$(call get,GCN_MONITOR,$*)" \
	  --mask_joint_p "$(call get,MASK_JOINT_P,$*)" --mask_frame_p "$(call get,MASK_FRAME_P,$*)" \
	  --thr_min "$(GCN_THR_MIN)" --thr_max "$(GCN_THR_MAX)" --thr_step "$(call get,GCN_THR_STEP,$*)" \
	  --dropout "$(call get,GCN_DROPOUT,$*)" \
	  --pos_weight "$(call get,GCN_POS_WEIGHT,$*)" $(if $(filter 1,$(call get,GCN_BALANCED_SAMPLER,$*)),--balanced_sampler,) \
	  --save_dir "$(OUT_DIR)/$*_gcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)"
	@test -f "$@"

# ============================================================


# Dataset-scoped front-door targets (avoid pattern collisions like treating 'gcn-le2i' as a dataset).
.PHONY: $(addprefix fit-ops-,$(DATASETS)) $(addprefix fit-ops-gcn-,$(DATASETS))

$(addprefix fit-ops-,$(DATASETS)): fit-ops-%: $(OPS_DIR)/tcn_%$(OUT_TAG).yaml
	@:

$(addprefix fit-ops-gcn-,$(DATASETS)): fit-ops-gcn-%: $(OPS_DIR)/gcn_%$(OUT_TAG).yaml
	@:

# Explicit alias: recalibrate CAUCAFall GCN using existing checkpoint only.
.PHONY: fit-ops-gcn-caucafall-force-ckpt
fit-ops-gcn-caucafall-force-ckpt:
	@$(MAKE) -B fit-ops-gcn-caucafall ADAPTER_USE="$(ADAPTER_USE)" OUT_TAG="$(OUT_TAG)"

.PHONY: repro-best-tcn-caucafall repro-best-gcn-caucafall repro-best-caucafall apply-locked-ops-caucafall repro-best-gcn-le2i-paper repro-deploy-gcn-le2i apply-deploy-ops-le2i-gcn repro-best-tcn-muvim repro-best-gcn-muvim repro-best-muvim
repro-best-tcn-caucafall: $(LOCK_TCN_CAUC_OPS) $(LOCK_TCN_CAUC_MET)
	@echo "[ok] tcn locked profile reproduced:"
	@echo "     ops=$(LOCK_TCN_CAUC_OPS)"
	@echo "     metrics=$(LOCK_TCN_CAUC_MET)"

repro-best-gcn-caucafall: $(LOCK_GCN_CAUC_OPS) $(LOCK_GCN_CAUC_MET)
	@echo "[ok] gcn locked profile reproduced:"
	@echo "     ops=$(LOCK_GCN_CAUC_OPS)"
	@echo "     metrics=$(LOCK_GCN_CAUC_MET)"

repro-best-caucafall: repro-best-tcn-caucafall repro-best-gcn-caucafall
	@echo "[ok] caucafall locked TCN+GCN profiles reproduced."

repro-best-tcn-muvim: $(LOCK_TCN_MUVIM_MET)
	@echo "[ok] muvim locked TCN profile reproduced:"
	@echo "     ops=$(LOCK_TCN_MUVIM_OPS)"
	@echo "     metrics=$(LOCK_TCN_MUVIM_MET)"

repro-best-gcn-muvim: $(LOCK_GCN_MUVIM_MET)
	@echo "[ok] muvim locked GCN profile reproduced:"
	@echo "     ops=$(LOCK_GCN_MUVIM_OPS)"
	@echo "     metrics=$(LOCK_GCN_MUVIM_MET)"

repro-best-muvim: repro-best-tcn-muvim repro-best-gcn-muvim
	@echo "[ok] muvim locked TCN+GCN profiles reproduced."

repro-best-gcn-le2i-paper: $(LOCK_GCN_LE2I_PAPER_MET)
	@echo "[ok] le2i gcn paper profile reproduced:"
	@echo "     ops=$(LOCK_GCN_LE2I_PAPER_OPS)"
	@echo "     metrics=$(LOCK_GCN_LE2I_PAPER_MET)"

repro-deploy-gcn-le2i: repro-best-gcn-le2i-paper
	@echo "[ok] LE2i GCN deployment profile is locked to paper_profile."

apply-deploy-ops-le2i-gcn: repro-best-gcn-le2i-paper $(LOCK_GCN_LE2I_DEPLOY_MET)
	@cp "$(LOCK_GCN_LE2I_PAPER_OPS)" "$(OPS_DIR)/gcn_le2i.yaml"
	@echo "[ok] canonical LE2i GCN deploy ops updated:"
	@echo "     $(OPS_DIR)/gcn_le2i.yaml"
	@echo "     $(LOCK_GCN_LE2I_DEPLOY_MET)"

apply-locked-ops-caucafall: repro-best-caucafall
	@cp "$(LOCK_TCN_CAUC_OPS)" "$(OPS_DIR)/tcn_caucafall.yaml"
	@cp "$(LOCK_GCN_CAUC_OPS)" "$(OPS_DIR)/gcn_caucafall.yaml"
	@echo "[ok] canonical ops updated:"
	@echo "     $(OPS_DIR)/tcn_caucafall.yaml"
	@echo "     $(OPS_DIR)/gcn_caucafall.yaml"

$(LOCK_GCN_LE2I_PAPER_MET): $(LOCK_GCN_LE2I_PAPER_OPS)
	@mkdir -p "$(@D)"
	@test -f "$(LOCK_GCN_LE2I_PAPER_CKPT)" || (echo "[ERR] missing LE2i paper-profile ckpt: $(LOCK_GCN_LE2I_PAPER_CKPT)" && exit 1)
	@test -f "$(LOCK_GCN_LE2I_PAPER_OPS)" || (echo "[ERR] missing LE2i paper-profile ops: $(LOCK_GCN_LE2I_PAPER_OPS)" && exit 1)
	$(RUN) ops/scripts/eval_metrics.py \
	  --win_dir "$(call win_eval_dir,le2i)/test" \
	  --ckpt "$(LOCK_GCN_LE2I_PAPER_CKPT)" \
	  --ops_yaml "$(LOCK_GCN_LE2I_PAPER_OPS)" \
	  --out_json "$@" \
	  --fps_default "$(FPS_le2i)" \
	  $(METRICS_SWEEP_FLAGS)

$(LOCK_GCN_LE2I_DEPLOY_MET): $(LOCK_GCN_LE2I_PAPER_OPS)
	@mkdir -p "$(@D)"
	@test -f "$(LOCK_GCN_LE2I_PAPER_CKPT)" || (echo "[ERR] missing LE2i deploy ckpt: $(LOCK_GCN_LE2I_PAPER_CKPT)" && exit 1)
	$(RUN) ops/scripts/eval_metrics.py \
	  --win_dir "$(call win_eval_dir,le2i)/test" \
	  --ckpt "$(LOCK_GCN_LE2I_PAPER_CKPT)" \
	  --ops_yaml "$(LOCK_GCN_LE2I_PAPER_OPS)" \
	  --out_json "$@" \
	  --fps_default "$(FPS_le2i)" \
	  $(METRICS_SWEEP_FLAGS)

$(LOCK_TCN_MUVIM_MET): $(LOCK_TCN_MUVIM_OPS)
	@mkdir -p "$(@D)"
	@test -f "$(LOCK_TCN_MUVIM_CKPT)" || (echo "[ERR] missing MUVIM locked TCN ckpt: $(LOCK_TCN_MUVIM_CKPT)" && exit 1)
	@test -f "$(LOCK_TCN_MUVIM_OPS)" || (echo "[ERR] missing MUVIM locked TCN ops: $(LOCK_TCN_MUVIM_OPS)" && exit 1)
	$(RUN) ops/scripts/eval_metrics.py \
	  --win_dir "$(call win_eval_dir,muvim)/test" \
	  --ckpt "$(LOCK_TCN_MUVIM_CKPT)" \
	  --ops_yaml "$(LOCK_TCN_MUVIM_OPS)" \
	  --out_json "$@" \
	  --fps_default "$(FPS_muvim)" \
	  $(METRICS_SWEEP_FLAGS)

$(LOCK_GCN_MUVIM_MET): $(LOCK_GCN_MUVIM_OPS)
	@mkdir -p "$(@D)"
	@test -f "$(LOCK_GCN_MUVIM_CKPT)" || (echo "[ERR] missing MUVIM locked GCN ckpt: $(LOCK_GCN_MUVIM_CKPT)" && exit 1)
	@test -f "$(LOCK_GCN_MUVIM_OPS)" || (echo "[ERR] missing MUVIM locked GCN ops: $(LOCK_GCN_MUVIM_OPS)" && exit 1)
	$(RUN) ops/scripts/eval_metrics.py \
	  --win_dir "$(call win_eval_dir,muvim)/test" \
	  --ckpt "$(LOCK_GCN_MUVIM_CKPT)" \
	  --ops_yaml "$(LOCK_GCN_MUVIM_OPS)" \
	  --out_json "$@" \
	  --fps_default "$(FPS_muvim)" \
	  $(METRICS_SWEEP_FLAGS)

$(LOCK_TCN_CAUC_OPS): $(STAMP_DIR)/windows_eval/caucafall.stamp
	@mkdir -p "$(@D)" "$(MET_DIR)"
	@test -f "$(LOCK_TCN_CAUC_CKPT)" || (echo "[ERR] missing locked TCN ckpt: $(LOCK_TCN_CAUC_CKPT)" && exit 1)
	@if [ "$@" = "$(OPS_DIR)/tcn_caucafall_r1_ctrl.yaml" ]; then \
	  test -f "$@" || (echo "[ERR] missing precomputed locked TCN ops: $@" && exit 1); \
	  echo "[ok] using precomputed locked TCN ops: $@"; \
	else \
	  $(RUN) ops/scripts/fit_ops.py --arch tcn \
	    --val_dir "$(call win_eval_dir,caucafall)/val" \
	    --ckpt "$(LOCK_TCN_CAUC_CKPT)" \
	    --out "$@" \
	    --fps_default "$(FPS_caucafall)" \
	    --center "$(CENTER)" --use_motion "$(FEAT_USE_MOTION)" --use_conf_channel "$(FEAT_USE_CONF_CHANNEL)" --use_bone "$(FEAT_USE_BONE)" --use_bone_length "$(FEAT_USE_BONE_LEN)" \
	    --ema_alpha "0.20" --k "2" --n "3" --cooldown_s "30" --tau_low_ratio "0.78" --confirm "0" --confirm_s "2.0" --confirm_min_lying "0.65" --confirm_max_motion "0.08" --confirm_require_low "1" \
	    --thr_min "0.01" --thr_max "0.95" --thr_step "0.01" --time_mode "center" --merge_gap_s "1.0" --overlap_slack_s "0.5" \
	    --op1_recall "0.95" --op3_fa24h "1.0" --op2_objective "f1" --cost_fn "5.0" --cost_fp "1.0" \
	    --ops_picker "conservative" --op_tie_break "max_thr" --tie_eps "1e-3" \
	    --save_sweep_json "1" --allow_degenerate_sweep "0" --emit_absolute_paths "0" --min_tau_high "0.20"; \
	fi

$(LOCK_GCN_CAUC_OPS): $(STAMP_DIR)/windows_eval/caucafall.stamp
	@mkdir -p "$(@D)" "$(MET_DIR)"
	@test -f "$(LOCK_GCN_CAUC_CKPT)" || (echo "[ERR] missing locked GCN ckpt: $(LOCK_GCN_CAUC_CKPT)" && exit 1)
	$(RUN) ops/scripts/fit_ops.py --arch gcn \
	  --val_dir "$(call win_eval_dir,caucafall)/val" \
	  --ckpt "$(LOCK_GCN_CAUC_CKPT)" \
	  --out "$@" \
	  --fps_default "$(FPS_caucafall)" \
	  --center "$(CENTER)" --use_motion "$(FEAT_USE_MOTION)" --use_conf_channel "$(FEAT_USE_CONF_CHANNEL)" --use_bone "$(FEAT_USE_BONE)" --use_bone_length "$(FEAT_USE_BONE_LEN)" \
	  --ema_alpha "0.20" --k "2" --n "3" --cooldown_s "30" --tau_low_ratio "0.78" --confirm "0" --confirm_s "2.0" --confirm_min_lying "0.65" --confirm_max_motion "0.08" --confirm_require_low "1" \
	  --thr_min "0.01" --thr_max "0.95" --thr_step "0.01" --time_mode "center" --merge_gap_s "1.0" --overlap_slack_s "0.5" \
	  --op1_recall "0.95" --op3_fa24h "1.0" --op2_objective "cost_sensitive" --cost_fn "5.0" --cost_fp "10" \
	  --ops_picker "conservative" --op_tie_break "max_thr" --tie_eps "1e-3" \
	  --save_sweep_json "1" --allow_degenerate_sweep "0" --emit_absolute_paths "0" --min_tau_high "0.30"

$(LOCK_TCN_CAUC_MET): $(LOCK_TCN_CAUC_OPS)
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/eval_metrics.py \
	  --win_dir "$(call win_eval_dir,caucafall)/test" \
	  --ckpt "$(LOCK_TCN_CAUC_CKPT)" \
	  --ops_yaml "$(LOCK_TCN_CAUC_OPS)" \
	  --out_json "$@" \
	  --fps_default "$(FPS_caucafall)" \
	  $(METRICS_SWEEP_FLAGS)

$(LOCK_GCN_CAUC_MET): $(LOCK_GCN_CAUC_OPS)
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/eval_metrics.py \
	  --win_dir "$(call win_eval_dir,caucafall)/test" \
	  --ckpt "$(LOCK_GCN_CAUC_CKPT)" \
	  --ops_yaml "$(LOCK_GCN_CAUC_OPS)" \
	  --out_json "$@" \
	  --fps_default "$(FPS_caucafall)" \
	  $(METRICS_SWEEP_FLAGS)

# fit_ops → ops YAML
# ============================================================

$(OPS_DIR)/tcn_%$(OUT_TAG).yaml: $(STAMP_DIR)/windows_eval/%.stamp $$(if $$(filter 1,$$(FITOPS_USE_FA)),$(STAMP_DIR)/fa_windows/$$*.stamp,)
	@mkdir -p "$(@D)" "$(CAL_DIR)"
	@test -f "$(OUT_DIR)/$*_tcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt" || (echo "[ERR] missing checkpoint: $(OUT_DIR)/$*_tcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt. Run train-tcn-$* first (or set OUT_TAG to an existing run)." && exit 1)
	$(RUN) ops/scripts/fit_ops.py --arch tcn \
	  --val_dir "$(call win_eval_dir,$*)/val" \
	  --ckpt "$(OUT_DIR)/$*_tcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt" \
	  --out "$@" \
	  --fps_default "$(FPS_$*)" \
	  $(FITOPS_FEAT_FLAGS) \
	  $(FITOPS_POLICY_FLAGS) $(FITOPS_SWEEP_FLAGS) $(FITOPS_PICKER_FLAGS) $(FITOPS_GUARD_FLAGS) \
	  --min_tau_high "$(call get,FITOPS_MIN_TAU_HIGH,$*)" \
	  $(FITOPS_FA_ARG)

$(OPS_DIR)/gcn_%$(OUT_TAG).yaml: $(STAMP_DIR)/windows_eval/%.stamp $$(if $$(filter 1,$$(FITOPS_USE_FA)),$(STAMP_DIR)/fa_windows/$$*.stamp,)
	@mkdir -p "$(@D)" "$(CAL_DIR)"
	@test -f "$(OUT_DIR)/$*_gcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt" || (echo "[ERR] missing checkpoint: $(OUT_DIR)/$*_gcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt. Run train-gcn-$* first (or set OUT_TAG to an existing run)." && exit 1)
	$(RUN) ops/scripts/fit_ops.py --arch gcn \
	  --val_dir "$(call win_eval_dir,$*)/val" \
	  --ckpt "$(OUT_DIR)/$*_gcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt" \
	  --out "$@" \
	  --fps_default "$(FPS_$*)" \
	  $(FITOPS_FEAT_FLAGS) \
	  $(FITOPS_POLICY_FLAGS) $(FITOPS_SWEEP_FLAGS) $(FITOPS_PICKER_FLAGS) $(FITOPS_GUARD_FLAGS) \
	  --min_tau_high "$(call get,FITOPS_MIN_TAU_HIGH,$*)" \
	  $(FITOPS_FA_ARG)

# ============================================================
# Eval → metrics JSON
# Dataset-scoped eval targets (avoid pattern collisions like 'eval-gcn-le2i' matching 'eval-%').
.PHONY: $(addprefix eval-,$(DATASETS)) $(addprefix eval-gcn-,$(DATASETS)) \
        $(addprefix eval-unlabeled-,$(DATASETS)) $(addprefix eval-unlabeled-gcn-,$(DATASETS))

$(addprefix eval-,$(DATASETS)): eval-%: $(MET_DIR)/tcn_%$(OUT_TAG).json
	@:

$(addprefix eval-gcn-,$(DATASETS)): eval-gcn-%: $(MET_DIR)/gcn_%$(OUT_TAG).json
	@:

$(addprefix eval-unlabeled-,$(DATASETS)): eval-unlabeled-%: $(UNLABELED_MET_DIR)/tcn_%$(OUT_TAG)_unlabeled_fa.json
	@:

$(addprefix eval-unlabeled-gcn-,$(DATASETS)): eval-unlabeled-gcn-%: $(UNLABELED_MET_DIR)/gcn_%$(OUT_TAG)_unlabeled_fa.json
	@:

# ============================================================

$(MET_DIR)/tcn_%$(OUT_TAG).json: $(OPS_DIR)/tcn_%$(OUT_TAG).yaml $(OUT_DIR)/%_tcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt $(STAMP_DIR)/windows_eval/%.stamp
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/eval_metrics.py \
	  --win_dir "$(call win_eval_dir,$*)/test" \
	  --ckpt "$(OUT_DIR)/$*_tcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt" \
	  --ops_yaml "$(OPS_DIR)/tcn_$*$(OUT_TAG).yaml" \
	  --out_json "$@" \
	  --fps_default "$(FPS_$*)" \
	  $(METRICS_SWEEP_FLAGS)

$(MET_DIR)/gcn_%$(OUT_TAG).json: $(OPS_DIR)/gcn_%$(OUT_TAG).yaml $(OUT_DIR)/%_gcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt $(STAMP_DIR)/windows_eval/%.stamp
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/eval_metrics.py \
	  --win_dir "$(call win_eval_dir,$*)/test" \
	  --ckpt "$(OUT_DIR)/$*_gcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt" \
	  --ops_yaml "$(OPS_DIR)/gcn_$*$(OUT_TAG).yaml" \
	  --out_json "$@" \
	  --fps_default "$(FPS_$*)" \
	  $(METRICS_SWEEP_FLAGS)

$(UNLABELED_MET_DIR)/tcn_%$(OUT_TAG)_unlabeled_fa.json: $(STAMP_DIR)/windows_unlabeled/%.stamp
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/eval_metrics.py \
	  --win_dir "$(call win_unlabeled_dir,$*)/$(UNLABELED_SUBSET_EVAL)" \
	  --ckpt "$(if $(strip $(UNLABELED_CKPT)),$(UNLABELED_CKPT),$(OUT_DIR)/$*_tcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt)" \
	  --ops_yaml "$(if $(strip $(UNLABELED_OPS_YAML)),$(UNLABELED_OPS_YAML),$(OPS_DIR)/tcn_$*$(OUT_TAG).yaml)" \
	  --out_json "$@" \
	  --fps_default "$(FPS_$*)" \
	  $(METRICS_SWEEP_FLAGS)

$(UNLABELED_MET_DIR)/gcn_%$(OUT_TAG)_unlabeled_fa.json: $(STAMP_DIR)/windows_unlabeled/%.stamp
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/eval_metrics.py \
	  --win_dir "$(call win_unlabeled_dir,$*)/$(UNLABELED_SUBSET_EVAL)" \
	  --ckpt "$(if $(strip $(UNLABELED_CKPT)),$(UNLABELED_CKPT),$(OUT_DIR)/$*_gcn_W$(WIN_W)S$(WIN_S)$(OUT_TAG)/best.pt)" \
	  --ops_yaml "$(if $(strip $(UNLABELED_OPS_YAML)),$(UNLABELED_OPS_YAML),$(OPS_DIR)/gcn_$*$(OUT_TAG).yaml)" \
	  --out_json "$@" \
	  --fps_default "$(FPS_$*)" \
	  $(METRICS_SWEEP_FLAGS)

# ============================================================
# Plot
# Dataset-scoped plot targets (avoid pattern collisions like 'plot-gcn-le2i' matching 'plot-%').
.PHONY: $(addprefix plot-,$(DATASETS)) $(addprefix plot-gcn-,$(DATASETS)) $(addprefix plot-confmat-,$(DATASETS)) $(addprefix plot-confmat-gcn-,$(DATASETS)) $(addprefix plot-failure-,$(DATASETS)) $(addprefix plot-failure-gcn-,$(DATASETS)) $(addprefix plot-balance-,$(DATASETS))

$(addprefix plot-,$(DATASETS)): plot-%: \
  $(PLOT_DIR)/tcn_%$(OUT_TAG)_recall_vs_fa.png \
  $(PLOT_DIR)/tcn_%$(OUT_TAG)_f1_vs_tau.png
	@:

$(addprefix plot-gcn-,$(DATASETS)): plot-gcn-%: \
  $(PLOT_DIR)/gcn_%$(OUT_TAG)_recall_vs_fa.png \
  $(PLOT_DIR)/gcn_%$(OUT_TAG)_f1_vs_tau.png
	@:

$(addprefix plot-confmat-,$(DATASETS)): plot-confmat-%: \
  $(PLOT_DIR)/tcn_%$(OUT_TAG)_confusion_matrix.png
	@:

$(addprefix plot-confmat-gcn-,$(DATASETS)): plot-confmat-gcn-%: \
  $(PLOT_DIR)/gcn_%$(OUT_TAG)_confusion_matrix.png
	@:

$(addprefix plot-failure-,$(DATASETS)): plot-failure-%: \
  $(PLOT_DIR)/tcn_%$(OUT_TAG)_failure_scatter.png
	@:

$(addprefix plot-failure-gcn-,$(DATASETS)): plot-failure-gcn-%: \
  $(PLOT_DIR)/gcn_%$(OUT_TAG)_failure_scatter.png
	@:

$(addprefix plot-balance-,$(DATASETS)): plot-balance-%: \
  $(PLOT_DIR)/%$(OUT_TAG)_class_balance.png
	@:

# ============================================================

$(PLOT_DIR)/tcn_%$(OUT_TAG)_recall_vs_fa.png: $(MET_DIR)/tcn_%$(OUT_TAG).json
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/plot_fa_recall.py --reports "$<" --out_fig "$@"

$(PLOT_DIR)/tcn_%$(OUT_TAG)_f1_vs_tau.png: $(MET_DIR)/tcn_%$(OUT_TAG).json
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/plot_f1_vs_tau.py --reports "$<" --out_fig "$@"

$(PLOT_DIR)/gcn_%$(OUT_TAG)_recall_vs_fa.png: $(MET_DIR)/gcn_%$(OUT_TAG).json
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/plot_fa_recall.py --reports "$<" --out_fig "$@"

$(PLOT_DIR)/gcn_%$(OUT_TAG)_f1_vs_tau.png: $(MET_DIR)/gcn_%$(OUT_TAG).json
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/plot_f1_vs_tau.py --reports "$<" --out_fig "$@"

$(PLOT_DIR)/tcn_%$(OUT_TAG)_confusion_matrix.png: $(MET_DIR)/tcn_%$(OUT_TAG).json
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/plot_confusion_matrix.py --metrics_json "$<" --out_fig "$@" --normalize 1

$(PLOT_DIR)/gcn_%$(OUT_TAG)_confusion_matrix.png: $(MET_DIR)/gcn_%$(OUT_TAG).json
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/plot_confusion_matrix.py --metrics_json "$<" --out_fig "$@" --normalize 1

$(PLOT_DIR)/tcn_%$(OUT_TAG)_failure_scatter.png: $(MET_DIR)/tcn_%$(OUT_TAG).json
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/plot_failure_scatter.py --metrics_json "$<" --out_fig "$@" --fa_log 1 --style box

$(PLOT_DIR)/gcn_%$(OUT_TAG)_failure_scatter.png: $(MET_DIR)/gcn_%$(OUT_TAG).json
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/plot_failure_scatter.py --metrics_json "$<" --out_fig "$@" --fa_log 1 --style box

$(PLOT_DIR)/%$(OUT_TAG)_class_balance.png: $(LABELS_DIR)/%.json $(SPLITS_DIR)/%_train.txt $(SPLITS_DIR)/%_val.txt $(SPLITS_DIR)/%_test.txt
	@mkdir -p "$(@D)"
	$(RUN) ops/scripts/plot_dataset_balance.py --dataset "$*" --labels_json "$(LABELS_DIR)/$*.json" --splits_dir "$(SPLITS_DIR)" --out_fig "$@"

# ============================================================
# Sanity checks
# ============================================================
check-windows: $(addprefix check-windows-,$(DATASETS))

check-windows-%: windows-%
	$(RUN) ops/scripts/check_windows.py --root "$(call win_dir,$*)"

# ============================================================
# Audit gates
# ============================================================
AUDIT_DATASETS ?= le2i,caucafall
PROFILE ?= cpu_local
DS ?= le2i
MODEL ?= tcn
PROFILE_IO_ONLY ?= 1

.PHONY: audit-smoke audit-ci audit-static audit-runtime-imports audit-api-contract audit-api-v1-parity audit-api-smoke audit-integration-contract audit-ops-sanity audit-artifact-bundle audit-promoted-profiles audit-numeric audit-temporal audit-parity-le2i audit-parity-le2i-strict baseline-capture-le2i audit-all profile-infer

audit-smoke:
	$(RUN) ops/scripts/audit_smoke.py --root "."

audit-ci: audit-smoke audit-static audit-runtime-imports audit-api-contract audit-api-v1-parity audit-api-smoke
	$(RUN) -m pytest -q qa/tests/test_import_smoke.py qa/tests/test_data_sources_config.py qa/tests/test_windows_contract.py qa/tests/test_adapter_contract.py qa/tests/test_event_time_semantics.py qa/tests/test_audit_api_contract.py qa/tests/test_audit_api_v1_parity.py qa/tests/test_server_integration_contract.py qa/tests/test_repro_manifest_schema.py qa/tests/test_monitor_benchmark_schema.py qa/tests/test_monitor_fault_injection.py qa/tests/test_split_group_leakage.py
	@echo "[ok] audit-ci passed"

audit-static:
	$(RUN) ops/scripts/audit_static.py --roots "ml/src,ops/scripts,applications/backend,ops/configs"

audit-runtime-imports:
	$(RUN) ops/scripts/audit_runtime_imports.py --paths "ml/src/fall_detection/deploy,applications/backend/deploy_runtime.py"

audit-api-contract:
	$(RUN) ops/scripts/audit_api_contract.py

audit-api-v1-parity:
	$(RUN) ops/scripts/audit_api_v1_parity.py

audit-api-smoke:
	$(RUN) ops/scripts/smoke_api_contract.py

audit-integration-contract: audit-api-contract audit-api-v1-parity audit-api-smoke
	$(RUN) -m pytest -q qa/tests/test_server_integration_contract.py
	@echo "[ok] integration-contract audit passed"

audit-ops-sanity:
	$(RUN) ops/scripts/audit_ops_sanity.py --ops_dir "ops/configs/ops"

audit-artifact-bundle:
	$(RUN) ops/scripts/audit_artifact_bundle.py --bundle_json "artifacts/artifact_bundle.json"

audit-promoted-profiles:
	$(RUN) ops/scripts/audit_promoted_profiles.py \
	  --check "le2i_tcn|outputs/metrics/tcn_le2i_hneg_pack_tsm_promoted.json|artifacts/reports/hneg_cycle/tcn_le2i_hneg_pack_tsm_promoted_unlabeled_fa.json|1.0|0|0.0|0.0|0" \
	  --check "caucafall_tcn|outputs/metrics/tcn_caucafall_promoted.json|artifacts/reports/hneg_cycle/caucafall_tcn_promoted_unlabeled_fa.json|1.0|0|0.0|0.0|0" \
	  --check "caucafall_gcn|outputs/metrics/gcn_caucafall_promoted2.json|artifacts/reports/hneg_cycle/gcn_caucafall_promoted2_unlabeled_fa.json|1.0|0|0.0|0.0|0" \
	  --out_json "artifacts/reports/promoted_profiles_$(shell date +%Y%m%d).json"

audit-numeric:
	$(RUN) ops/scripts/audit_numeric.py \
	  --gates_json "ops/configs/audit_gates.json" \
	  --processed_root "$(PROCESSED)" \
	  --datasets "$(AUDIT_DATASETS)" \
	  --out_json "artifacts/reports/numeric_fingerprint_$(shell date +%Y%m%d).json"

audit-temporal:
	$(RUN) ops/scripts/audit_temporal.py \
	  --gates_json "ops/configs/audit_gates.json" \
	  --processed_root "$(PROCESSED)" \
	  --datasets "$(AUDIT_DATASETS)" \
	  --stride_frames "$(WIN_S)" \
	  --target_window_seconds "1.92" \
	  --target_stride_seconds "0.48" \
	  --out_json "artifacts/reports/temporal_span_$(shell date +%Y%m%d).json"

audit-parity-le2i:
	$(RUN) ops/scripts/audit_parity.py \
	  --baseline_dir "baselines/le2i/58813e8" \
	  --op "op2" \
	  --current_metrics_json "$(MET_DIR)/$(MODEL)_le2i.json" \
	  --out_json "artifacts/reports/parity_le2i_$(MODEL)_$(shell date +%Y%m%d).json"

audit-parity-le2i-strict:
	$(RUN) ops/scripts/audit_parity.py \
	  --baseline_dir "baselines/le2i/58813e8" \
	  --op "op2" \
	  --current_metrics_json "$(MET_DIR)/$(MODEL)_le2i.json" \
	  --require_perf_targets "1" \
	  --allow_missing_perf_baseline "0" \
	  --out_json "artifacts/reports/parity_le2i_$(MODEL)_strict_$(shell date +%Y%m%d).json"

baseline-capture-le2i:
	$(RUN) ops/scripts/baseline_set_performance.py \
	  --baseline_perf_json "baselines/le2i/58813e8/performance_baseline.json" \
	  --metrics_json "$(MET_DIR)/$(MODEL)_le2i.json" \
	  --checkpoint "$(OUT_DIR)/le2i_$(MODEL)_W$(WIN_W)S$(WIN_S)/best.pt" \
	  --op "op2" \
	  --notes "Captured via Make baseline-capture-le2i on $(shell date +%Y-%m-%d)"

audit-all: audit-smoke audit-static audit-runtime-imports audit-integration-contract audit-ops-sanity audit-artifact-bundle audit-promoted-profiles audit-numeric audit-temporal audit-parity-le2i-strict
	@echo "[ok] audit-all passed"

profile-infer:
	$(RUN) ops/scripts/profile_infer.py \
	  --profile "$(PROFILE)" \
	  --arch "$(MODEL)" \
	  --win_dir "$(call win_eval_dir,$(DS))/test" \
	  --ckpt "$(OUT_DIR)/$(DS)_$(MODEL)_W$(WIN_W)S$(WIN_S)/best.pt" \
	  --io_only "$(PROFILE_IO_ONLY)" \
	  --out_json "artifacts/reports/infer_profile_$(PROFILE)_$(MODEL)_$(DS).json"

# ============================================================
# Pipelines (pure deps; -j friendly)
# ============================================================
# Back-compat aliases
pipeline-%-data: pipeline-data-%   # e.g., pipeline-le2i-data
	@:
preprocess-%-only: preprocess-only-%  # e.g., preprocess-le2i-only
	@:

pipeline-data-%: DO_EXTRACT=1
pipeline-data-%: windows-%
	@:

pipeline-%-noextract: DO_EXTRACT=0
pipeline-%-noextract: windows-%
	@:

pipeline-%: DO_EXTRACT=1
pipeline-%: plot-%
	@:

pipeline-gcn-%: DO_EXTRACT=1
pipeline-gcn-%: plot-gcn-%
	@:

pipeline-all: $(addprefix pipeline-,$(DATASETS))
pipeline-all-gcn: $(addprefix pipeline-gcn-,$(DATASETS))
pipeline-all-noextract: $(addsuffix -noextract,$(addprefix pipeline-,$(DATASETS)))
pipeline-all-gcn-noextract: $(addsuffix -noextract,$(addprefix pipeline-gcn-,$(DATASETS)))

# ============================================================
# Auto pipeline orchestration (single-command, model-specific)
# Enforces adapter-mode window generation for sampling parity.
# ============================================================
.PHONY: $(addprefix pipeline-auto-tcn-,$(DATASETS)) $(addprefix pipeline-auto-gcn-,$(DATASETS))
AUTO_DO_EXTRACT ?= 0
AUTO_FORCE_REBUILD ?= 0
AUTO_WIN_EVAL_CLEAN ?= 0

$(addprefix pipeline-auto-tcn-,$(DATASETS)): pipeline-auto-tcn-%:
	@mkdir -p ".make/locks"
	@while ! mkdir ".make/locks/windows-$*.lock" 2>/dev/null; do echo "[wait] windows lock busy for $*"; sleep 1; done; \
	trap 'rmdir ".make/locks/windows-$*.lock"' EXIT INT TERM; \
	$(MAKE) $(if $(filter 1,$(AUTO_FORCE_REBUILD)),-B,) DO_EXTRACT="$(AUTO_DO_EXTRACT)" ADAPTER_USE=1 WIN_EVAL_CLEAN="$(AUTO_WIN_EVAL_CLEAN)" windows-$* windows-eval-$*
	@$(if $(filter 1,$(FITOPS_USE_FA)),$(MAKE) ADAPTER_USE=1 fa-windows-$*,:)
	@$(MAKE) ADAPTER_USE=1 train-tcn-$*
	@$(MAKE) ADAPTER_USE=1 fit-ops-$*
	@$(MAKE) ADAPTER_USE=1 eval-$*
	# Optional future step (when standardized):
	# $(MAKE) ADAPTER_USE=1 mine-hard-negatives-tcn-$*
	@$(MAKE) ADAPTER_USE=1 plot-$*

$(addprefix pipeline-auto-gcn-,$(DATASETS)): pipeline-auto-gcn-%:
	@mkdir -p ".make/locks"
	@while ! mkdir ".make/locks/windows-$*.lock" 2>/dev/null; do echo "[wait] windows lock busy for $*"; sleep 1; done; \
	trap 'rmdir ".make/locks/windows-$*.lock"' EXIT INT TERM; \
	$(MAKE) $(if $(filter 1,$(AUTO_FORCE_REBUILD)),-B,) DO_EXTRACT="$(AUTO_DO_EXTRACT)" ADAPTER_USE=1 WIN_EVAL_CLEAN="$(AUTO_WIN_EVAL_CLEAN)" windows-$* windows-eval-$*
	@$(if $(filter 1,$(FITOPS_USE_FA)),$(MAKE) ADAPTER_USE=1 fa-windows-$*,:)
	@$(MAKE) ADAPTER_USE=1 train-gcn-$*
	@$(MAKE) ADAPTER_USE=1 fit-ops-gcn-$*
	@$(MAKE) ADAPTER_USE=1 eval-gcn-$*
	# Optional future step (when standardized):
	# $(MAKE) ADAPTER_USE=1 mine-hard-negatives-gcn-$*
	@$(MAKE) ADAPTER_USE=1 plot-gcn-$*

eval-all: $(addprefix eval-,$(DATASETS))
plot-all: $(addprefix plot-,$(DATASETS))
eval-all-gcn: $(addprefix eval-gcn-,$(DATASETS))
plot-all-gcn: $(addprefix plot-gcn-,$(DATASETS))

# Offline
# make fit-ops-caucafall ALERT_CONFIRM=0 ALERT_EMA_ALPHA=0.5
# make eval-caucafall ALERT_CONFIRM=0 ALERT_EMA_ALPHA=0.5
# Online
# make fit-ops-unlabeled ALERT_CONFIRM=1 ALERT_CONFIRM_S=2.0 ALERT_EMA_ALPHA=0.2
