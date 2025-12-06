# -----------------------------------------
# Makefile (task runner) — Fall Detection v2
# -----------------------------------------
SHELL := /bin/bash
.DEFAULT_GOAL := help

# Python + venv
PY=python
VENV=source .venv/bin/activate

# Windowing knobs (override: make WIN_W=64 WIN_S=16 train)
WIN_W ?= 48
WIN_S ?= 12

# Split knobs (keep simple to avoid Make parsing issues)
SPLIT_SEED ?= 33724876
TRAIN_FRAC ?= 0.80
VAL_FRAC   ?= 0.10
TEST_FRAC  ?= 0.10

# Derived paths
WIN_DIR_LE2I      := data/processed/le2i/windows_W$(WIN_W)_S$(WIN_S)
WIN_DIR_URFD      := data/processed/urfd/windows_W$(WIN_W)_S$(WIN_S)
WIN_DIR_CAUC      := data/processed/caucafall/windows_W$(WIN_W)_S$(WIN_S)
WIN_DIR_MUVIM     := data/processed/muvim/windows_W$(WIN_W)_S$(WIN_S)

# TCN checkpoints
CKPT_LE2I         := outputs/le2i_tcn_W$(WIN_W)S$(WIN_S)/best.pt
CKPT_URFD         := outputs/urfd_tcn_W$(WIN_W)S$(WIN_S)/best.pt
CKPT_CAUC         := outputs/caucafall_tcn_W$(WIN_W)S$(WIN_S)/best.pt

# (optional) GCN checkpoints – useful later for eval / server
CKPT_LE2I_GCN     := outputs/le2i_gcn_W$(WIN_W)S$(WIN_S)/best.pt
CKPT_URFD_GCN     := outputs/urfd_gcn_W$(WIN_W)S$(WIN_S)/best.pt
CKPT_CAUC_GCN     := outputs/caucafall_gcn_W$(WIN_W)S$(WIN_S)/best.pt

.PHONY: help install \
        extract-le2i extract-urfd extract-caucafall extract-muvim \
        labels-le2i le2i-unlabeled-list labels-urfd labels-caucafall \
        splits-le2i splits-urfd splits-caucafall splits-muvim splits-all \
        windows-le2i windows-le2i-unlabeled windows-urfd windows-caucafall windows-muvim \
        check-le2i check-urfd check-caucafall check-muvim \
        train train-le2i train-urfd train-caucafall \
        train-gcn-le2i train-gcn-urfd train-gcn-caucafall train-gcn \
        fit-ops fit-ops-le2i fit-ops-urfd fit-ops-caucafall \
        eval-le2i eval-urfd eval-le2i-on-urfd eval-caucafall eval-caucafall-on-urfd \
        plot plot-urfd-cross plot-le2i plot-caucafall plot-caucafall-on-urfd \
        pipeline score-unlabeled-le2i \
        clean clean-le2i clean-urfd clean-caucafall clean-muvim clean-windows clean-outputs

help:
	@echo "Targets:"
	@echo "  make install                 # create venv, lock & install deps"
	@echo "  make extract-<ds>            # extract pose/NPZ for le2i|urfd|caucafall|muvim"
	@echo "  make labels-<ds>             # build labels for le2i/urfd/caucafall"
	@echo "  make splits-<ds>             # stratified splits"
	@echo "  make windows-<ds>            # windowify datasets"
	@echo "  make check-<ds>              # sanity-check window counts/labels"
	@echo "  make train-le2i|train-urfd|train-caucafall        # train TCN per dataset"
	@echo "  make train-gcn-le2i|train-gcn-urfd|train-gcn-caucafall  # train GCN per dataset"
	@echo "  make train                   # alias for train-le2i (baseline TCN)"
	@echo "  make train-gcn               # train GCN on all datasets"
	@echo "  make fit-ops-le2i            # fit OP1/2/3 on LE2I val (TCN)"
	@echo "  make fit-ops                 # alias for fit-ops-le2i"
	@echo "  make eval-le2i               # LE2I in-domain (TCN)"
	@echo "  make eval-urfd               # alias: LE2I TCN model on URFD (cross)"
	@echo "  make eval-caucafall          # CAUCAFall in-domain (TCN)"
	@echo "  make eval-caucafall-on-urfd  # CAUCAFall TCN model on URFD (cross)"
	@echo "  make plot-urfd-cross         # FA/24h vs recall plot for LE2I→URFD (TCN)"
	@echo "  make plot-le2i|plot-caucafall# FA/24h vs recall plots (in-domain TCN)"
	@echo "  make plot                    # alias for plot-urfd-cross"
	@echo "  make pipeline                # end-to-end LE2I pipeline (extract→train→fit-ops)"
	@echo "  make clean*                  # remove intermediates/outputs"

# -----------------------------------------
# Setup
# -----------------------------------------
install:
	@[ -d .venv ] || python3 -m venv .venv
	$(VENV) && $(PY) -m pip install --upgrade "pip==24.2" "pip-tools==7.4.1" setuptools wheel && \
	pip-compile --resolver=backtracking requirements.in && \
	pip-sync requirements.txt

# -----------------------------------------
# Extract
# -----------------------------------------
extract-le2i:
	$(VENV) && $(PY) pose/extract_2d.py --videos_glob 'data/raw/LE2i/**/videos/*.avi' --out_dir 'data/interim/le2i/pose_npz' || true
	$(VENV) && $(PY) pose/extract_2d.py --videos_glob 'data/raw/LE2i/**/*.avi'          --out_dir 'data/interim/le2i/pose_npz' || true
	$(VENV) && $(PY) pose/extract_2d.py --videos_glob 'data/raw/LE2i/**/videos/*.mp4' --out_dir 'data/interim/le2i/pose_npz' || true
	$(VENV) && $(PY) pose/extract_2d.py --videos_glob 'data/raw/LE2i/**/*.mp4'          --out_dir 'data/interim/le2i/pose_npz' || true

extract-urfd:
	$(VENV) && $(PY) pose/extract_2d_from_images.py \
	  --images_glob 'data/raw/UR_Fall/adl/*/*.png' 'data/raw/UR_Fall/fall/*/*.png' \
	  --sequence_id_depth 1 \
	  --out_dir data/interim/urfd/pose_npz \
	  --dataset urfd

extract-caucafall:
	$(VENV) && $(PY) pose/extract_2d_from_images.py \
	  --images_glob 'data/raw/CAUCAFall/CAUCAFall/*/*/*.png' \
	  --sequence_id_depth 2 \
	  --out_dir data/interim/caucafall/pose_npz \
	  --dataset caucafall

# Optional: requires pose/parse_muvim_csv.py
extract-muvim:
	$(VENV) && $(PY) pose/parse_muvim_csv.py \
	  --csv_glob 'data/raw/MUVIM/*/*.csv' \
	  --out_root data/interim/muvim/pose_npz \
	  --x_pat 'x{i}' --y_pat 'y{i}' --score_pat 'score{i}' --i_start 1 --i_end 34

# -----------------------------------------
# Labels
# -----------------------------------------
labels-le2i:
	@mkdir -p configs/labels
	$(VENV) && $(PY) tools/parse_le2i_annotations.py \
	  --raw_root data/raw/LE2i \
	  --npz_dir  data/interim/le2i/pose_npz \
	  --out_labels configs/labels/le2i.json \
	  --out_spans  configs/labels/le2i_spans.json

le2i-unlabeled-list:
	@mkdir -p configs/splits
	$(VENV) && $(PY) tools/make_unlabeled_test_list.py \
	  --npz_dir data/interim/le2i/pose_npz \
	  --out configs/splits/le2i_test_unlabeled.txt \
	  --scenes Office "Lecture room"

labels-urfd:
	@mkdir -p configs/labels
	$(VENV) && $(PY) tools/make_urfd_labels_from_stems.py \
	  --npz_dir data/interim/urfd/pose_npz \
	  --out configs/labels/urfd_auto.json

labels-caucafall:
	@echo "CAUCAFall labels were generated during extraction: configs/labels/caucafall_auto.json"

# -----------------------------------------
# Splits
# -----------------------------------------
splits-le2i:
	@mkdir -p configs/splits
	$(VENV) && $(PY) tools/make_splits.py \
	  --labels_json configs/labels/le2i.json \
	  --out_dir configs/splits \
	  --train $(TRAIN_FRAC) --val $(VAL_FRAC) --test $(TEST_FRAC) --seed $(SPLIT_SEED)

splits-urfd:
	@mkdir -p configs/splits
	$(VENV) && $(PY) tools/make_splits.py \
	  --labels_json configs/labels/urfd_auto.json \
	  --out_dir configs/splits \
	  --train $(TRAIN_FRAC) --val $(VAL_FRAC) --test $(TEST_FRAC) --seed $(SPLIT_SEED) \
	  --prefix urfd

splits-caucafall:
	@mkdir -p configs/splits
	$(VENV) && $(PY) tools/make_splits.py \
	  --labels_json configs/labels/caucafall_auto.json \
	  --out_dir configs/splits \
	  --train $(TRAIN_FRAC) --val $(VAL_FRAC) --test $(TEST_FRAC) --seed $(SPLIT_SEED) \
	  --prefix caucafall

splits-muvim:
	@mkdir -p configs/splits
	$(VENV) && $(PY) tools/make_random_splits_from_npz.py \
	  --npz_dir data/interim/muvim/pose_npz \
	  --out_dir configs/splits \
	  --train $(TRAIN_FRAC) --val $(VAL_FRAC) --test $(TEST_FRAC) --seed $(SPLIT_SEED) \
	  --prefix muvim

splits-all: splits-le2i splits-urfd splits-caucafall splits-muvim
	@echo "Splits created for all datasets."

# -----------------------------------------
# Windowing
# -----------------------------------------
windows-le2i:
	$(VENV) && $(PY) pose/make_windows.py \
	  --npz_dir data/interim/le2i/pose_npz \
	  --labels_json configs/labels/le2i.json \
	  --spans_json  configs/labels/le2i_spans.json \
	  --out_dir $(WIN_DIR_LE2I) \
	  --W $(WIN_W) --stride $(WIN_S) \
	  --train_list configs/splits/le2i_train.txt \
	  --val_list   configs/splits/le2i_val.txt \
	  --test_list  configs/splits/le2i_test.txt

windows-le2i-unlabeled:
	$(VENV) && $(PY) tools/make_unlabeled_windows.py \
	  --npz_dir data/interim/le2i/pose_npz \
	  --stems_txt configs/splits/le2i_test_unlabeled.txt \
	  --out_dir $(WIN_DIR_LE2I) \
	  --W $(WIN_W) --stride $(WIN_S)

windows-urfd:
	$(VENV) && $(PY) pose/make_windows.py \
	  --npz_dir data/interim/urfd/pose_npz \
	  --labels_json configs/labels/urfd_auto.json \
	  --out_dir $(WIN_DIR_URFD) \
	  --W $(WIN_W) --stride $(WIN_S) \
	  --train_list configs/splits/urfd_train.txt \
	  --val_list   configs/splits/urfd_val.txt \
	  --test_list  configs/splits/urfd_test.txt

windows-caucafall:
	$(VENV) && $(PY) pose/make_windows.py \
	  --npz_dir data/interim/caucafall/pose_npz \
	  --labels_json configs/labels/caucafall_auto.json \
	  --out_dir $(WIN_DIR_CAUC) \
	  --W $(WIN_W) --stride $(WIN_S) \
	  --train_list configs/splits/caucafall_train.txt \
	  --val_list   configs/splits/caucafall_val.txt \
	  --test_list  configs/splits/caucafall_test.txt

windows-muvim:
	$(VENV) && $(PY) pose/make_windows.py \
	  --npz_dir data/interim/muvim/pose_npz \
	  --out_dir $(WIN_DIR_MUVIM) \
	  --W $(WIN_W) --stride $(WIN_S) --use_npz_labels \
	  --train_list configs/splits/muvim_train.txt \
	  --val_list   configs/splits/muvim_val.txt \
	  --test_list  configs/splits/muvim_test.txt

# -----------------------------------------
# Checks
# -----------------------------------------
check-le2i:
	$(VENV) && $(PY) tools/check_windows.py --root $(WIN_DIR_LE2I)

check-urfd:
	$(VENV) && $(PY) tools/check_windows.py --root $(WIN_DIR_URFD)

check-caucafall:
	$(VENV) && $(PY) tools/check_windows.py --root $(WIN_DIR_CAUC)

check-muvim:
	$(VENV) && $(PY) tools/check_windows.py --root $(WIN_DIR_MUVIM)

# -----------------------------------------
# Train / Eval / Plot
# -----------------------------------------

# ----------------------
# TCN training
# ----------------------
# Baseline LE2I training
train-le2i:
	$(VENV) && $(PY) models/train_tcn.py \
	  --train_dir $(WIN_DIR_LE2I)/train \
	  --val_dir   $(WIN_DIR_LE2I)/val \
	  --epochs 50 --batch 128 --lr 1e-3 --seed $(SPLIT_SEED) \
	  --save_dir outputs/le2i_tcn_W$(WIN_W)S$(WIN_S)

# Alias: old 'make train' = train-le2i
train: train-le2i

train-urfd:
	$(VENV) && $(PY) models/train_tcn.py \
	  --train_dir $(WIN_DIR_URFD)/train \
	  --val_dir   $(WIN_DIR_URFD)/val \
	  --epochs 50 --batch 128 --lr 1e-3 --seed $(SPLIT_SEED) \
	  --save_dir outputs/urfd_tcn_W$(WIN_W)S$(WIN_S)

train-caucafall:
	$(VENV) && $(PY) models/train_tcn.py \
	  --train_dir $(WIN_DIR_CAUC)/train \
	  --val_dir   $(WIN_DIR_CAUC)/val \
	  --epochs 50 --batch 128 --lr 1e-3 --seed $(SPLIT_SEED) \
	  --save_dir outputs/caucafall_tcn_W$(WIN_W)S$(WIN_S)

# ----------------------
# GCN training
# ----------------------
train-gcn-le2i:
	$(VENV) && $(PY) models/train_gcn.py \
	  --train_dir $(WIN_DIR_LE2I)/train \
	  --val_dir   $(WIN_DIR_LE2I)/val \
	  --test_dir  $(WIN_DIR_LE2I)/test \
	  --epochs 50 --batch 128 --lr 1e-3 --seed $(SPLIT_SEED) \
	  --save_dir      outputs/le2i_gcn_W$(WIN_W)S$(WIN_S) \
	  --report_json   outputs/reports/le2i_gcn_in_domain.json \
	  --report_dataset_name test

train-gcn-urfd:
	$(VENV) && $(PY) models/train_gcn.py \
	  --train_dir $(WIN_DIR_URFD)/train \
	  --val_dir   $(WIN_DIR_URFD)/val \
	  --test_dir  $(WIN_DIR_URFD)/test \
	  --epochs 50 --batch 128 --lr 1e-3 --seed $(SPLIT_SEED) \
	  --save_dir      outputs/urfd_gcn_W$(WIN_W)S$(WIN_S) \
	  --report_json   outputs/reports/urfd_gcn_in_domain.json \
	  --report_dataset_name test

train-gcn-caucafall:
	$(VENV) && $(PY) models/train_gcn.py \
	  --train_dir $(WIN_DIR_CAUC)/train \
	  --val_dir   $(WIN_DIR_CAUC)/val \
	  --test_dir  $(WIN_DIR_CAUC)/test \
	  --epochs 50 --batch 128 --lr 1e-3 --seed $(SPLIT_SEED) \
	  --save_dir      outputs/caucafall_gcn_W$(WIN_W)S$(WIN_S) \
	  --report_json   outputs/reports/caucafall_gcn_in_domain.json \
	  --report_dataset_name test

# Convenience: train all three GCNs
train-gcn: train-gcn-le2i train-gcn-urfd train-gcn-caucafall
	@echo "GCN training complete for LE2I, URFD, CAUCAFall"

# ----------------------
# Fit OP thresholds (TCN)
# ----------------------
fit-ops-le2i:
	@mkdir -p configs
	$(VENV) && $(PY) eval/fit_ops.py \
	  --val_dir $(WIN_DIR_LE2I)/val \
	  --ckpt $(CKPT_LE2I) \
	  --out configs/ops_le2i.yaml

fit-ops-urfd:
	@mkdir -p configs
	$(VENV) && $(PY) eval/fit_ops.py \
	  --val_dir $(WIN_DIR_URFD)/val \
	  --ckpt $(CKPT_URFD) \
	  --out configs/ops_urfd.yaml

fit-ops-caucafall:
	@mkdir -p configs
	$(VENV) && $(PY) eval/fit_ops.py \
	  --val_dir $(WIN_DIR_CAUC)/val \
	  --ckpt $(CKPT_CAUC) \
	  --out configs/ops_caucafall.yaml

# Alias: used by pipeline
fit-ops: fit-ops-le2i
	@echo "fit-ops → fit-ops-le2i (TCN)"

# ----------------------
# Evaluations (TCN)
# ----------------------
# In-domain LE2I evaluation
eval-le2i:
	@mkdir -p outputs/reports
	$(VENV) && $(PY) eval/metrics.py \
	  --eval_dir $(WIN_DIR_LE2I)/test \
	  --ckpt $(CKPT_LE2I) \
	  --ops configs/ops_le2i.yaml --fps 30 \
	  --report outputs/reports/le2i_in_domain.json

# Cross-dataset: LE2I model → URFD test
eval-le2i-on-urfd:
	@mkdir -p outputs/reports
	$(VENV) && $(PY) eval/metrics.py \
	  --eval_dir $(WIN_DIR_URFD)/test \
	  --ckpt $(CKPT_LE2I) \
	  --ops configs/ops_le2i.yaml --fps 30 \
	  --report outputs/reports/urfd_cross.json

# Alias for backwards compatibility
eval-urfd: eval-le2i-on-urfd
	@echo "eval-urfd → eval-le2i-on-urfd (TCN LE2I→URFD)"

# In-domain CAUCAFall evaluation
eval-caucafall:
	@mkdir -p outputs/reports
	$(VENV) && $(PY) eval/metrics.py \
	  --eval_dir $(WIN_DIR_CAUC)/test \
	  --ckpt $(CKPT_CAUC) \
	  --ops configs/ops_caucafall.yaml --fps 30 \
	  --report outputs/reports/caucafall_in_domain.json

# Cross: CAUCAFall model → URFD test
eval-caucafall-on-urfd:
	@mkdir -p outputs/reports
	$(VENV) && $(PY) eval/metrics.py \
	  --eval_dir $(WIN_DIR_URFD)/test \
	  --ckpt $(CKPT_CAUC) \
	  --ops configs/ops_caucafall.yaml --fps 30 \
	  --report outputs/reports/caucafall_on_urfd.json

# ----------------------
# Plots (TCN)
# ----------------------
plot-urfd-cross:
	@mkdir -p outputs/figures
	$(VENV) && $(PY) eval/plot_fa_recall.py \
	  --reports outputs/reports/urfd_cross.json \
	  --title "Innovation Impact: Reducing Alarm Fatigue at Target Recall (H2)" \
	  --subtitle "FA/24h vs. Recall (OP-3 Validation)" \
	  --out_fig outputs/figures/tcn_le2i_on_urfd_fa_recall_W$(WIN_W)_S$(WIN_S).png

plot-le2i:
	@mkdir -p outputs/figures
	$(VENV) && $(PY) eval/plot_fa_recall.py \
	  --reports outputs/reports/le2i_in_domain.json \
	  --title "LE2I In-Domain: FA/24h vs Recall" \
	  --subtitle "TCN (W$(WIN_W), S$(WIN_S))" \
	  --out_fig outputs/figures/tcn_le2i_on_le2i_fa_recall_W$(WIN_W)_S$(WIN_S).png

plot-caucafall:
	@mkdir -p outputs/figures
	$(VENV) && $(PY) eval/plot_fa_recall.py \
	  --reports outputs/reports/caucafall_in_domain.json \
	  --title "CAUCAFall In-Domain: FA/24h vs Recall" \
	  --subtitle "TCN (W$(WIN_W), S$(WIN_S))" \
	  --out_fig outputs/figures/tcn_caucafall_on_caucafall_fa_recall_W$(WIN_W)_S$(WIN_S).png

plot-caucafall-on-urfd:
	@mkdir -p outputs/figures
	$(VENV) && $(PY) eval/plot_fa_recall.py \
	  --reports outputs/reports/caucafall_on_urfd.json \
	  --title "CAUCAFall Model → URFD: FA/24h vs Recall" \
	  --subtitle "TCN (W$(WIN_W), S$(WIN_S))" \
	  --out_fig outputs/figures/tcn_caucafall_on_urfd_fa_recall_W$(WIN_W)_S$(WIN_S).png

# Alias: plot = plot-urfd-cross
plot: plot-urfd-cross
	@echo "plot → plot-urfd-cross"

# ----------------------
# Plots for GCN (optional, nice for the report)
# ----------------------
plot-le2i-gcn:
	@mkdir -p outputs/figures
	$(VENV) && $(PY) eval/plot_fa_recall.py \
	  --reports outputs/reports/le2i_gcn_in_domain.json \
	  --title "LE2I In-Domain: FA/24h vs Recall (GCN)" \
	  --subtitle "GCN (W$(WIN_W), S$(WIN_S))" \
	  --out_fig outputs/figures/gcn_le2i_on_le2i_fa_recall_W$(WIN_W)_S$(WIN_S).png

plot-urfd-gcn:
	@mkdir -p outputs/figures
	$(VENV) && $(PY) eval/plot_fa_recall.py \
	  --reports outputs/reports/urfd_gcn_in_domain.json \
	  --title "URFD In-Domain: FA/24h vs Recall (GCN)" \
	  --subtitle "GCN (W$(WIN_W), S$(WIN_S))" \
	  --out_fig outputs/figures/gcn_urfd_on_urfd_fa_recall_W$(WIN_W)_S$(WIN_S).png

plot-caucafall-gcn:
	@mkdir -p outputs/figures
	$(VENV) && $(PY) eval/plot_fa_recall.py \
	  --reports outputs/reports/caucafall_gcn_in_domain.json \
	  --title "CAUCAFall In-Domain: FA/24h vs Recall (GCN)" \
	  --subtitle "GCN (W$(WIN_W), S$(WIN_S))" \
	  --out_fig outputs/figures/gcn_caucafall_on_caucafall_fa_recall_W$(WIN_W)_S$(WIN_S).png


# Shadow-deploy alert rate on unlabeled Office/Lecture
THR ?= 0.50
FPS ?= 25
COOL ?= 3
score-unlabeled-le2i:
	@mkdir -p outputs/reports
	$(VENV) && $(PY) tools/score_unlabeled_alert_rate.py \
	  --windows_dir $(WIN_DIR_LE2I)/test_unlabeled \
	  --ckpt $(CKPT_LE2I) \
	  --thr $(THR) --fps $(FPS) --stride $(WIN_S) --cooldown_sec $(COOL) \
	  --csv_out outputs/reports/office_lecture_unlabeled_scores.csv

# -----------------------------------------
# Pipelines
# -----------------------------------------
pipeline: extract-le2i labels-le2i splits-le2i windows-le2i le2i-unlabeled-list windows-le2i-unlabeled check-le2i train fit-ops
	@echo "LE2i pipeline complete."

# -----------------------------------------
# Cleaning
# -----------------------------------------
clean:
	rm -rf data/interim data/processed outputs

clean-le2i:
	rm -rf data/interim/le2i data/processed/le2i

clean-urfd:
	rm -rf data/interim/urfd data/processed/urfd

clean-caucafall:
	rm -rf data/interim/caucafall data/processed/caucafall

clean-muvim:
	rm -rf data/interim/muvim data/processed/muvim

clean-windows:
	rm -rf data/processed/*/windows_W*_S*

clean-outputs:
	rm -rf outputs
