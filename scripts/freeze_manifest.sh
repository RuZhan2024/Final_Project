#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FREEZE_CORE_PATHS=(
  "README.md"
  "docker-compose.yml"
  "src/fall_detection"
  "server"
  "apps"
  "scripts"
  "tests"
  "configs/ops/README.md"
  "configs/ops/tcn_caucafall.yaml"
  "configs/ops/gcn_caucafall.yaml"
  "configs/ops/tcn_le2i.yaml"
  "configs/ops/gcn_le2i.yaml"
  "configs/ops/tcn_caucafall_r2_train_hneg.yaml"
  "configs/ops/gcn_caucafall_r2_recallpush_b.yaml"
  "configs/ops/tcn_le2i_opt33_r2.yaml"
  "configs/ops/gcn_le2i_opt33_r2.yaml"
  "docs/project_targets/README.md"
  "docs/project_targets/CLAIM_TABLE.md"
  "docs/project_targets/THESIS_EVIDENCE_MAP.md"
  "docs/project_targets/FINAL_CANDIDATES.md"
  "docs/project_targets/STABILITY_REPORT.md"
  "docs/project_targets/SIGNIFICANCE_REPORT.md"
  "docs/project_targets/CROSS_DATASET_REPORT.md"
  "docs/project_targets/RESEARCH_QUESTIONS_MAPPING.md"
  "docs/project_targets/PAPER_CLAIMS_AND_LIMITATIONS_DRAFT.md"
  "docs/project_targets/PAPER_PUBLICATION_READINESS_PLAN.md"
  "docs/project_targets/PAPER_SUBMISSION_READINESS_CHECKLIST.md"
  "docs/project_targets/PROJECT_FINAL_YEAR_EXECUTION_PLAN.md"
  "docs/project_targets/DEPLOYMENT_LOCK.md"
  "docs/project_targets/DEPLOYMENT_DEFAULT_PROFILE.md"
  "docs/project_targets/DELIVERY_ALIGNMENT_STATUS.md"
  "docs/project_targets/FINAL_DEMO_WALKTHROUGH.md"
  "docs/project_targets/FINAL_SUBMISSION_CHECKLIST.md"
  "docs/project_targets/CLEAN_DRY_RUN_MINIMUM_PATH.md"
  "docs/project_targets/SUBMISSION_PACK_INDEX.md"
  "docs/project_targets/FIELD_VALIDATION_RUNBOOK.md"
  "docs/project_targets/FIELD_VALIDATION_MINIMUM_PACK.md"
  "docs/project_targets/DEPLOYMENT_FIELD_VALIDATION.md"
  "docs/project_targets/REPLAY_LIVE_ACCEPTANCE_LOCK.md"
  "docs/project_targets/LOCKED_PARAMS_RUNBOOK.md"
  "docs/project_targets/PLOT_EVIDENCE_CHECKLIST.md"
  "docs/project_targets/PROJECT_DELIVERY_EXCELLENCE_STANDARD.md"
  "artifacts/reports/stability_summary.csv"
  "artifacts/reports/stability_summary.json"
  "artifacts/reports/significance_summary.json"
  "artifacts/reports/cross_dataset_manifest.json"
  "artifacts/reports/cross_dataset_summary.csv"
  "artifacts/reports/cross_dataset_error_taxonomy.md"
  "artifacts/reports/online_mc_replay_matrix_20260402.csv"
  "artifacts/reports/online_mc_replay_matrix_20260402.json"
  "artifacts/reports/deployment_lock_validation.md"
  "artifacts/reports/deployment_field_eval.json"
  "artifacts/reports/deployment_field_failures.json"
  "artifacts/reports/deployment_field_validation_summary.md"
  "artifacts/reports/deployment_field_observations.csv"
  "artifacts/reports/clean_dry_run_report.md"
  "artifacts/reports/replay_live_acceptance.md"
  "artifacts/reports/release_snapshot.md"
  "artifacts/reports/release_bundle_status.json"
  "artifacts/figures/report/system_architecture_diagram.svg"
  "artifacts/figures/report/alert_policy_flow.svg"
  "artifacts/figures/report/offline_stability_comparison.png"
  "artifacts/figures/report/cross_dataset_transfer_summary.png"
  "artifacts/figures/report/cross_dataset_f1_comparison.png"
  "artifacts/figures/report/online_replay_accuracy_heatmap.png"
  "artifacts/figures/report/online_mc_dropout_delta.png"
  "artifacts/figures/report/stability_f1_errorbars.png"
)

missing=0

echo "[freeze-core-paths]"
printf '%s\n' "${FREEZE_CORE_PATHS[@]}"
echo

echo "[freeze-core-missing]"
for path in "${FREEZE_CORE_PATHS[@]}"; do
  if [[ ! -e "$path" ]]; then
    printf 'missing %s\n' "$path"
    missing=1
  fi
done
echo

echo "[freeze-core-git-status]"
git status --short -- "${FREEZE_CORE_PATHS[@]}"
echo

if [[ "$missing" -eq 0 ]]; then
  echo "freeze manifest check: all allowlisted paths exist"
else
  echo "freeze manifest check: missing allowlisted paths detected" >&2
  exit 1
fi
