#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_MD="${1:-$ROOT_DIR/docs/reports/drafts/HIGH_STANDARD_FINAL_PROJECT_REPORT_DRAFT_2026-03-29.md}"
OUT_DIR="${2:-$ROOT_DIR/artifacts/report_build}"
BASE_NAME="${3:-HIGH_STANDARD_FINAL_PROJECT_REPORT_2026-04-05}"

mkdir -p "$OUT_DIR"

DOCX_OUT="$OUT_DIR/$BASE_NAME.docx"
PDF_OUT="$OUT_DIR/$BASE_NAME.pdf"

common_args=(
  --from markdown+table_captions
  --standalone
  --toc
  --number-sections
  --resource-path="$ROOT_DIR"
  --metadata=title:"Pose-Based Fall Detection with Temporal and Graph Neural Models"
  --metadata=author:"Ru Zhan"
  --metadata=date:"2026-04-05"
)

pandoc "$INPUT_MD" "${common_args[@]}" -o "$DOCX_OUT"
pandoc "$INPUT_MD" "${common_args[@]}" --pdf-engine=xelatex -V geometry:margin=1in -o "$PDF_OUT"

echo "Built:"
echo "  $DOCX_OUT"
echo "  $PDF_OUT"
