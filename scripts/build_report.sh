#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_INPUT="$ROOT_DIR/docs/reports/drafts/HIGH_STANDARD_FINAL_PROJECT_REPORT_DRAFT_2026-03-29.md"
DEFAULT_OUT_DIR="$ROOT_DIR/artifacts/report_build"
DEFAULT_TITLE="Pose-Based Fall Detection with Temporal and Graph Neural Models"
DEFAULT_AUTHOR="Ru Zhan"
DEFAULT_DATE="$(date +%F)"

show_help() {
  cat <<'EOF'
Usage:
  ./scripts/build_report.sh [input_md] [out_dir] [base_name] [--pdf-only|--docx-only]

Examples:
  ./scripts/build_report.sh
  ./scripts/build_report.sh docs/reports/drafts/HIGH_STANDARD_FINAL_PROJECT_REPORT_DRAFT_2026-03-29.md
  ./scripts/build_report.sh "" "" final_report --pdf-only

Notes:
  - default input is the current high-standard report draft
  - default output directory is artifacts/report_build
  - default base name is derived from the input filename plus today's date
  - environment overrides:
      REPORT_TITLE
      REPORT_AUTHOR
      REPORT_DATE
      REPORT_REFERENCE_DOCX
EOF
}

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required tool '$1' is not available in PATH" >&2
    exit 1
  fi
}

input_md="$DEFAULT_INPUT"
out_dir="$DEFAULT_OUT_DIR"
base_name=""
format_flag=""

positionals=()
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      show_help
      exit 0
      ;;
    --pdf-only|--docx-only)
      if [[ -n "$format_flag" ]]; then
        echo "error: multiple format flags provided" >&2
        show_help >&2
        exit 2
      fi
      format_flag="$arg"
      ;;
    *)
      positionals+=("$arg")
      ;;
  esac
done

if (( ${#positionals[@]} > 3 )); then
  echo "error: too many positional arguments" >&2
  show_help >&2
  exit 2
fi

if (( ${#positionals[@]} >= 1 )) && [[ -n "${positionals[0]}" ]]; then
  input_md="${positionals[0]}"
fi
if (( ${#positionals[@]} >= 2 )) && [[ -n "${positionals[1]}" ]]; then
  out_dir="${positionals[1]}"
fi
if (( ${#positionals[@]} >= 3 )) && [[ -n "${positionals[2]}" ]]; then
  base_name="${positionals[2]}"
fi

if [[ ! -f "$input_md" ]]; then
  echo "error: input markdown not found: $input_md" >&2
  exit 1
fi

require_tool pandoc
if [[ "$format_flag" != "--docx-only" ]]; then
  require_tool xelatex
fi

mkdir -p "$out_dir"

input_stem="$(basename "$input_md")"
input_stem="${input_stem%.md}"
if [[ -z "${base_name:-}" ]]; then
  base_name="${input_stem}_$DEFAULT_DATE"
fi

docx_out="$out_dir/$base_name.docx"
pdf_out="$out_dir/$base_name.pdf"
report_title="${REPORT_TITLE:-$DEFAULT_TITLE}"
report_author="${REPORT_AUTHOR:-$DEFAULT_AUTHOR}"
report_date="${REPORT_DATE:-$DEFAULT_DATE}"
reference_docx="${REPORT_REFERENCE_DOCX:-}"

common_args=(
  --from markdown+table_captions
  --standalone
  --toc
  --number-sections
  --resource-path="$ROOT_DIR"
  --metadata=title:"$report_title"
  --metadata=author:"$report_author"
  --metadata=date:"$report_date"
)

docx_args=("${common_args[@]}")
if [[ -n "$reference_docx" ]]; then
  if [[ ! -f "$reference_docx" ]]; then
    echo "error: REPORT_REFERENCE_DOCX points to a missing file: $reference_docx" >&2
    exit 1
  fi
  docx_args+=(--reference-doc="$reference_docx")
fi

if [[ "$format_flag" != "--pdf-only" ]]; then
  pandoc "$input_md" "${docx_args[@]}" -o "$docx_out"
fi

if [[ "$format_flag" != "--docx-only" ]]; then
  pandoc "$input_md" "${common_args[@]}" --pdf-engine=xelatex -V geometry:margin=1in -o "$pdf_out"
fi

echo "Built report assets:"
[[ "$format_flag" != "--pdf-only" ]] && echo "  DOCX: $docx_out"
[[ "$format_flag" != "--docx-only" ]] && echo "  PDF : $pdf_out"
