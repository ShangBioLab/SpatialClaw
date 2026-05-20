#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GROUP="core"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)
      GROUP="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage:
  bash scripts/run_spatial_cli_smoke.sh --group <core|reference|metadata|routing|all> [--dry-run]

Environment overrides:
  ST_INPUT
  SCRNA_REF
  FIXTURE_DIR
  OUTPUT_ROOT
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

ST_INPUT="${ST_INPUT:-$ROOT/data/deconvolution/Human_Lymph_Node/ST.h5ad}"
SCRNA_REF="${SCRNA_REF:-$ROOT/data/deconvolution/Human_Lymph_Node/scRNA.h5ad}"
FIXTURE_DIR="${FIXTURE_DIR:-$ROOT/data/test_fixtures/spatial}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/output/skill_tests}"

mkdir -p "$OUTPUT_ROOT"

run_cmd() {
  echo
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

ensure_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    echo "Generate fixtures first with: python scripts/generate_spatial_test_fixtures.py" >&2
    exit 1
  fi
}

run_core() {
  ensure_file "$ST_INPUT"

  run_cmd python spatialclaw.py run spatial-preprocessing \
    --input "$ST_INPUT" \
    --output "$OUTPUT_ROOT/00_preprocess"

  local processed="$OUTPUT_ROOT/00_preprocess/processed.h5ad"

  run_cmd python spatialclaw.py run spatial-domain-identification --input "$processed" --output "$OUTPUT_ROOT/01_domains"
  run_cmd python spatialclaw.py run spatial-svg-detection --input "$processed" --output "$OUTPUT_ROOT/02_genes"
  run_cmd python spatialclaw.py run spatial-statistics --input "$processed" --output "$OUTPUT_ROOT/03_statistics"
  run_cmd python spatialclaw.py run spatial-de --input "$processed" --output "$OUTPUT_ROOT/04_de"
  run_cmd python spatialclaw.py run spatial-cell-communication --input "$processed" --output "$OUTPUT_ROOT/05_communication"
  run_cmd python spatialclaw.py run spatial-enrichment --input "$processed" --output "$OUTPUT_ROOT/06_enrichment"
  run_cmd python spatialclaw.py run spatial-cell-annotation --input "$processed" --output "$OUTPUT_ROOT/07_annotate_marker" --method marker_based
  run_cmd python spatialclaw.py run spatial-cnv --input "$processed" --output "$OUTPUT_ROOT/08_cnv"
  run_cmd python spatialclaw.py run spatial-trajectory --input "$processed" --output "$OUTPUT_ROOT/09_trajectory"
}

run_reference() {
  ensure_file "$SCRNA_REF"
  ensure_file "$OUTPUT_ROOT/00_preprocess/processed.h5ad"

  local processed="$OUTPUT_ROOT/00_preprocess/processed.h5ad"

  run_cmd python spatialclaw.py run spatial-cell-annotation \
    --input "$processed" \
    --reference "$SCRNA_REF" \
    --output "$OUTPUT_ROOT/10_annotate_tangram" \
    --method tangram

  run_cmd python spatialclaw.py run spatial-deconvolution \
    --input "$processed" \
    --reference "$SCRNA_REF" \
    --output "$OUTPUT_ROOT/11_deconvolution"
}

run_metadata() {
  ensure_file "$FIXTURE_DIR/multi_sample_raw.h5ad"
  ensure_file "$FIXTURE_DIR/multi_slice_raw.h5ad"
  ensure_file "$FIXTURE_DIR/velocity_fixture.h5ad"

  run_cmd python spatialclaw.py run spatial-preprocessing \
    --input "$FIXTURE_DIR/multi_sample_raw.h5ad" \
    --output "$OUTPUT_ROOT/20_preprocess_multi_sample"

  run_cmd python spatialclaw.py run spatial-condition-comparison \
    --input "$OUTPUT_ROOT/20_preprocess_multi_sample/processed.h5ad" \
    --output "$OUTPUT_ROOT/21_condition" \
    --condition-key condition \
    --sample-key sample_id

  run_cmd python spatialclaw.py run spatial-integration \
    --input "$OUTPUT_ROOT/20_preprocess_multi_sample/processed.h5ad" \
    --output "$OUTPUT_ROOT/22_integrate" \
    --batch-key batch

  run_cmd python spatialclaw.py run spatial-preprocessing \
    --input "$FIXTURE_DIR/multi_slice_raw.h5ad" \
    --output "$OUTPUT_ROOT/23_preprocess_multi_slice"

  run_cmd python spatialclaw.py run spatial-registration \
    --input "$OUTPUT_ROOT/23_preprocess_multi_slice/processed.h5ad" \
    --output "$OUTPUT_ROOT/24_register"

  run_cmd python spatialclaw.py run spatial-velocity \
    --input "$FIXTURE_DIR/velocity_fixture.h5ad" \
    --output "$OUTPUT_ROOT/25_velocity"
}

run_routing() {
  ensure_file "$ST_INPUT"

  run_cmd python skills/spatial/spatial-orchestrator/spatial_orchestrator.py --list-skills
  run_cmd python skills/spatial/spatial-orchestrator/spatial_orchestrator.py \
    --query "find spatially variable genes" \
    --output "$OUTPUT_ROOT/30_orchestrator_query"
  run_cmd python skills/spatial/spatial-orchestrator/spatial_orchestrator.py \
    --pipeline standard \
    --input "$ST_INPUT" \
    --output "$OUTPUT_ROOT/31_orchestrator_standard"
}

case "$GROUP" in
  core)
    run_core
    ;;
  reference)
    run_reference
    ;;
  metadata)
    run_metadata
    ;;
  routing)
    run_routing
    ;;
  all)
    run_core
    run_reference
    run_metadata
    run_routing
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 1
    ;;
esac
