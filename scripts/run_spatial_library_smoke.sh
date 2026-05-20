#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
TEST_FILE="$ROOT/tests/spatial/test_library_only_skills.py"
SKILL=all
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_spatial_library_smoke.sh [--skill <name>] [--dry-run] [--list]

Options:
  --skill <name>   one of: all, niches, regulation, sc2spatial, histology, wsi,
                   morphology, translation, oncology, targets, tls
  --dry-run        print the pytest command without executing it
  --list           show supported skill names
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skill)
      SKILL="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --list)
      echo "all"
      echo "niches"
      echo "regulation"
      echo "sc2spatial"
      echo "histology"
      echo "wsi"
      echo "morphology"
      echo "translation"
      echo "oncology"
      echo "targets"
      echo "tls"
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$SKILL" in
  all) FILTER="" ;;
  niches) FILTER="test_spatial_niches_smoke" ;;
  regulation) FILTER="test_spatial_regulation_smoke" ;;
  sc2spatial) FILTER="test_spatial_sc2spatial_smoke" ;;
  histology) FILTER="test_spatial_histology_smoke" ;;
  wsi) FILTER="test_spatial_wsi_smoke" ;;
  morphology) FILTER="test_spatial_morphology_smoke" ;;
  translation) FILTER="test_spatial_translation_smoke" ;;
  oncology) FILTER="test_spatial_oncology_smoke" ;;
  targets) FILTER="test_spatial_targets_smoke" ;;
  tls) FILTER="test_spatial_tls_smoke" ;;
  *)
    echo "Unsupported skill: $SKILL" >&2
    usage >&2
    exit 1
    ;;
esac

CMD=(pytest "$TEST_FILE")
if [[ -n "$FILTER" ]]; then
  CMD+=(-k "$FILTER")
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

cd "$ROOT"
"${CMD[@]}"
