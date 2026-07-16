#!/usr/bin/env bash
#
# Train GLiNER on every dataset config in configs/.
#
# Runs `uv run python train.py --config configs/<name>.yaml` for each dataset
# config below. The config*.yaml base templates are deliberately excluded --
# they point at placeholder data (data.json), not real datasets.
#
# For each config it: skips it if the train_data hasn't been converted yet
# (run the matching data/convert_*.py first), otherwise trains and tees the
# output to logs/train_<name>.log. A failure does not stop the batch; a summary
# of PASSED / FAILED / SKIPPED prints at the end.
#
# Usage:
#   scripts/train_all.sh                    # all dataset configs
#   scripts/train_all.sh wikievents casie   # only the named configs
#
set -o pipefail
cd "$(dirname "$0")/.." || exit 1   # repo root

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Dataset configs, grouped by task. (config*.yaml templates excluded.)
CONFIGS=(
  # --- Event extraction (trigger/argument) ---
  wikievents ace2005 casie rams cmnee docee maven leven events_biotech
  # --- Named entity recognition ---
  bc5cdr biomed_ner finer_ord gliner_multilingual kaznerd klue_ner
  knowledgator_gliner nuner paraloq_json pile_ner_definition
  pubmed_abstracts_ner stockmark_ner text2json
  # --- Relation extraction ---
  bio_ner_relations biored docred redocred klue_re scierc sentence_rex
  # --- Aggregate (need every sub-dataset converted first) ---
  aggregate_events aggregate_ner aggregate_re
)

# Restrict to configs named on the command line, if any.
if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
fi

# Print every train_data path for a config (single path or a YAML list).
train_data_paths() {
  uv run python - "$1" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1])).get("data") or {}
td = data.get("train_data")
for p in ([td] if isinstance(td, str) else (td or [])):
    if p:
        print(p)
PY
}

PASSED=()
FAILED=()
SKIPPED=()

for name in "${CONFIGS[@]}"; do
  cfg="configs/${name}.yaml"
  if [ ! -f "$cfg" ]; then
    echo ">> SKIP $name  (config not found: $cfg)"
    SKIPPED+=("$name")
    continue
  fi

  # Skip when any train_data file is missing (dataset not converted yet).
  missing=""
  while IFS= read -r p; do
    if [ -n "$p" ] && [ ! -f "$p" ]; then
      missing="$p"
      break
    fi
  done < <(train_data_paths "$cfg")
  if [ -n "$missing" ]; then
    echo ">> SKIP $name  (missing data: $missing -- convert it first)"
    SKIPPED+=("$name")
    continue
  fi

  echo ""
  echo "==================== TRAIN: $name ===================="
  if uv run python train.py --config "$cfg" 2>&1 | tee "$LOG_DIR/train_${name}.log"; then
    echo "++ $name: done  (log: $LOG_DIR/train_${name}.log)"
    PASSED+=("$name")
  else
    echo "-- $name: FAILED  (see $LOG_DIR/train_${name}.log)"
    FAILED+=("$name")
  fi
done

echo ""
echo "==================== SUMMARY ===================="
echo "PASSED  (${#PASSED[@]}): ${PASSED[*]:-none}"
echo "FAILED  (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "SKIPPED (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
[ "${#FAILED[@]}" -eq 0 ]
