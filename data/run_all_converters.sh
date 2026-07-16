#!/usr/bin/env bash
# Runs every GLiNER data/convert_*.py converter, writing GLiNER-schema
# JSONL into $OUT_DIR (default: data/). Mirrors the converter reference
# tables in docs/data_converters.md and docs/events.md#data-converters.
#
# Every dataset ends up with three sibling files: <name>.train.jsonl,
# <name>.val.jsonl, <name>.test.jsonl. Where the source ships its own
# real train/val/test (or train/val) split, those real splits are used --
# each is a separate converter invocation below, writing directly to the
# matching <name>.{train,val,test}.jsonl name regardless of what the
# source itself calls that split (dev/validation/valid all become "val").
# Where the source has no such split, the converter is invoked once and
# the new SplitWriter mechanism in data/_jsonl.py deterministically
# carves a fresh 80/10/10 train/val/test out of the single pass (seeded,
# reproducible -- override via --split-ratios/--split-seed, see the
# converters' own --help). This mirrors GLiNER2's tools/data/_split.py.
#
# A source with train+val but no *usable* test (DocRED's test split ships
# unlabeled; LEVEN/MAVEN's test ships stripped-annotation "candidates")
# is left with no test.jsonl -- fabricating one from train would silently
# misrepresent the split, so this is a documented gap, not something this
# script papers over.
#
# Converters whose source auto-downloads (HuggingFace / GitHub-raw / NCBI
# FTP) run unconditionally. Converters needing a manually obtained source
# (LDC license, gated Google Drive, or a repo the converter doesn't fetch
# itself) run only if the expected local path already exists, and are
# skipped with a pointer to the source otherwise. Each of those paths is
# overridable via the environment variable named next to it below; the
# val/test sibling paths derive from it (e.g. RAMS_INPUT's "train" ->
# "dev"/"test") and are documented, not independently overridable.
#
# A failing converter does not stop the rest of the batch; failures are
# collected and reported in the summary, and the script exits non-zero if
# any occurred.
#
# Usage:
#   uv sync --extra data
#   ./data/run_all_converters.sh
#   OUT_DIR=/somewhere/else ./data/run_all_converters.sh
#   RAMS_INPUT=/path/to/RAMS_1.0c/data/train.jsonlines ./data/run_all_converters.sh

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT_DIR="${OUT_DIR:-data}"
mkdir -p "$OUT_DIR"

done_list=()
failed_list=()
skipped_list=()

run() {
    local name="$1"
    shift
    echo
    echo "=== $name ==="
    if uv run python "$@"; then
        done_list+=("$name")
    else
        echo ">>> FAILED: $name" >&2
        failed_list+=("$name")
    fi
}

skip() {
    local name="$1" path="$2" note="$3"
    echo
    echo "=== $name === SKIPPED"
    echo "    expected source at: $path"
    echo "    $note"
    skipped_list+=("$name")
}

# ---------------------------------------------------------------------------
# Auto-downloading, single-source converters -- no real train/val/test split
# exists upstream, so each gets a fresh synthetic 80/10/10 in one pass.
# --out is a base name; write_jsonl_split derives .train/.val/.test.jsonl.
# ---------------------------------------------------------------------------

run biomed_ner           data/convert_biomed_ner.py           --out "$OUT_DIR/biomed_ner.jsonl"
run casie                data/convert_casie.py                --out "$OUT_DIR/casie.jsonl"
run gliner_multilingual  data/convert_gliner_multilingual.py  --out "$OUT_DIR/gliner_multilingual.jsonl"
run klue_ner             data/convert_klue_ner.py             --out "$OUT_DIR/klue_ner.jsonl"
run klue_re              data/convert_klue_re.py              --out "$OUT_DIR/klue_re.jsonl"
run knowledgator_gliner  data/convert_knowledgator_gliner.py  --out "$OUT_DIR/knowledgator_gliner.jsonl"
run nuner                data/convert_nuner.py --split full   --out "$OUT_DIR/nuner.jsonl"
run paraloq_json         data/convert_paraloq_json.py         --out "$OUT_DIR/paraloq_json.jsonl"
run pile_ner_definition  data/convert_pile_ner_definition.py  --out "$OUT_DIR/pile_ner_definition.jsonl"
run pubmed_abstracts_ner data/convert_pubmed_abstracts_ner.py --out "$OUT_DIR/pubmed_abstracts_ner.jsonl"
run sentence_rex         data/convert_sentence_rex.py         --out "$OUT_DIR/sentence_rex.jsonl"
run stockmark_ner        data/convert_stockmark_ner.py        --out "$OUT_DIR/stockmark_ner.jsonl"
run text2json            data/convert_text2json.py            --out "$OUT_DIR/text2json.jsonl"

# ---------------------------------------------------------------------------
# Auto-downloading converters with a real split on the source -- one
# invocation per available split, each writing straight to its
# <name>.{train,val,test}.jsonl name.
# ---------------------------------------------------------------------------

run bio_ner_relations_train data/convert_bio_ner_relations.py --split train --out "$OUT_DIR/bio_ner_relations.train.jsonl"
run bio_ner_relations_val   data/convert_bio_ner_relations.py --split val   --out "$OUT_DIR/bio_ner_relations.val.jsonl"
run bio_ner_relations_test  data/convert_bio_ner_relations.py --split test  --out "$OUT_DIR/bio_ner_relations.test.jsonl"

run biored_train data/convert_biored.py --split train --out "$OUT_DIR/biored.train.jsonl"
run biored_val   data/convert_biored.py --split dev   --out "$OUT_DIR/biored.val.jsonl"
run biored_test  data/convert_biored.py --split test  --out "$OUT_DIR/biored.test.jsonl"

# DocRED's test split ships with labels stripped (Kaggle-style held-out
# eval) -- unusable for local scoring, so only train/val are produced.
run docred_train data/convert_docred.py --split train      --out "$OUT_DIR/docred.train.jsonl"
run docred_val   data/convert_docred.py --split validation --out "$OUT_DIR/docred.val.jsonl"

run finer_ord_train data/convert_finer_ord.py --split train      --out "$OUT_DIR/finer_ord.train.jsonl"
run finer_ord_val   data/convert_finer_ord.py --split validation --out "$OUT_DIR/finer_ord.val.jsonl"
run finer_ord_test  data/convert_finer_ord.py --split test       --out "$OUT_DIR/finer_ord.test.jsonl"

run kaznerd_train data/convert_hf_token_ner.py --repo yeshpanovrustem/kaznerd --split train      --out "$OUT_DIR/kaznerd.train.jsonl"
run kaznerd_val   data/convert_hf_token_ner.py --repo yeshpanovrustem/kaznerd --split validation --out "$OUT_DIR/kaznerd.val.jsonl"
run kaznerd_test  data/convert_hf_token_ner.py --repo yeshpanovrustem/kaznerd --split test       --out "$OUT_DIR/kaznerd.test.jsonl"

run bc5cdr_train data/convert_hf_token_ner.py --repo tner/bc5cdr --revision refs/convert/parquet \
                      --tags-col tags --label-file dataset/label.json --split train      --out "$OUT_DIR/bc5cdr.train.jsonl"
run bc5cdr_val   data/convert_hf_token_ner.py --repo tner/bc5cdr --revision refs/convert/parquet \
                      --tags-col tags --label-file dataset/label.json --split validation --out "$OUT_DIR/bc5cdr.val.jsonl"
run bc5cdr_test  data/convert_hf_token_ner.py --repo tner/bc5cdr --revision refs/convert/parquet \
                      --tags-col tags --label-file dataset/label.json --split test       --out "$OUT_DIR/bc5cdr.test.jsonl"

run redocred_train data/convert_redocred.py --split train      --out "$OUT_DIR/redocred.train.jsonl"
run redocred_val   data/convert_redocred.py --split validation --out "$OUT_DIR/redocred.val.jsonl"
run redocred_test  data/convert_redocred.py --split test       --out "$OUT_DIR/redocred.test.jsonl"

# ~695 MB tarball, downloaded fresh per split (no local caching between
# these three invocations) -- pass --json processed_data/json/<split>.json
# instead if you've already extracted the tarball once.
run scierc_train data/convert_scierc.py --split train --out "$OUT_DIR/scierc.train.jsonl"
run scierc_val   data/convert_scierc.py --split dev   --out "$OUT_DIR/scierc.val.jsonl"
run scierc_test  data/convert_scierc.py --split test  --out "$OUT_DIR/scierc.test.jsonl"

run wikievents_train data/convert_wikievents.py --split train --out "$OUT_DIR/wikievents.train.jsonl"
run wikievents_val   data/convert_wikievents.py --split dev   --out "$OUT_DIR/wikievents.val.jsonl"
run wikievents_test  data/convert_wikievents.py --split test  --out "$OUT_DIR/wikievents.test.jsonl"

# ---------------------------------------------------------------------------
# Converters needing a manually obtained source.
# ---------------------------------------------------------------------------

ACE2005_INPUT="${ACE2005_INPUT:-ace_2005_td_v7/data/English}"
if [[ -d "$ACE2005_INPUT" ]]; then
    run ace2005 data/convert_ace2005.py --input "$ACE2005_INPUT" --out "$OUT_DIR/ace2005.jsonl"
else
    skip ace2005 "$ACE2005_INPUT" \
        "LDC2006T06 (paid LDC license) -- no auto-download possible. Set ACE2005_INPUT to override. No real split -- ace2005.jsonl gets a synthetic 80/10/10."
fi

# CMNEE ships canonical train/valid/test splits (Zhu et al., LREC-COLING
# 2024) -- val/test paths derive from CMNEE_INPUT by swapping "train" for
# "valid"/"test"; set them individually if your local layout differs.
CMNEE_INPUT="${CMNEE_INPUT:-data/cmnee/CMNEE/train.json}"
if [[ -f "$CMNEE_INPUT" ]]; then
    run cmnee_train data/convert_cmnee.py --input "$CMNEE_INPUT" --out "$OUT_DIR/cmnee.train.jsonl"
    CMNEE_VAL_INPUT="${CMNEE_VAL_INPUT:-${CMNEE_INPUT/train/valid}}"
    CMNEE_TEST_INPUT="${CMNEE_TEST_INPUT:-${CMNEE_INPUT/train/test}}"
    [[ -f "$CMNEE_VAL_INPUT" ]] && run cmnee_val data/convert_cmnee.py --input "$CMNEE_VAL_INPUT" --out "$OUT_DIR/cmnee.val.jsonl" \
        || skip cmnee_val "$CMNEE_VAL_INPUT" "expected alongside CMNEE_INPUT. Set CMNEE_VAL_INPUT to override."
    [[ -f "$CMNEE_TEST_INPUT" ]] && run cmnee_test data/convert_cmnee.py --input "$CMNEE_TEST_INPUT" --out "$OUT_DIR/cmnee.test.jsonl" \
        || skip cmnee_test "$CMNEE_TEST_INPUT" "expected alongside CMNEE_INPUT. Set CMNEE_TEST_INPUT to override."
else
    skip cmnee "$CMNEE_INPUT" \
        "Google Drive (gdown) -- see the CMNEE repo README (Zhu et al., LREC-COLING 2024). Set CMNEE_INPUT to override."
fi

# DocEE's normal_setting/ directory is documented upstream as shipping
# train/dev/test.json siblings -- not verified against a real download in
# this environment (Google-Drive-distributed), unlike the HF-hosted
# datasets above. Set the *_INPUT vars individually if your layout differs.
DOCEE_INPUT="${DOCEE_INPUT:-data/DocEE-en/normal_setting/train.json}"
if [[ -f "$DOCEE_INPUT" ]]; then
    run docee_train data/convert_docee.py --input "$DOCEE_INPUT" --out "$OUT_DIR/docee.train.jsonl"
    DOCEE_VAL_INPUT="${DOCEE_VAL_INPUT:-${DOCEE_INPUT/train/dev}}"
    DOCEE_TEST_INPUT="${DOCEE_TEST_INPUT:-${DOCEE_INPUT/train/test}}"
    [[ -f "$DOCEE_VAL_INPUT" ]] && run docee_val data/convert_docee.py --input "$DOCEE_VAL_INPUT" --out "$OUT_DIR/docee.val.jsonl" \
        || skip docee_val "$DOCEE_VAL_INPUT" "expected alongside DOCEE_INPUT (unverified in this environment). Set DOCEE_VAL_INPUT to override."
    [[ -f "$DOCEE_TEST_INPUT" ]] && run docee_test data/convert_docee.py --input "$DOCEE_TEST_INPUT" --out "$OUT_DIR/docee.test.jsonl" \
        || skip docee_test "$DOCEE_TEST_INPUT" "expected alongside DOCEE_INPUT (unverified in this environment). Set DOCEE_TEST_INPUT to override."
else
    skip docee "$DOCEE_INPUT" \
        "Google Drive -- see https://github.com/tongmeihan1995/docee. Set DOCEE_INPUT to override."
fi

# LEVEN's test.jsonl ships stripped-annotation "candidates", not real
# events -- only train/valid are usable, so no test is produced.
LEVEN_INPUT="${LEVEN_INPUT:-data/LEVEN/train.jsonl}"
if [[ -f "$LEVEN_INPUT" ]]; then
    run leven_train data/convert_leven.py --input "$LEVEN_INPUT" --out "$OUT_DIR/leven.train.jsonl"
    LEVEN_VAL_INPUT="${LEVEN_VAL_INPUT:-${LEVEN_INPUT/train/valid}}"
    [[ -f "$LEVEN_VAL_INPUT" ]] && run leven_val data/convert_leven.py --input "$LEVEN_VAL_INPUT" --out "$OUT_DIR/leven.val.jsonl" \
        || skip leven_val "$LEVEN_VAL_INPUT" "expected alongside LEVEN_INPUT. Set LEVEN_VAL_INPUT to override."
else
    skip leven "$LEVEN_INPUT" \
        "see https://github.com/thunlp/LEVEN. Set LEVEN_INPUT to override."
fi

# MAVEN's test split is blind (no public annotations) -- only train/valid
# are usable, so no test is produced.
MAVEN_INPUT="${MAVEN_INPUT:-data/maven/train.jsonl}"
if [[ -f "$MAVEN_INPUT" ]]; then
    run maven_train data/convert_maven.py --input "$MAVEN_INPUT" --out "$OUT_DIR/maven.train.jsonl"
    MAVEN_VAL_INPUT="${MAVEN_VAL_INPUT:-${MAVEN_INPUT/train/valid}}"
    [[ -f "$MAVEN_VAL_INPUT" ]] && run maven_val data/convert_maven.py --input "$MAVEN_VAL_INPUT" --out "$OUT_DIR/maven.val.jsonl" \
        || skip maven_val "$MAVEN_VAL_INPUT" "expected alongside MAVEN_INPUT. Set MAVEN_VAL_INPUT to override."
else
    skip maven "$MAVEN_INPUT" \
        "see https://github.com/THU-KEG/MAVEN-dataset. Set MAVEN_INPUT to override."
fi

# RAMS_1.0c's official tarball ships data/{train,dev,test}.jsonlines --
# not verified against a real download in this environment.
RAMS_INPUT="${RAMS_INPUT:-data/RAMS_1.0c/data/train.jsonlines}"
if [[ -f "$RAMS_INPUT" ]]; then
    run rams_train data/convert_rams.py --input "$RAMS_INPUT" --out "$OUT_DIR/rams.train.jsonl"
    RAMS_VAL_INPUT="${RAMS_VAL_INPUT:-${RAMS_INPUT/train/dev}}"
    RAMS_TEST_INPUT="${RAMS_TEST_INPUT:-${RAMS_INPUT/train/test}}"
    [[ -f "$RAMS_VAL_INPUT" ]] && run rams_val data/convert_rams.py --input "$RAMS_VAL_INPUT" --out "$OUT_DIR/rams.val.jsonl" \
        || skip rams_val "$RAMS_VAL_INPUT" "expected alongside RAMS_INPUT (unverified in this environment). Set RAMS_VAL_INPUT to override."
    [[ -f "$RAMS_TEST_INPUT" ]] && run rams_test data/convert_rams.py --input "$RAMS_TEST_INPUT" --out "$OUT_DIR/rams.test.jsonl" \
        || skip rams_test "$RAMS_TEST_INPUT" "expected alongside RAMS_INPUT (unverified in this environment). Set RAMS_TEST_INPUT to override."
else
    skip rams "$RAMS_INPUT" \
        "see https://nlp.jhu.edu/rams/ (RAMS_1.0c.tar.gz). Set RAMS_INPUT to override."
fi

# No real split exists (single pre-converted local JSONL) -- synthetic
# 80/10/10, same as the auto-downloading single-source converters above.
EVENTS_BIOTECH_INPUT="${EVENTS_BIOTECH_INPUT:-data/events_biotech.jsonl}"
if [[ -n "$EVENTS_BIOTECH_INPUT" && -f "$EVENTS_BIOTECH_INPUT" ]]; then
    run events_biotech data/convert_events_biotech.py --input "$EVENTS_BIOTECH_INPUT" --out "$OUT_DIR/events_biotech.jsonl"
else
    skip events_biotech "\$EVENTS_BIOTECH_INPUT (unset)" \
        "requires GLiNER2's own converted events_biotech JSONL (input/output/classifications shape) as input -- run GLiNER2's tools/data converter first, then set EVENTS_BIOTECH_INPUT."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo
echo "===== Summary ====="
echo "done:    ${#done_list[@]}  (${done_list[*]:-none})"
echo "skipped: ${#skipped_list[@]}  (${skipped_list[*]:-none})"
echo "failed:  ${#failed_list[@]}  (${failed_list[*]:-none})"

if [[ ${#failed_list[@]} -gt 0 ]]; then
    exit 1
fi
