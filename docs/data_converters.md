# Data Converters

`data/convert_*.py` in the repo root ports GLiNER2's `tools/data/` dataset
converters to GLiNER's own training schema. GLiNER2's converters emit a
surface-form-string schema (`{"input": text, "output": {"entities": {...}}}`)
built for its own runtime prompting; GLiNER needs actual token-index spans
(`{"tokenized_text": [...], "ner": [[start, end, label], ...], "relations":
[[head_idx, tail_idx, type], ...]}`), so these are not 1:1 wrappers around
GLiNER2's converters -- most re-derive spans from the *original* raw source
data rather than reshape GLiNER2's already-lossy output.

For the 9 event-specific converters (trigger/argument schema) see
[Event Extraction](events.md#data-converters). This page covers the other
18 datasets (19 converter files -- KLUE's NER and RE tasks are split into
`klue_ner.py` / `klue_re.py`).

## Fidelity tiers

Converters fall into three tiers depending on what the raw source actually
provides. Higher tiers are preferred wherever the source supports them --
going back to raw, offset-bearing data avoids a silent trap: reshaping an
already-surface-bucketed intermediate (like GLiNER2's own output) collapses
every repeated mention of an entity to one representative surface, which is
harmless for single-mention entities but silently wrong for document-level
relation datasets where a relation links a *specific* mention, not just an
entity's first occurrence.

1. **Token-indexed at the source** -- no mapping needed, spans copied
   directly: `gliner_multilingual`, `knowledgator_gliner`, `finer_ord`,
   `hf_token_ner`, `scierc`, `pubmed_abstracts_ner`.
2. **Char-offset at the source** -- mapped to token spans via
   `data/_charspan.py`'s `char_span_to_token_span`: `biomed_ner`, `biored`,
   `bio_ner_relations`, `klue_ner`, `klue_re`, `stockmark_ner`. For
   document-level relation datasets with coreference clusters (`docred`,
   `redocred`, `biored`), a relation between two entity *clusters* is
   anchored to a representative mention (the cluster's first mention) --
   correct here, not a limitation, since the relation itself is
   cluster-to-cluster, not mention-to-mention.
3. **Surface-only at the source** -- no offsets exist at all; located via
   `data/_charspan.py`'s `find_surface_span` (first verbatim occurrence).
   This is the achievable ceiling for these sources, not a shortcut taken
   in place of something better: `nuner`, `paraloq_json`, `pile_ner_definition`,
   `text2json`, `sentence_rex`. Verified per-converter against real data
   that this doesn't silently drop most of the signal (e.g. NuNER: ~85% of
   entity mentions found verbatim in 30 sampled real rows).

Chinese/Japanese/Korean sources (`cmnee`, `klue_ner`, `klue_re`,
`stockmark_ner`) have no whitespace word boundaries, so `tokenized_text` is
`list(text)` (one character per token) rather than word-tokenized -- this
makes a char offset *equal* to its token index directly, no mapping step
needed even in the char-offset tier.

## `sentence_rex`: inline-tagged entities, and a judgment call

`sentence_rex` is the one converter with a genuinely different source shape
-- entities are marked inline with `<e1>...</e1>` / `<e2>...</e2>` tags
rather than given as offsets or a surface list. Tags must be stripped
*before* tokenizing; tokenizing tagged text first (or computing spans
against it) silently shifts every downstream position once the tag tokens
are removed.

The source also has no independent entity typing for `e1`/`e2` -- only the
relation label connecting them, and unlike RAMS (where the role name at
least distinguishes head from tail) there's no role-like field either.
Rather than invent a false distinction, both spans get a single placeholder
NER type, `"entity"` -- an explicit choice, documented here and in the
converter's own docstring, not a default that happened silently.

## Skipped: `gliclass_logic`, `scientific_text`

Both are pure text classification (whole-document label, no spans at all).
GLiNER has no classification head to train this data against -- porting
them would require new architecture work, not a converter. Not attempted
here; flagged as a scope boundary rather than force-fit into the NER/RE
schema the way the event track's `events_biotech` was (see
[Event Extraction](events.md#data-converters) for why that case was
different: it was explicitly requested, and the event architecture at least
has a trigger-token mechanism to hang a degenerate signal on).

## Setup

Most converters stream from HuggingFace and need the optional `data`
dependency group:

```bash
uv sync --extra data
```

A few need no extra dependency: `biored` and `klue_ner`/`klue_re` fetch
directly via stdlib `urllib`/`zipfile` (NCBI FTP, raw GitHub files);
`pubmed_abstracts_ner` and `text2json` use `huggingface_hub` (already a
core GLiNER dependency) to download one named file rather than going
through `datasets`.

## Converter reference

| Dataset | Task | Source | License |
|---|---|---|---|
| `gliner_multilingual` | NER | HF: `knowledgator/gliner-multilingual-synthetic` | -- |
| `knowledgator_gliner` | NER | HF: `knowledgator/GLINER-multi-task-synthetic-data` | -- |
| `finer_ord` | NER | HF: `gtfintechlab/finer-ord` | CC-BY-NC-4.0 |
| `hf_token_ner` | NER (generic) | HF, parametrized by `--repo` (kaznerd, bc4chemd, tner/bc5cdr, ...) | varies |
| `biomed_ner` | NER | HF: `knowledgator/biomed_NER` | -- |
| `stockmark_ner` | NER | HF: `stockmark/ner-wikipedia-dataset` | CC-BY-SA-3.0 |
| `pubmed_abstracts_ner` | NER | HF: `knowledgator/PubMedAbstractsNER` | -- |
| `klue_ner` | NER | GitHub: KLUE-benchmark release TSV | CC-BY-SA-4.0 |
| `nuner` | NER | HF: `numind/NuNER` | -- |
| `paraloq_json` | NER (schema-driven) | HF: `paraloq/json_data_extraction` | Apache-2.0 |
| `pile_ner_definition` | NER (free-text labels) | HF: `Universal-NER/Pile-NER-definition` | -- |
| `text2json` | NER (schema-driven) | HF: `knowledgator/text2json-training-data` | -- |
| `scierc` | NER + RE | AI2 SciERC (`sthoran/scierc_processed_data` mirror or the official tarball) | research use |
| `docred` | RE | HF: `thunlp/docred` (parquet revision) | -- |
| `redocred` | RE | HF: `tonytan48/Re-DocRED` | -- |
| `biored` | NER + RE | NCBI FTP: `BIORED.zip` | NCBI / NLM |
| `bio_ner_relations` | NER + RE | HF: `knowledgator/bio-NER-relations` | -- |
| `klue_re` | RE | GitHub: KLUE-benchmark release JSON | CC-BY-SA-4.0 |
| `sentence_rex` | RE | HF: `knowledgator/sentence_rex` | -- |

## Testing

Each converter has a hermetic unit test file (`tests/test_data_converters*.py`)
exercising its pure conversion logic against small synthetic inputs shaped
like the real raw source -- no network access needed to run the suite. Every
converter was additionally run against real data (streamed/downloaded from
its actual source) during development and spot-checked by decoding
converted spans back to surface text and comparing against the source
content, not just checking output shape.
