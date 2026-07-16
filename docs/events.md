# Event Extraction (Trigger/Argument)

Event extraction identifies **event triggers** (the word or phrase that
signals an event happened, e.g. "attacked", "acquired", "died") and, for
each trigger, the **arguments** that fill semantic roles around it (e.g.
`Attacker`, `Victim`, `Place`). Unlike relation extraction, where a fixed
head entity connects to a fixed tail entity, an event trigger can connect
to an arbitrary number of arguments with different roles.

GLiNER supports this as an opt-in extension of `UniEncoderSpanRelex`
(`event_mode=True` on `UniEncoderRelexConfig`) rather than a separate
architecture — it reuses the same span encoder, the same KGE triple-scoring
layer, and the same adjacency/relation losses used for entity-relation
extraction, changing only how spans are selected and paired.

## Architecture

Plain relation extraction selects **one** set of entity spans and builds
pairs from that single set (any entity can be a head or a tail). Events
need two **independent** selections — triggers and arguments live in
different label spaces and a trigger should never be paired with another
trigger.

`UniEncoderSpanRelexModel.represent_spans_bipartite` does this by calling
the existing `select_span_target_embedding` **twice**, once with the
class-score columns masked to trigger types (`trigger_class_mask`) and
once masked to everything else, producing genuinely separate
`trigger_rep` / `arg_rep` tensors instead of one merged entity set.

`BipartiteRelationsRepLayer` (`gliner/modeling/multitask/relations_layers.py`)
computes the `(B, num_triggers, num_args)` adjacency matrix from those two
tensors — the bipartite counterpart of `RelationsRepLayer`'s homogeneous
`(B, N, N)` adjacency. Only `"dot"`, `"mlp"`, and `"bilinear"` are supported
in this mode; `"gcn"` / `"gat"` / `"attention"` raise `ValueError` since
message-passing over one graph doesn't have a bipartite equivalent.

`build_trigger_argument_pairs` (`gliner/modeling/utils.py`) builds the
actual `(trigger, argument)` pairs fed to the relation classifier — the
bipartite counterpart of `build_entity_pairs`. Unlike the entity-pair
version, it needs no self-pair exclusion (trigger and argument index
spaces are independent), and builds the full trigger×argument cartesian
product when no adjacency matrix is given (inference), or filters by a
thresholded adjacency matrix otherwise.

Everything downstream — `TriplesScoreLayer` (TransE / DistMult / ComplEx /
...), `adj_loss`, `rel_loss` — is unchanged; it operates on `(head, tail)`
embeddings and doesn't care whether they came from a homogeneous or
bipartite selection. See "TransE and other norm-based `triples_layer`
options saturate at transformer scale" under Training below before picking
a `triples_layer` for real training — it's a pre-existing issue shared with
plain (non-event) relation extraction, not specific to events.

## Enabling event mode

```python
from gliner.config import UniEncoderSpanRelexConfig

config = UniEncoderSpanRelexConfig(
    model_name="microsoft/deberta-v3-base",
    relations_layer="dot",       # or "mlp" / "bilinear" -- not "gcn"/"gat"/"attention"
    triples_layer=None,          # concatenation scoring; see Training below re: TransE
    event_mode=True,
    ...
)
```

`event_mode` defaults to `False`; with it off, `forward()` is byte-for-byte
identical to the existing relation-extraction path. With it on, `forward()`
requires a `trigger_class_mask` (a `(B, C)` boolean tensor marking which of
the `C` label columns are trigger types) — this is built automatically by
`EventExtractionSpanDataCollator` / `EventExtractionSpanProcessor` at both
train and inference time.

## Data format

Same `{tokenized_text, ner, relations}` shape used for relation extraction,
with one convention change: `relations` entries are always
`[trigger_ner_idx, arg_ner_idx, role]` — the head is always a trigger, the
tail is always an argument.

```python
record = {
    "tokenized_text": ["John", "Smith", "attacked", "the", "embassy", "."],
    "ner": [
        [0, 1, "PER.Individual"],      # entity (also usable as an argument)
        [2, 2, "Conflict.Attack"],     # trigger
        [4, 4, "FAC.Building-Grounds"] # entity/argument
    ],
    "relations": [
        [1, 0, "Attacker"],  # trigger idx 1 -> argument idx 0
        [1, 2, "Target"],    # trigger idx 1 -> argument idx 2
    ],
}
```

`ner` entries that are never a relation head (`PER.Individual`,
`FAC.Building-Grounds` above) still contribute plain NER supervision — the
processor doesn't require every entity to be linked to a trigger.

### `EventExtractionSpanProcessor`

```python
from gliner.data_processing import EventExtractionSpanProcessor, EventExtractionSpanDataCollator

processor = EventExtractionSpanProcessor(
    config, tokenizer, words_splitter=None,
    trigger_types={"Conflict.Attack", "Life.Die.Unspecified", ...},
)
collator = EventExtractionSpanDataCollator(config, data_processor=processor, prepare_labels=True)
```

`trigger_types` tells the processor which `ner` labels are triggers vs.
arguments (`trigger_class_mask`) and which pairs are gold roles
(`create_event_labels`, iterated in the same row-major
`for trigger in triggers: for arg in arguments` order as
`build_trigger_argument_pairs` uses at train time — this alignment is the
single most safety-critical invariant in the pipeline; a misordering here
silently trains the model on shuffled role labels while every tensor shape
check still passes).

`num_triggers` / `num_args` count **distinct span positions, not entities.**
The model's bipartite selection (`select_span_target_embedding`) produces one
slot per `(start, end)` span, so when several `ner` entities share a span —
routine in real corpora with coreferent or overlapping annotations — they must
collapse to a single trigger/argument rank. Ranking per entity instead
overcounts `num_triggers`/`num_args`, so the gold `adj_matrix` `(B, max_T,
max_A)` no longer matches the model's `pred_adj_matrix` and `adj_loss` crashes
on the shape mismatch (`RuntimeError: The size of tensor a (...) must match ...`).
`EventExtractionSpanProcessor.preprocess_example` handles this by keying its
trigger/argument ranks on `(start, end)`.

### Deriving `trigger_types` — don't use a label-shape heuristic

The converted JSONL doesn't self-describe which `ner` labels are triggers.
**Don't** derive `trigger_types` from a heuristic like "labels containing a
dot are triggers" — that happens to work for WikiEvents
(`Life.Die.Unspecified`) but silently breaks for CMNEE, whose trigger types
are bare words (`Manoeuvre`) with the same shape as some argument roles. If
you get this wrong, `trigger_class_mask` comes out wrong, no shape check
catches it, and the model trains with zero real event supervision.

`gliner/data_processing/trigger_types.py` provides the dataset-agnostic
derivation (also re-exported from `data/_trigger_types.py` for the
converters): since `relations` entries are always `[trigger_idx, arg_idx,
role]`, a *sound* (never wrong) trigger vocabulary is the set of `ner`
labels that ever appear as a relation **head**:

```python
from gliner.data_processing.trigger_types import derive_trigger_types

trigger_types = derive_trigger_types(records)
```

**You usually don't call this yourself.** `train.py` and
`scripts/custom_train.py` auto-derive `trigger_types` from the raw training
records at startup (via `apply_derived_trigger_types`) whenever an
`event_mode` config leaves `trigger_types` empty — so the shipped event
configs can keep `trigger_types: []` and still train a live event head.
Deriving by hand (above) is only needed when driving the processor
directly, as in the low-level Training example below.

For every converter that has both triggers and relations (WikiEvents,
RAMS, CASIE, CMNEE, ACE2005), head and tail label sets are disjoint by
construction (event types vs. entity/role types) — this derivation never
misclassifies an argument type as a trigger. It is **not** guaranteed
*complete*, though: an event type that never happens to co-occur with any
argument in the records you derive from doesn't appear as a relation head
and gets silently dropped from the derived set (confirmed on the real
WikiEvents fixture — 2 of 24 event types present in `ner` never appear as a
head there). If a dataset ships a documented, fixed event-type vocabulary,
cross-check the derived set against it, or pass the known-complete
vocabulary explicitly instead of deriving it.

Datasets with **no relations at all** (MAVEN, LEVEN — trigger-only, no
argument annotations; events_biotech — synthetic trigger-per-label, no
arguments) have nothing to derive from at all. The training-script
auto-derivation **raises `ValueError`** at startup for these rather than
training a silently dead head, so set their `trigger_types` explicitly in
the config (or, when driving the processor directly, pass the vocabulary
via `derive_trigger_types(records, explicit={...})`).

## Training

```python
import torch
from transformers import AutoTokenizer
from gliner.config import UniEncoderSpanRelexConfig
from gliner.modeling.base import UniEncoderSpanRelexModel
from gliner.data_processing import EventExtractionSpanProcessor, EventExtractionSpanDataCollator

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_tokens(["<<ENT>>", "<<REL>>"], special_tokens=True)

config = UniEncoderSpanRelexConfig(
    model_name="bert-base-uncased",
    class_token_index=tokenizer.convert_tokens_to_ids("<<ENT>>"),
    rel_token_index=tokenizer.convert_tokens_to_ids("<<REL>>"),
    vocab_size=len(tokenizer),
    relations_layer="dot",
    triples_layer=None,  # see "TransE ... saturate" below before using a KGE triples_layer
    event_mode=True,
    max_types=60,  # see "max_types and trigger truncation" below
)

processor = EventExtractionSpanProcessor(config, tokenizer, words_splitter=None, trigger_types=trigger_types)
collator = EventExtractionSpanDataCollator(config, data_processor=processor, prepare_labels=True)

model = UniEncoderSpanRelexModel(config, from_pretrained=False)
model.token_rep_layer.resize_token_embeddings(len(tokenizer))  # required after add_tokens
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
for records_batch in ...:
    batch = collator(records_batch)
    optimizer.zero_grad()
    output = model(**batch)
    output.loss.backward()
    optimizer.step()
```

Two things every training run must get right, both easy to get silently
wrong:

- **`<<ENT>>` / `<<REL>>` must be registered as real special tokens**
  (`tokenizer.add_tokens([...], special_tokens=True)`) and the model's
  embedding table resized to match
  (`model.token_rep_layer.resize_token_embeddings(len(tokenizer))`).
  A literal string like `"[unused1]"` reused as a config value is **not**
  protected from WordPiece subword splitting — `class_token_index` /
  `rel_token_index` then never actually match a real token position, and
  `prompts_embedding` silently comes out empty.

- **`max_types` and trigger truncation.** GLiNER randomly samples up to
  `max_types` label columns per training example. With many event types, a
  small `max_types` can truncate away every trigger type from an example's
  sampled label set, zeroing `trigger_class_mask` for that record. This was
  confirmed reachable, not just theoretical: reproduced with real CASIE
  label counts at `max_types=15` (~0.24% of examples per trial), and it used
  to **crash** rather than degrade gracefully -- `create_event_labels`
  floored `max_T`/`max_A` to at least 1 even when the true trigger/argument
  count was 0, while the model's own span selection
  (`select_target_embedding`) produces a genuinely zero-width dimension in
  that case, so `adj_loss` hit a shape mismatch
  (`RuntimeError: size of tensor a (4) must match tensor b (0)`) whenever
  every example in a batch lost all its triggers this way. Fixed in
  `gliner/data_processing/processor.py`'s `create_event_labels` by not
  flooring `max_T`/`max_A` — the record now correctly contributes zero
  event-specific supervision instead of crashing the batch. Regression test:
  `tests/test_event_extraction_integration.py::TestZeroTriggerSurvivalAfterMaxTypesTruncation`.

  This doesn't make the underlying data loss free, though — a record with
  zero surviving triggers still contributes no event supervision on that
  step. Set `max_types` high enough that trigger types aren't at meaningful
  risk of being dropped, particularly for corpora with dozens of event types
  (WikiEvents: 35, ACE2005: 33) or many argument types relative to
  `max_types` (CASIE: 4-5 trigger types against 17+ argument types).

  The same unfloored-vs-floored mismatch pattern exists in the pre-existing,
  non-event `create_relation_labels` (`max_En = max(*batch_ents_cpu, 1)`,
  same file) -- not fixed here, since it predates and is out of scope for
  the event work, but it carries the identical crash risk for plain relation
  extraction whenever an entire batch loses all entities to `max_types`
  truncation.

- **TransE and other norm-based `triples_layer` options have a provable
  zero-gradient bug at transformer scale.** `NormBasedInteraction` (the
  base class for `TransEInteraction`, and also `UM`/`SE`/`TransH`/
  `TransF`/`PairRE`/`TripleRE` in
  `gliner/modeling/multitask/triples_layers.py`) hard-clamps its distance
  score to `clamp_norm=10.0` by default before negating it. At
  `hidden_size=64` with standard transformer-scale representations,
  `||h + r - t||_1` for TransE averages ~88.7 (1000 random samples: mean
  88.66, min 61.90 -- 100% exceeded the clamp). Past the clamp ceiling the
  score is a constant `-10.0` regardless of whether the triple is a true
  positive or a random negative, and `torch.clamp`'s gradient is exactly
  zero past its ceiling, so no training signal reaches the relation-scoring
  path at all -- confirmed empirically: `rel_loss` frozen to 4 significant
  figures across dozens of training steps, and `sigmoid(logits)` at
  gold-positive relation cells frozen at exactly `0.0000`.

  Fix: this feature's example configs and `tests/test_event_training_sanity.py`
  now use `triples_layer=None` (concatenation + dot-product against
  relation-prompt embeddings) instead of `triples_layer="TransE"`. This has
  no clamp and does not exhibit the frozen-gradient symptom. This is a
  pre-existing issue in GLiNER's core KGE scoring code, not introduced by
  the event-mode work -- it equally affects the pre-existing, non-event
  RelEx path whenever `triples_layer` is set to a norm-based interaction.
  It has not been fixed upstream (no change to `triples_layers.py`); the
  mitigation here is a configuration choice (avoid norm-based
  `triples_layer` options for now), not a code fix. `triples_layer="TransE"`
  still appears in a few of this feature's smoke/shape tests
  (`test_wikievents_vertical_slice.py`, `test_event_extraction_integration.py`,
  `test_modeling.py`'s `TestUniEncoderSpanRelexModelEventMode`) where it's
  incidental to what's being tested (forward/backward doesn't crash, shapes
  are correct) -- don't copy those configs as a training starting point.

  `tests/test_event_training_sanity.py` asserts `adj_loss` and `rel_loss`
  each drop by a real margin over 15 steps, not just total loss -- enough
  to catch a dead/disconnected path like TransE's. It is **not** a
  discrimination or output-quality benchmark, and role-classification
  quality at production scale is unverified: this CI-scale harness
  (`hidden_size=64`, one short document, a few hundred steps at most)
  cannot validate learning quality for any classification head in this
  model, including span/entity classification -- a real training run
  (production data volume, a properly tuned learning rate, and the focal
  loss / negative subsampling knobs `gliner/training/trainer.py` exposes)
  is needed to answer that, and hasn't been done as part of this work.

## Inference / decoding

```python
from gliner.decoding.decoder import EventSpanDecoder

collator = EventExtractionSpanDataCollator(
    config, data_processor=processor, prepare_labels=False,
    return_id_to_classes=True, return_rel_id_to_classes=True,
)
batch = collator(records_batch)
with torch.no_grad():
    output = model(**batch)

decoder = EventSpanDecoder(config)
spans, events = decoder.decode(
    tokens=[r["tokenized_text"] for r in records_batch],
    id_to_classes=batch["id_to_classes"],
    model_output=output.logits,
    rel_id_to_classes=batch["rel_id_to_classes"],
    trigger_spans=output.trigger_spans,
    arg_spans=output.arg_spans,
    rel_idx=output.rel_idx,
    rel_logits=output.rel_logits,
    rel_mask=output.rel_mask,
    threshold=0.5, relation_threshold=0.5,
)
# events[i] is a list of (trigger_idx, role_label, arg_idx, score) tuples,
# indices into spans[i].
```

`EventSpanDecoder.decode` calls `BaseSpanDecoder.decode` directly (not
`SpanRelexDecoder.decode`, which assumes one merged entity-span list) and
resolves roles through two independent span lookups —
`trigger_spans` / `arg_spans`, both new optional fields on
`GLiNERRelexOutput` populated only in eval mode with `event_mode=True`.

**Uncalibrated-threshold candidate blowup**: at inference, `threshold=0.5`
on an untrained (or lightly trained) model marks a large fraction of all
spans as candidates, and `build_trigger_argument_pairs` pads every
document in a batch to the largest document's pair count. This is a
property of threshold-based candidate architectures generally (also true
of the pre-existing non-event relation-extraction path), not
event-specific — a trained model produces far sparser positives. It mainly
matters for batching heterogeneous document lengths together at inference
time; training is unaffected since it's bounded by the gold adjacency
matrix plus negative sampling, not raw thresholding.

## Data converters

`data/convert_*.py` in the GLiNER repo root ports 9 event datasets from
GLiNER2's `tools/data/` to this schema. Where the source ships usable
offsets, converters use them directly rather than substring-searching for
surface text (substring search silently resolves a repeated surface string
to its first occurrence only — see `data/_charspan.py`).

| Dataset | Triggers | Arguments | Offset strategy |
|---|---|---|---|
| WikiEvents | yes | yes (typed) | native word-token offsets |
| RAMS | yes | yes (role = type, no independent typing) | native word-token offsets |
| MAVEN | yes | none (trigger-only) | native word-token offsets |
| LEVEN | yes | none (trigger-only, Chinese) | native word-token offsets (reuses MAVEN's `convert_row`, identical schema) |
| CASIE | yes | yes (typed, independent of role) | native char offsets, mapped to word tokens via `data/_charspan.py` |
| CMNEE | yes | yes (role = type) | native char offsets == token indices (Chinese, one char per token) |
| DocEE | **synthetic** (see below) | yes (typed) | native char offsets, mapped to word tokens |
| events_biotech | **synthetic** (see below) | none | n/a (pure classification source) |
| ACE2005 | yes | yes (typed when resolvable, else role = type) | code port only — **not run against real data**, see below |

**DocEE and events_biotech are architectural mismatches force-fit into this
schema** — DocEE's event type is a document-level label with no annotated
trigger span at all (a "strict one-event-per-document" corpus), and
events_biotech is pure multi-label text classification with no span-level
signal whatsoever. Both converters prepend a synthetic trigger token
(`f"[{event_type}]"`) to the document rather than dropping the dataset —
DocEE gets one synthetic trigger per document (with real, offset-based
argument spans), events_biotech gets one synthetic trigger per label with
no arguments at all (`relations` is always empty). This is a genuinely
degenerate training signal: the model only learns to recognize a literal
bracketed placeholder it was told is a trigger, not a real-world event
mention. Included because it was explicitly requested, not because either
is expected to generalize to real trigger detection.

**ACE2005 is LDC-licensed (LDC2006T06)** — no locally-licensed copy is
available in this environment, so `data/convert_ace2005.py` is a code port
of GLiNER2's converter logic, exercised only against a small hand-built
synthetic APF/SGM fixture (`tests/test_data_converters.py::TestConvertAce2005`),
not real corpus data. It also diverges from the offset-based converters
above: ACE's `charseq` offsets are relative to the original SGM file, but
both this converter and GLiNER2's strip SGML markup first, which
invalidates those offsets — so, like GLiNER2's converter, it falls back to
a substring search on the stripped, word-tokenized text (the same
first-occurrence-only limitation that native offsets are used elsewhere
specifically to avoid).

## Testing

- `tests/test_relations_layers.py` — `BipartiteRelationsRepLayer` and its
  dot/MLP/bilinear decoders in isolation.
- `tests/test_modeling.py::TestBuildTriggerArgumentPairs`,
  `TestUniEncoderSpanRelexModelEventMode` — pair-building and the full
  model forward/backward pass in `event_mode`.
- `tests/test_data_processing.py::TestEventExtractionSpanProcessor` —
  proves the label-alignment invariant with exact-value assertions at known
  matrix positions, not just shape checks.
- `tests/test_data_converters.py` — each converter's conversion logic
  against small synthetic inputs shaped like the real raw source schema.
- `tests/test_wikievents_vertical_slice.py` — real WikiEvents documents
  (`tests/fixtures/wikievents_sample.jsonl`) through the full
  processor → collator → model → decoder pipeline.
- `tests/test_event_training_sanity.py` — a genuine overfit run on real
  WikiEvents text (not just "doesn't crash"): asserts total loss, `adj_loss`,
  and `rel_loss` each decrease by more than noise over 15 training steps.
  Uses `triples_layer=None` -- see "TransE ... saturate" above for why.
- `tests/test_trigger_types.py` — `trigger_types` derivation, the
  `apply_derived_trigger_types` wiring against both a dict (train.py) and a
  namespace (custom_train.py) config, the `data/_trigger_types.py` re-export
  shim, and a `train.main()` integration guard that fails if the
  auto-derivation wiring is removed from the training script.
