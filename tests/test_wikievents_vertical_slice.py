"""Vertical slice: real WikiEvents data -> EventExtractionSpanProcessor ->
EventExtractionSpanDataCollator -> UniEncoderSpanRelexModel(event_mode=True)
-> EventSpanDecoder, end to end.

tests/fixtures/wikievents_sample.jsonl holds 5 real documents produced by
data/convert_wikievents.py (from the public WikiEvents dev split), each
with at least one (trigger, role, argument) edge -- not synthetic/hand-built
data, so this is the point where schema, pipeline, and model either agree on
real text or don't. Unlike test_event_extraction_integration.py (a single
4-token hand-built sentence), this exercises real document lengths (40 to
2000+ tokens, so max_len truncation is genuinely exercised), real label
diversity (35 WikiEvents/KAIROS event types, 14 entity types), and multiple
documents batched together.
"""

import json
import sys
from pathlib import Path

import torch
import pytest
from transformers import AutoTokenizer

from gliner.config import UniEncoderSpanRelexConfig
from gliner.modeling.base import UniEncoderSpanRelexModel
from gliner.data_processing import EventExtractionSpanProcessor, EventExtractionSpanDataCollator
from gliner.decoding.decoder import EventSpanDecoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from _trigger_types import derive_trigger_types  # noqa: E402

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "wikievents_sample.jsonl"


def _load_fixture():
    records = []
    with FIXTURE_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            records.append({
                "tokenized_text": rec["tokenized_text"],
                "ner": [tuple(e) for e in rec["ner"]],
                "relations": [tuple(r) for r in rec["relations"]],
            })
    return records


# Trigger types are derived from relation-head labels (data/_trigger_types.py),
# not from a label-shape heuristic like "contains a dot" -- that would happen
# to work for WikiEvents (dotted event types) but silently misclassify
# datasets whose trigger types are bare words (e.g. CMNEE's "Manoeuvre").
_trigger_types = derive_trigger_types


@pytest.fixture(scope="module")
def records():
    recs = _load_fixture()
    assert len(recs) == 5, "fixture should hold the 5 sample WikiEvents documents"
    assert any(rec["relations"] for rec in recs), "fixture must include at least one real event role"
    return recs


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok.add_tokens(["<<ENT>>", "<<REL>>"], special_tokens=True)
    return tok


@pytest.fixture(scope="module")
def config(tokenizer, records):
    return UniEncoderSpanRelexConfig(
        model_name="bert-base-uncased",
        hidden_size=64,
        dropout=0.1,
        max_width=8,
        max_len=384,
        span_mode="markerV0",
        class_token_index=tokenizer.convert_tokens_to_ids("<<ENT>>"),
        rel_token_index=tokenizer.convert_tokens_to_ids("<<REL>>"),
        vocab_size=len(tokenizer),
        has_rnn=False,
        post_fusion_schema="",
        embed_ent_token=True,
        embed_rel_token=True,
        relations_layer="dot",
        triples_layer="TransE",
        event_mode=True,
        augment_data_prob=0.0,
        max_types=60,  # >= 35 trigger + 14 argument types actually present
    )


@pytest.fixture(scope="module")
def processor(config, tokenizer, records):
    return EventExtractionSpanProcessor(
        config, tokenizer, words_splitter=None, trigger_types=_trigger_types(records)
    )


def _build_model(config, tokenizer):
    model = UniEncoderSpanRelexModel(config, from_pretrained=False)
    model.token_rep_layer.resize_token_embeddings(len(tokenizer))
    return model


class TestWikiEventsVerticalSlice:
    def test_converted_fixture_has_correct_span_offsets(self, records):
        """Sanity-check the fixture itself: every (start, end) NER span must
        index within its own document's token list."""
        for rec in records:
            num_tokens = len(rec["tokenized_text"])
            for start, end, _ in rec["ner"]:
                assert 0 <= start <= end < num_tokens

    def test_collator_handles_a_batch_of_real_documents(self, config, processor, records):
        collator = EventExtractionSpanDataCollator(config, data_processor=processor, prepare_labels=True)

        batch = collator(records)

        assert batch["input_ids"].shape[0] == len(records)
        assert batch["adj_matrix"].shape[0] == len(records)
        assert batch["rel_matrix"].shape[0] == len(records)
        assert batch["trigger_class_mask"].shape[0] == len(records)
        # at least one real trigger class column, given the fixture has real events
        assert batch["trigger_class_mask"].sum().item() >= 1

    def test_forward_and_backward_on_real_batch(self, config, tokenizer, processor, records):
        collator = EventExtractionSpanDataCollator(config, data_processor=processor, prepare_labels=True)
        batch = collator(records)

        model = _build_model(config, tokenizer)
        output = model(**batch)

        assert output.loss is not None
        assert torch.isfinite(output.loss)
        output.loss.backward()

        # Gradients must actually reach the span representation layer feeding
        # the bipartite path, not just the classification head -- otherwise a
        # finite loss could mean the event-specific path was silently a
        # no-op. ("dot" adjacency and TransE have no learnable parameters of
        # their own, so this is the right place to check gradient flow.)
        span_params = list(model.span_rep_layer.parameters())
        assert span_params
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in span_params)

    def test_inference_and_decode_on_real_documents(self, config, tokenizer, processor, records):
        """Full loop: real text -> model (untrained, but must not crash) ->
        EventSpanDecoder -> a shaped, well-formed (if not necessarily
        accurate) set of predicted spans/events per document.

        Scoped to the single smallest fixture document rather than the full
        batch of 5: an untrained model's scores are uncalibrated, so at
        threshold=0.5 a large fraction of all (span, class) combinations
        are "positive", and candidate-pair building pads every document in
        a batch to the largest document's pair count. On the full 5-doc
        batch (one of which is 2127 tokens) this produces tens of millions
        of candidate pairs and the Python-level decode loop does not finish
        in reasonable time -- confirmed by direct profiling, not a decode
        bug, and not something a trained model (far sparser positives) or
        the training path (bounded by gold adj_matrix + negative sampling,
        not raw thresholding) would hit. One real document is still a
        genuine, non-synthetic check of the full pipeline.
        """
        small_record = min(records, key=lambda r: len(r["tokenized_text"]))
        collator = EventExtractionSpanDataCollator(
            config, data_processor=processor, prepare_labels=False,
            return_id_to_classes=True, return_rel_id_to_classes=True,
        )
        batch = collator([small_record])
        id_to_classes = batch["id_to_classes"]
        rel_id_to_classes = batch["rel_id_to_classes"]

        model = _build_model(config, tokenizer)
        with torch.no_grad():
            output = model(**batch)

        decoder = EventSpanDecoder(config)
        tokens = [small_record["tokenized_text"]]

        spans, events = decoder.decode(
            tokens=tokens,
            id_to_classes=id_to_classes,
            model_output=output.logits,
            rel_id_to_classes=rel_id_to_classes,
            trigger_spans=output.trigger_spans,
            arg_spans=output.arg_spans,
            rel_idx=output.rel_idx,
            rel_logits=output.rel_logits,
            rel_mask=output.rel_mask,
            threshold=0.5,
            relation_threshold=0.5,
        )

        assert len(spans) == 1
        assert len(events) == 1
        for doc_events in events:
            for trigger_idx, role_label, arg_idx, score in doc_events:
                assert isinstance(role_label, str)
                assert 0.0 <= score <= 1.0
