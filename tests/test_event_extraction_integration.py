"""End-to-end integration test for event extraction: real tokenizer ->
EventExtractionSpanProcessor -> EventExtractionSpanDataCollator ->
UniEncoderSpanRelexModel(event_mode=True) forward/backward.

This is the vertical slice the model-level (test_modeling.py) and
processor-level (test_data_processing.py) unit tests deliberately don't
cover: those exercise the bipartite architecture and the label-alignment
convention in isolation, on hand-built tensors. This test proves the two
actually agree once real tokenization sits between them -- a real trigger
span and a real argument span, tokenized and re-collated, must still land
on spans the model can select and score correctly.

class_token_index/rel_token_index must correspond to <<ENT>>/<<REL>> being
real, atomic special tokens -- registering a *string* that happens to match
an existing vocab entry (e.g. "[unused1]") is not enough, since WordPiece
still splits it into pieces on encode() unless it's added via
tokenizer.add_tokens(special_tokens=True). So this mirrors
gliner.model.BaseGLiNER._resize_token_embeddings exactly: add the tokens,
then grow the model's embedding table to match (both discovered the hard
way -- an earlier version of this test used [unused1]/[unused2] directly and
silently exercised a zero-column prompts_embedding, papered over whenever
`labels` happened to be present to force target_C).

augment_data_prob is set to 0 -- its default (0.5) randomly drops entity
types per example, which is real training-time behavior but only adds
nondeterminism here, where the point is to check specific known spans.
"""

import torch
import pytest
from transformers import AutoTokenizer

from gliner.config import UniEncoderSpanRelexConfig
from gliner.modeling.base import UniEncoderSpanRelexModel
from gliner.data_processing import EventExtractionSpanProcessor, EventExtractionSpanDataCollator


@pytest.fixture
def tokenizer():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok.add_tokens(["<<ENT>>", "<<REL>>"], special_tokens=True)
    return tok


@pytest.fixture
def config(tokenizer):
    return UniEncoderSpanRelexConfig(
        model_name="bert-base-uncased",
        hidden_size=64,
        dropout=0.1,
        max_width=5,
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
    )


def _build_model(config, tokenizer):
    """Construct the model and grow its embedding table to match the
    tokenizer's vocab (2 tokens larger than the base checkpoint, from
    <<ENT>>/<<REL>>) -- mirrors BaseGLiNER._resize_token_embeddings."""
    model = UniEncoderSpanRelexModel(config, from_pretrained=False)
    model.token_rep_layer.resize_token_embeddings(len(tokenizer))
    return model


@pytest.fixture
def processor(config, tokenizer):
    return EventExtractionSpanProcessor(config, tokenizer, words_splitter=None, trigger_types={"Attack"})


@pytest.fixture
def raw_example():
    return {
        "tokenized_text": ["John", "attacked", "Paris", "yesterday"],
        "ner": [(2, 2, "Location"), (1, 1, "Attack"), (0, 0, "Person")],
        "relations": [(1, 0, "Place"), (1, 2, "Attacker")],
    }


class TestEventExtractionEndToEnd:
    def test_collator_produces_model_ready_batch(self, config, processor, raw_example):
        collator = EventExtractionSpanDataCollator(config, data_processor=processor, prepare_labels=True)

        batch = collator([raw_example])

        for key in ("input_ids", "attention_mask", "span_idx", "span_mask", "labels",
                    "adj_matrix", "rel_matrix", "trigger_class_mask"):
            assert key in batch, f"missing '{key}' in collated batch"

        assert batch["input_ids"].shape[0] == 1
        assert batch["trigger_class_mask"].dtype == torch.bool
        # 3 entities were labeled (Location, Attack, Person) -> at least 3 class columns
        assert batch["trigger_class_mask"].shape[-1] >= 3
        # Exactly one of those columns (Attack) is a trigger column
        assert batch["trigger_class_mask"][0].sum().item() == 1

    def test_real_tokenized_spans_still_produce_correct_bipartite_adjacency(self, config, processor, raw_example):
        """The known-link property proven on hand-built tensors in
        test_data_processing.py must survive real tokenization: the trigger
        (Attack/attacked) must end up adjacent to both arguments it has
        roles with, in a (1, num_triggers, num_args) matrix."""
        collator = EventExtractionSpanDataCollator(config, data_processor=processor, prepare_labels=True)
        batch = collator([raw_example])

        adj = batch["adj_matrix"]
        assert adj.shape[0] == 1
        assert adj.shape[1] >= 1  # at least the one trigger
        assert adj.shape[2] >= 2  # at least the two arguments
        # every trigger-argument pair should have at least one linked entry
        assert adj.sum().item() >= 2.0

    def test_forward_and_backward_through_full_pipeline(self, config, tokenizer, processor, raw_example):
        collator = EventExtractionSpanDataCollator(config, data_processor=processor, prepare_labels=True)
        batch = collator([raw_example])

        model = _build_model(config, tokenizer)
        output = model(**batch)

        assert output.loss is not None
        assert torch.isfinite(output.loss)
        output.loss.backward()

    def test_inference_mode_returns_trigger_and_arg_spans_not_entity_spans(self, config, tokenizer, processor, raw_example):
        """Without gold labels, decode-relevant outputs must come back as the
        two separate trigger_spans/arg_spans lists (event mode), not a single
        merged entity_spans list (homogeneous mode)."""
        collator = EventExtractionSpanDataCollator(config, data_processor=processor, prepare_labels=False)
        batch = collator([raw_example])
        batch.pop("labels", None)
        batch.pop("adj_matrix", None)
        batch.pop("rel_matrix", None)

        model = _build_model(config, tokenizer)
        with torch.no_grad():
            output = model(**batch)

        assert output.loss is None
        assert output.entity_spans is None
        assert output.trigger_spans is not None
        assert output.arg_spans is not None


class TestZeroTriggerSurvivalAfterMaxTypesTruncation:
    """Regression test for a real crash found via deeper training validation.

    GLiNER randomly samples up to `max_types` label columns per training
    example (see batch_generate_class_mappings in
    gliner/data_processing/processor.py). With many event types, an unlucky
    shuffle can exclude every trigger type from the sampled vocabulary for a
    whole batch -- reachable in practice (empirically ~0.24% per-example
    with CASIE's real label counts at max_types=15, a small but realistic
    setting) -- leaving num_triggers=0 for every example in the batch.

    Root cause: create_event_labels floored max_T/max_A to >= 1 even when
    the true count is 0, but the model's own trigger/arg selection
    (select_target_embedding in gliner/modeling/base.py) produces a
    genuinely zero-width dimension in that case -- a shape mismatch that
    crashed adj_loss (RuntimeError: size of tensor a (4) must match tensor
    b (0)). Fixed by not flooring max_T/max_A, so the label and model
    prediction shapes agree; the record then gracefully contributes no
    event-specific supervision instead of crashing the batch.
    """

    def test_forward_backward_survives_when_every_example_loses_all_triggers(self, config, tokenizer, processor):
        # entity_types deliberately excludes "Attack" (the only trigger type)
        # -- exactly what an unlucky max_types shuffle could produce.
        raw_example = {
            "tokenized_text": ["John", "attacked", "Paris", "yesterday"],
            "ner": [(2, 2, "Location"), (1, 1, "Attack"), (0, 0, "Person")],
            "relations": [(1, 0, "Place"), (1, 2, "Attacker")],
        }
        collator = EventExtractionSpanDataCollator(config, data_processor=processor, prepare_labels=True)
        batch = collator(
            [raw_example],
            entity_types=[["Location", "Person"]],
            relation_types=[["Place", "Attacker"]],
        )
        assert not batch["trigger_class_mask"].any(), "test setup must actually exclude the trigger type"

        model = _build_model(config, tokenizer)
        output = model(**batch)

        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
