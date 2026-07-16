"""Tests for EventSpanDecoder (bipartite trigger/argument decode path).

rel_idx/rel_logits/rel_mask/trigger_spans/arg_spans are always passed
explicitly to decode() -- SpanRelexDecoder.decode() (and this class) does
not pull them from model_output automatically, so a test that only mutates
model_output attributes without passing the matching kwarg would silently
decode zero relations/events and pass vacuously. Assertions here check
specific known decoded content, not just tuple shape, for the same reason
the processor-level alignment tests do (test_data_processing.py).
"""

import torch
import pytest

from gliner.decoding.decoder import Span, EventSpanDecoder


@pytest.fixture
def config():
    from unittest.mock import Mock
    cfg = Mock()
    cfg.max_width = 3
    return cfg


@pytest.fixture
def scenario():
    """'The quick brown fox jumps': trigger span at (0,0)='The' (class 1,
    'Attack'), argument span at (2,2)='brown' (class 2, 'Location'), with one
    role edge (trigger 0 -> arg 0, role 'Place')."""
    tokens = [["The", "quick", "brown", "fox", "jumps"]]
    seq_length, max_width, num_classes = 5, 3, 2

    logits = torch.zeros(1, seq_length, max_width, num_classes)
    logits[0, 0, 0, 0] = 5.0  # (start=0, width=0) -> span (0,0), class idx 0 -> id 1 "Attack"
    logits[0, 2, 0, 1] = 4.0  # (start=2, width=0) -> span (2,2), class idx 1 -> id 2 "Location"

    id_to_classes = {1: "Attack", 2: "Location"}
    rel_id_to_classes = {1: "Place"}

    trigger_spans = torch.tensor([[[0, 0]]])  # (B=1, T=1, 2) -> matches decoded (0,0)
    arg_spans = torch.tensor([[[2, 2]]])      # (B=1, A=1, 2) -> matches decoded (2,2)

    rel_idx = torch.tensor([[[0, 0]]])            # (B=1, num_pairs=1, 2): trigger 0 -> arg 0
    rel_logits = torch.zeros(1, 1, 1)
    rel_logits[0, 0, 0] = 5.0                      # high confidence "Place"
    rel_mask = torch.tensor([[True]])

    return {
        "tokens": tokens,
        "model_output": logits,
        "id_to_classes": id_to_classes,
        "rel_id_to_classes": rel_id_to_classes,
        "trigger_spans": trigger_spans,
        "arg_spans": arg_spans,
        "rel_idx": rel_idx,
        "rel_logits": rel_logits,
        "rel_mask": rel_mask,
    }


class TestEventSpanDecoder:
    def test_output_structure(self, config, scenario):
        decoder = EventSpanDecoder(config)

        spans, events = decoder.decode(
            tokens=scenario["tokens"],
            id_to_classes=scenario["id_to_classes"],
            model_output=scenario["model_output"],
            rel_id_to_classes=scenario["rel_id_to_classes"],
            trigger_spans=scenario["trigger_spans"],
            arg_spans=scenario["arg_spans"],
            rel_idx=scenario["rel_idx"],
            rel_logits=scenario["rel_logits"],
            rel_mask=scenario["rel_mask"],
            threshold=0.5,
        )

        assert isinstance(spans, list) and isinstance(events, list)
        assert len(spans) == 1 and len(events) == 1

    def test_decodes_both_trigger_and_argument_as_spans(self, config, scenario):
        """logits covers all class columns regardless of trigger/argument
        membership, so both must appear in the flat spans list, typed
        correctly (the same property represent_spans_bipartite relies on:
        scores is never split, only *selection* is)."""
        decoder = EventSpanDecoder(config)

        spans, _ = decoder.decode(
            tokens=scenario["tokens"],
            id_to_classes=scenario["id_to_classes"],
            model_output=scenario["model_output"],
            threshold=0.5,
        )

        by_boundary = {(s.start, s.end): s.entity_type for s in spans[0]}
        assert by_boundary.get((0, 0)) == "Attack"
        assert by_boundary.get((2, 2)) == "Location"

    def test_known_role_decodes_to_correct_trigger_and_argument_spans(self, config, scenario):
        """The one known (trigger, role, argument) edge must resolve to the
        exact decoded Span indices for (0,0)/Attack and (2,2)/Location, not
        just *some* tuple of the right shape."""
        decoder = EventSpanDecoder(config)

        spans, events = decoder.decode(
            tokens=scenario["tokens"],
            id_to_classes=scenario["id_to_classes"],
            model_output=scenario["model_output"],
            rel_id_to_classes=scenario["rel_id_to_classes"],
            trigger_spans=scenario["trigger_spans"],
            arg_spans=scenario["arg_spans"],
            rel_idx=scenario["rel_idx"],
            rel_logits=scenario["rel_logits"],
            rel_mask=scenario["rel_mask"],
            threshold=0.5,
        )

        assert len(events[0]) == 1
        trigger_idx, role_label, arg_idx, score = events[0][0]
        assert role_label == "Place"
        assert isinstance(score, float) and score > 0.5

        decoded_trigger = spans[0][trigger_idx]
        decoded_arg = spans[0][arg_idx]
        assert (decoded_trigger.start, decoded_trigger.end) == (0, 0)
        assert decoded_trigger.entity_type == "Attack"
        assert (decoded_arg.start, decoded_arg.end) == (2, 2)
        assert decoded_arg.entity_type == "Location"

    def test_respects_role_mask(self, config, scenario):
        scenario["rel_mask"] = torch.tensor([[False]])
        decoder = EventSpanDecoder(config)

        _, events = decoder.decode(
            tokens=scenario["tokens"],
            id_to_classes=scenario["id_to_classes"],
            model_output=scenario["model_output"],
            rel_id_to_classes=scenario["rel_id_to_classes"],
            trigger_spans=scenario["trigger_spans"],
            arg_spans=scenario["arg_spans"],
            rel_idx=scenario["rel_idx"],
            rel_logits=scenario["rel_logits"],
            rel_mask=scenario["rel_mask"],
            threshold=0.5,
        )

        assert events[0] == []

    def test_filters_negative_indices(self, config, scenario):
        scenario["rel_idx"] = torch.tensor([[[-1, 0]]])
        decoder = EventSpanDecoder(config)

        _, events = decoder.decode(
            tokens=scenario["tokens"],
            id_to_classes=scenario["id_to_classes"],
            model_output=scenario["model_output"],
            rel_id_to_classes=scenario["rel_id_to_classes"],
            trigger_spans=scenario["trigger_spans"],
            arg_spans=scenario["arg_spans"],
            rel_idx=scenario["rel_idx"],
            rel_logits=scenario["rel_logits"],
            rel_mask=scenario["rel_mask"],
            threshold=0.5,
        )

        assert events[0] == []

    def test_no_events_when_not_requested(self, config, scenario):
        decoder = EventSpanDecoder(config)

        spans, events = decoder.decode(
            tokens=scenario["tokens"],
            id_to_classes=scenario["id_to_classes"],
            model_output=scenario["model_output"],
            threshold=0.5,
        )

        assert len(spans[0]) == 2  # spans still decode independently of events
        assert events[0] == []

    def test_below_threshold_role_is_dropped(self, config, scenario):
        scenario["rel_logits"][0, 0, 0] = -5.0  # sigmoid(-5) << 0.5
        decoder = EventSpanDecoder(config)

        _, events = decoder.decode(
            tokens=scenario["tokens"],
            id_to_classes=scenario["id_to_classes"],
            model_output=scenario["model_output"],
            rel_id_to_classes=scenario["rel_id_to_classes"],
            trigger_spans=scenario["trigger_spans"],
            arg_spans=scenario["arg_spans"],
            rel_idx=scenario["rel_idx"],
            rel_logits=scenario["rel_logits"],
            rel_mask=scenario["rel_mask"],
            threshold=0.5,
        )

        assert events[0] == []

    def test_trigger_span_not_in_decoded_spans_is_dropped(self, config, scenario):
        """If a trigger boundary from represent_spans_bipartite doesn't match
        any decoded span (e.g. it scored above the bipartite-selection
        threshold but below the decode threshold), the role referencing it
        must be dropped, not crash or fabricate an index."""
        scenario["trigger_spans"] = torch.tensor([[[1, 1]]])  # "quick" -- never decoded as a span
        decoder = EventSpanDecoder(config)

        _, events = decoder.decode(
            tokens=scenario["tokens"],
            id_to_classes=scenario["id_to_classes"],
            model_output=scenario["model_output"],
            rel_id_to_classes=scenario["rel_id_to_classes"],
            trigger_spans=scenario["trigger_spans"],
            arg_spans=scenario["arg_spans"],
            rel_idx=scenario["rel_idx"],
            rel_logits=scenario["rel_logits"],
            rel_mask=scenario["rel_mask"],
            threshold=0.5,
        )

        assert events[0] == []
