"""Unit tests for data/convert_finer_ord.py.

Hermetic -- small synthetic BIO-labeled token sequences. Also run against
real sentences streamed from HuggingFace during development.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_finer_ord as finer  # noqa: E402


class TestConvertFinerOrd:
    def test_decodes_multi_token_bio_span(self):
        tokens = ["John", "Smith", "works", "at", "KPMG", "."]
        labels = [1, 2, 0, 0, 5, 0]  # PER_B, PER_I, O, O, ORG_B, O
        rec = finer.sentence_to_record(tokens, labels)
        assert rec is not None
        assert rec["ner"] == [[0, 1, "PER"], [4, 4, "ORG"]]
        assert rec["relations"] == []
        assert rec["tokenized_text"] == tokens

    def test_adjacent_same_type_begin_starts_a_new_span(self):
        # Two consecutive PER_B tags are two separate single-token entities,
        # not one merged span.
        tokens = ["Alice", "Bob"]
        labels = [1, 1]
        rec = finer.sentence_to_record(tokens, labels)
        assert rec["ner"] == [[0, 0, "PER"], [1, 1, "PER"]]

    def test_returns_none_for_all_o_sentence(self):
        assert finer.sentence_to_record(["no", "entities", "here"], [0, 0, 0]) is None

    def test_span_open_at_end_of_sentence_closes_correctly(self):
        tokens = ["visit", "New", "York"]
        labels = [0, 3, 4]  # O, LOC_B, LOC_I -- entity runs to the end
        rec = finer.sentence_to_record(tokens, labels)
        assert rec["ner"] == [[1, 2, "LOC"]]
