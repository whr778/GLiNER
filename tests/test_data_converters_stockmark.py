"""Unit tests for data/convert_stockmark_ner.py.

Hermetic -- small synthetic input shaped like the real raw source schema.
Also run against 20 real rows streamed from HuggingFace during development.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_stockmark_ner as stockmark  # noqa: E402


class TestConvertStockmarkNer:
    def test_char_span_is_directly_a_token_index_span(self):
        row = {"text": "SPRiNGSと最も仲の良い", "entities": [{"name": "SPRiNGS", "span": [0, 7], "type": "その他の組織名"}]}
        rec = stockmark.convert_row(row)
        assert rec is not None
        s, e, label = rec["ner"][0]
        assert "".join(rec["tokenized_text"][s:e + 1]) == "SPRiNGS"
        assert label == "その他の組織名"
        assert rec["relations"] == []

    def test_drops_out_of_bounds_span(self):
        row = {"text": "short", "entities": [{"name": "x", "span": [0, 99], "type": "PER"}]}
        assert stockmark.convert_row(row) is None

    def test_returns_none_without_entities(self):
        assert stockmark.convert_row({"text": "no entities", "entities": []}) is None
