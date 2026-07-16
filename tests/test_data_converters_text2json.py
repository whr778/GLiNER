"""Unit tests for data/convert_text2json.py.

Hermetic -- small synthetic (text, extracted) pairs shaped like the real raw
source schema. Also run against 30 real rows downloaded from HuggingFace
during development (24/30 converted).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_text2json as t2j  # noqa: E402


class TestConvertText2json:
    def test_entity_list_shape(self):
        row = {"text": "Sarah Cooley is a marine chemist.",
               "extracted": '{"entities": [{"entity": "Sarah Cooley", "type": "Person", "description": "marine chemist"}]}'}
        rec = t2j.convert_row(row)
        assert rec is not None
        s, e, label = rec["ner"][0]
        assert " ".join(rec["tokenized_text"][s:e + 1]) == "Sarah Cooley"
        assert label == "Person"
        assert rec["relations"] == []

    def test_flat_key_value_shape(self):
        row = {"text": "Tournament ROL-2024 was won by Sofia Petrova.",
               "extracted": '{"tournament_code": "ROL-2024", "winner": "Sofia Petrova"}'}
        rec = t2j.convert_row(row)
        assert rec is not None
        by_label = {label: " ".join(rec["tokenized_text"][s:e + 1]) for s, e, label in rec["ner"]}
        assert by_label == {"tournament_code": "ROL-2024", "winner": "Sofia Petrova"}

    def test_list_of_scalars_becomes_one_bucket(self):
        row = {"text": "Winners: Alice, Bob.", "extracted": '{"winner": ["Alice", "Bob"]}'}
        rec = t2j.convert_row(row)
        labels = [label for _, _, label in rec["ner"]]
        assert labels == ["winner", "winner"]

    def test_drops_value_not_verbatim_in_text(self):
        row = {"text": "Nothing matches.", "extracted": '{"winner": "Nobody Here"}'}
        assert t2j.convert_row(row) is None

    def test_malformed_json_returns_none(self):
        assert t2j.convert_row({"text": "x", "extracted": "{not json"}) is None
