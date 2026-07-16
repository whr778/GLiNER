"""Unit tests for data/convert_paraloq_json.py.

Hermetic -- small synthetic (text, item) pairs shaped like the real raw
source schema. Also run against 30 real rows streamed from HuggingFace
during development (30/30 converted).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_paraloq_json as paraloq  # noqa: E402


class TestConvertParaloqJson:
    def test_dict_keys_become_labels_for_nested_scalars(self):
        row = {"text": "Aiden Smith was born 1990-03-08.",
               "item": {"name": "Aiden Smith", "dateOfBirth": "1990-03-08"}}
        rec = paraloq.convert_row(row)
        assert rec is not None
        by_label = {label: " ".join(rec["tokenized_text"][s:e + 1]) for s, e, label in rec["ner"]}
        assert by_label == {"name": "Aiden Smith", "dateOfBirth": "1990-03-08"}
        assert rec["relations"] == []

    def test_list_items_inherit_parent_label(self):
        row = {"text": "Symptoms: fever, cough.",
               "item": {"symptoms": ["fever", "cough"]}}
        rec = paraloq.convert_row(row)
        assert rec is not None
        labels = [label for _, _, label in rec["ner"]]
        assert labels == ["symptoms", "symptoms"]

    def test_drops_value_not_verbatim_in_text(self):
        row = {"text": "Nothing matches here.", "item": {"name": "Someone Else"}}
        assert paraloq.convert_row(row) is None

    def test_string_item_is_json_parsed(self):
        row = {"text": "Aiden Smith.", "item": '{"name": "Aiden Smith"}'}
        rec = paraloq.convert_row(row)
        assert rec is not None
        assert rec["ner"][0][2] == "name"

    def test_drops_overlong_surface(self):
        long_value = " ".join(f"word{i}" for i in range(60))
        row = {"text": f"Text with {long_value} inline.", "item": {"field": long_value}}
        assert paraloq.convert_row(row) is None
