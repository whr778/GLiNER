"""Unit tests for data/convert_biomed_ner.py.

Hermetic -- small synthetic input shaped like the real raw source schema.
Also run against 30 real rows streamed from HuggingFace during development.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_biomed_ner  # noqa: E402


class TestConvertBiomedNer:
    def test_maps_char_offsets_to_token_spans(self):
        row = {
            "text": "Weed seed inactivation in soil mesocosms.",
            "entities": [{"start": 0, "end": 4, "class": "ORGANISM"}],
        }
        rec = convert_biomed_ner.convert_row(row)
        assert rec is not None
        s, e, label = rec["ner"][0]
        assert " ".join(rec["tokenized_text"][s:e + 1]) == "Weed"
        assert label == "ORGANISM"
        assert rec["relations"] == []

    def test_strips_class_whitespace_and_skips_unlabelled(self):
        row = {
            "text": "A chemical here.",
            "entities": [
                {"start": 2, "end": 10, "class": "CHEMICALS "},
                {"start": 0, "end": 1, "class": "Unlabelled"},
            ],
        }
        rec = convert_biomed_ner.convert_row(row)
        assert rec is not None
        assert len(rec["ner"]) == 1
        assert rec["ner"][0][2] == "CHEMICALS"

    def test_returns_none_without_entities(self):
        assert convert_biomed_ner.convert_row({"text": "no entities here", "entities": []}) is None
