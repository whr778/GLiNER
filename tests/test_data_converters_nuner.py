"""Unit tests for data/convert_nuner.py.

Hermetic -- small synthetic input shaped like the real raw source schema.
Also run against 30 real rows streamed from HuggingFace during development
(measured surface-find rate ~85%, so the verbatim check isn't silently
dropping most of the signal).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_nuner as nuner  # noqa: E402


class TestParseItems:
    def test_parses_surface_type_description_triples(self):
        raw = "['Paris <> location <> capital of France']"
        assert nuner.parse_items(raw) == [["Paris", "location", "capital of France"]]

    def test_parses_surface_type_only(self):
        raw = "['Paris <> location']"
        assert nuner.parse_items(raw) == [["Paris", "location"]]


class TestConvertNuner:
    def test_locates_surface_and_drops_description(self):
        row = {"input": "Paris is the capital of France.",
               "output": "['Paris <> location <> capital of France']"}
        rec = nuner.convert_row(row)
        assert rec is not None
        s, e, label = rec["ner"][0]
        assert " ".join(rec["tokenized_text"][s:e + 1]) == "Paris"
        assert label == "location"
        assert rec["relations"] == []

    def test_drops_surface_not_verbatim_in_text(self):
        row = {"input": "Paris is nice.", "output": "['London <> location']"}
        assert nuner.convert_row(row) is None

    def test_returns_none_for_missing_input(self):
        assert nuner.convert_row({"input": None, "output": "['x <> y']"}) is None
