"""Unit tests for data/convert_pubmed_abstracts_ner.py.

Hermetic -- small synthetic input shaped like the real raw source schema.
Also run against 20 real rows downloaded from HuggingFace during development.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_pubmed_abstracts_ner as pubmed  # noqa: E402


class TestSplitLabel:
    def test_splits_type_and_description(self):
        assert pubmed.split_label("Body Regions - Anatomical areas of the body.") == (
            "Body Regions", "Anatomical areas of the body.")

    def test_no_separator_returns_bare_type(self):
        assert pubmed.split_label("JustAType") == ("JustAType", None)


class TestConvertPubmedAbstractsNer:
    def test_uses_only_the_type_prefix_as_label(self):
        row = {
            "tokenized_text": ["The", "extremities", "were", "swollen", "."],
            "ner": [[1, 1, "Body Regions - Anatomical areas of the body."]],
        }
        rec = pubmed.convert_row(row)
        assert rec is not None
        assert rec["ner"] == [[1, 1, "Body Regions"]]
        assert rec["relations"] == []

    def test_drops_out_of_bounds_span(self):
        row = {"tokenized_text": ["a", "b"], "ner": [[0, 5, "X - desc"]]}
        assert pubmed.convert_row(row) is None
