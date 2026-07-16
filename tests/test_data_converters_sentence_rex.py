"""Unit tests for data/convert_sentence_rex.py.

Hermetic -- small synthetic inline-tagged sentences. Also run against 30
real rows streamed from HuggingFace during development (30/30 converted).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_sentence_rex as srex  # noqa: E402


class TestParseRow:
    def test_strips_tags_and_extracts_surfaces(self):
        row = {"sentences": "<e1>Pope Pius XII</e1> proclaimed him <e2>Venerable</e2> in 1985.",
               "labels": "canonization status"}
        clean, e1, e2, label = srex.parse_row(row)
        assert "<e1>" not in clean and "</e2>" not in clean
        assert clean == "Pope Pius XII proclaimed him Venerable in 1985."
        assert e1 == "Pope Pius XII"
        assert e2 == "Venerable"
        assert label == "canonization status"

    def test_missing_tag_returns_none(self):
        assert srex.parse_row({"sentences": "<e1>Only one tag</e1> here.", "labels": "x"}) is None


class TestConvertRow:
    def test_positions_computed_against_clean_text_not_tagged_text(self):
        # If spans were (incorrectly) computed against the tagged string
        # before stripping, "Venerable" would land at the wrong token index
        # once the <e1>...</e1> tag tokens are removed. This asserts the
        # decoded span, from the *converted* record, matches the real
        # surface -- proving stripping happened before tokenization.
        row = {"sentences": "<e1>Pope Pius XII</e1> proclaimed him <e2>Venerable</e2> in 1985.",
               "labels": "canonization status"}
        rec = srex.convert_row(row)
        assert rec is not None
        tokens, ner, relations = rec["tokenized_text"], rec["ner"], rec["relations"]
        h, t, rel = relations[0]
        assert " ".join(tokens[ner[h][0]:ner[h][1] + 1]) == "Pope Pius XII"
        assert " ".join(tokens[ner[t][0]:ner[t][1] + 1]) == "Venerable"
        assert rel == "canonization status"

    def test_both_entities_get_the_placeholder_type(self):
        row = {"sentences": "<e1>A</e1> relates to <e2>B</e2>.", "labels": "rel"}
        rec = srex.convert_row(row)
        assert rec["ner"][0][2] == srex.PLACEHOLDER_TYPE
        assert rec["ner"][1][2] == srex.PLACEHOLDER_TYPE

    def test_identical_surfaces_are_dropped_as_self_relation(self):
        row = {"sentences": "<e1>Acme</e1> works with <e2>Acme</e2>.", "labels": "partner"}
        assert srex.convert_row(row) is None
