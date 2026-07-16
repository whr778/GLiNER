"""Unit tests for data/convert_scierc.py.

Hermetic -- small synthetic input shaped like the real raw source schema
(SciERC's own token-span NER/relations, no offset mapping needed). Also run
against 20 real documents streamed from a HuggingFace SciERC mirror during
development.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_scierc  # noqa: E402


def _decode(tokens, ner, idx):
    s, e, label = ner[idx]
    return " ".join(tokens[s:e + 1]), label


class TestConvertScierc:
    def test_converts_token_span_ner_and_relations_directly(self):
        doc = {
            "sentences": [["A", "neural", "network", "is", "used", "for", "parsing", "."]],
            "ner": [[[1, 2, "Method"], [6, 6, "Task"]]],
            "relations": [[[1, 2, 6, 6, "USED-FOR"]]],
        }
        rec = convert_scierc.convert_doc(doc)
        assert rec is not None
        tokens, ner, relations = rec["tokenized_text"], rec["ner"], rec["relations"]
        assert _decode(tokens, ner, 0) == ("neural network", "Method")
        assert _decode(tokens, ner, 1) == ("parsing", "Task")
        assert relations == [[0, 1, "USED-FOR"]]

    def test_drops_relation_whose_span_has_no_matching_ner_entry(self):
        doc = {
            "sentences": [["A", "neural", "network", "."]],
            "ner": [[[1, 2, "Method"]]],
            "relations": [[[1, 2, 3, 3, "USED-FOR"]]],  # tail span (3,3) never annotated as NER
        }
        rec = convert_scierc.convert_doc(doc)
        assert rec is not None
        assert rec["relations"] == []

    def test_dedupes_identical_spans_across_sentences(self):
        doc = {
            "sentences": [["A", "network"], ["A", "network"]],
            "ner": [[[1, 1, "Method"]], [[1, 1, "Method"]]],
            "relations": [],
        }
        rec = convert_scierc.convert_doc(doc)
        # (1, 1) is a valid span in both sentence-local lists but they refer
        # to the same document-global token index once flattened -- the
        # second sentence's actual global span would be (3, 3), and here
        # both raw entries reuse (1, 1), which is what a duplicate-entry
        # source would look like; convert_doc should not emit a duplicate
        # ner row for the same (start, end) key.
        assert len(rec["ner"]) == 1

    def test_returns_none_without_tokens(self):
        assert convert_scierc.convert_doc({"sentences": [], "ner": [], "relations": []}) is None
