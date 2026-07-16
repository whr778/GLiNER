"""Unit tests for data/convert_bio_ner_relations.py.

Hermetic -- a small synthetic single-passage BioC-shaped row. Also run
against 50 real rows streamed from HuggingFace during development.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_bio_ner_relations as bnr  # noqa: E402


class TestConvertBioNerRelations:
    def test_resolves_relation_through_arg_ids(self):
        row = {
            "passages": [{"type": "full_text", "text": ["MST3 binds PRKCB1 directly."]}],
            "entities": [
                {"id": "T1", "type": "GeneProtein", "text": ["MST3"], "offsets": [[0, 4]]},
                {"id": "T2", "type": "GeneProtein", "text": ["PRKCB1"], "offsets": [[11, 17]]},
            ],
            "relations": [{"type": "bind", "arg1_id": "T1", "arg2_id": "T2"}],
        }
        rec = bnr.convert_row(row, bnr.DEFAULT_SKIP_TYPES)
        assert rec is not None
        tokens, ner, relations = rec["tokenized_text"], rec["ner"], rec["relations"]
        assert relations == [[0, 1, "bind"]]
        assert " ".join(tokens[ner[0][0]:ner[0][1] + 1]) == "MST3"
        assert " ".join(tokens[ner[1][0]:ner[1][1] + 1]) == "PRKCB1"

    def test_skips_default_umlsterm_type(self):
        row = {
            "passages": [{"type": "full_text", "text": ["A gene here."]}],
            "entities": [{"id": "T1", "type": "umlsterm", "text": ["gene"], "offsets": [[2, 6]]}],
            "relations": [],
        }
        assert bnr.convert_row(row, bnr.DEFAULT_SKIP_TYPES) is None

    def test_drops_multi_passage_documents(self):
        row = {
            "passages": [
                {"type": "title", "text": ["Title"]},
                {"type": "full_text", "text": ["Body"]},
            ],
            "entities": [{"id": "T1", "type": "gene", "text": ["Body"], "offsets": [[0, 4]]}],
            "relations": [],
        }
        assert bnr.convert_row(row, set()) is None

    def test_drops_relation_with_unresolved_arg(self):
        row = {
            "passages": [{"type": "full_text", "text": ["MST3 alone."]}],
            "entities": [{"id": "T1", "type": "GeneProtein", "text": ["MST3"], "offsets": [[0, 4]]}],
            "relations": [{"type": "bind", "arg1_id": "T1", "arg2_id": "T99"}],
        }
        rec = bnr.convert_row(row, bnr.DEFAULT_SKIP_TYPES)
        assert rec["relations"] == []
