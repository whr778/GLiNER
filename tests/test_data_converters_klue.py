"""Unit tests for data/convert_klue_ner.py and data/convert_klue_re.py.

Hermetic -- small synthetic inputs shaped like the real raw TSV/JSON.
Both converters were also run against 30 real rows fetched from the
canonical KLUE-benchmark GitHub release during development.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_klue_ner as klue_ner  # noqa: E402
import convert_klue_re as klue_re  # noqa: E402


class TestConvertKlueNer:
    def test_char_level_bio_decodes_to_token_index_span(self):
        chars = list("서울시")
        tags = ["B-LC", "I-LC", "I-LC"]
        rec = klue_ner.sentence_to_record(chars, tags)
        assert rec is not None
        assert rec["ner"] == [[0, 2, "LC"]]
        assert rec["tokenized_text"] == chars
        assert rec["relations"] == []

    def test_iter_records_splits_on_blank_lines(self):
        tsv = "서\tB-LC\n울\tI-LC\n\n부\tB-LC\n산\tI-LC\n"
        records = list(klue_ner.iter_records(tsv))
        assert len(records) == 2
        assert "".join(records[0]["tokenized_text"]) == "서울"
        assert "".join(records[1]["tokenized_text"]) == "부산"

    def test_returns_none_for_all_o_sentence(self):
        assert klue_ner.sentence_to_record(["a", "b"], ["O", "O"]) is None


class TestConvertKlueRe:
    def test_uses_native_char_offsets_for_both_entities(self):
        row = {
            "sentence": "비틀즈는 조지 해리슨과 함께 활동했다.",
            "subject_entity": {"word": "비틀즈", "start_idx": 0, "end_idx": 2, "type": "ORG"},
            "object_entity": {"word": "조지 해리슨", "start_idx": 5, "end_idx": 10, "type": "PER"},
            "label": "org:member_of",
        }
        rec = klue_re.convert_row(row)
        assert rec is not None
        tokens, ner, relations = rec["tokenized_text"], rec["ner"], rec["relations"]
        assert "".join(tokens[ner[0][0]:ner[0][1] + 1]) == "비틀즈"
        assert "".join(tokens[ner[1][0]:ner[1][1] + 1]) == "조지 해리슨"
        assert relations == [[0, 1, "org:member_of"]]

    def test_drops_no_relation_label(self):
        row = {
            "sentence": "A는 B다.",
            "subject_entity": {"word": "A", "start_idx": 0, "end_idx": 0, "type": "ORG"},
            "object_entity": {"word": "B", "start_idx": 2, "end_idx": 2, "type": "PER"},
            "label": "no_relation",
        }
        assert klue_re.convert_row(row) is None

    def test_drops_out_of_bounds_offset(self):
        row = {
            "sentence": "short",
            "subject_entity": {"word": "x", "start_idx": 0, "end_idx": 0, "type": "ORG"},
            "object_entity": {"word": "y", "start_idx": 100, "end_idx": 100, "type": "PER"},
            "label": "org:member_of",
        }
        assert klue_re.convert_row(row) is None
