"""Unit tests for data/convert_hf_token_ner.py.

Hermetic -- small synthetic BIO-tag sequences. Also run against real rows
from two different tag encodings (kaznerd's BIO strings, bc4chemd's
ClassLabel ints) streamed from HuggingFace during development.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_hf_token_ner as hftn  # noqa: E402


class TestBioToNerSpans:
    def test_decodes_bio_string_tags(self):
        tags = ["B-LAW", "I-LAW", "O", "B-ORG"]
        assert hftn.bio_to_ner_spans(tags) == [("LAW", 0, 1), ("ORG", 3, 3)]

    def test_tag_without_prefix_is_its_own_type(self):
        # A bare tag like "GENE" (no B-/I- prefix) is its own type, treated
        # as a single-token begin -- matches GLiNER2's `partition` fallback.
        assert hftn.bio_to_ner_spans(["GENE", "O"]) == [("GENE", 0, 0)]

    def test_adjacent_begin_of_same_type_starts_new_span(self):
        assert hftn.bio_to_ner_spans(["B-PER", "B-PER"]) == [("PER", 0, 0), ("PER", 1, 1)]

    def test_span_open_at_sentence_end(self):
        assert hftn.bio_to_ner_spans(["O", "B-LOC", "I-LOC"]) == [("LOC", 1, 2)]

    def test_all_o_returns_no_spans(self):
        assert hftn.bio_to_ner_spans(["O", "O"]) == []


class TestIdToName:
    def test_uses_label_file_names_when_given(self):
        assert hftn._id_to_name({}, "tags", ["O", "B-X"]) == ["O", "B-X"]

    def test_reads_classlabel_feature_names(self):
        class _Feature:
            names = ["O", "B-GENE", "I-GENE"]

        class _Sequence:
            feature = _Feature()

        assert hftn._id_to_name({"ner_tags": _Sequence()}, "ner_tags", None) == ["O", "B-GENE", "I-GENE"]
