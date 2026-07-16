"""Unit tests for data/convert_biored.py.

Hermetic -- a small synthetic BioC-shaped document. Also run against 30
real documents downloaded from NCBI's BIORED.zip during development.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_biored  # noqa: E402


def _decode(tokens, ner, idx):
    s, e, label = ner[idx]
    return " ".join(tokens[s:e + 1]), label


class TestReconstructText:
    def test_places_each_passage_at_its_absolute_offset(self):
        passages = [
            {"offset": 0, "text": "First passage."},
            {"offset": 20, "text": "Second passage."},
        ]
        text = convert_biored._reconstruct_text(passages)
        assert text[0:14] == "First passage."
        assert text[20:35] == "Second passage."


class TestConvertBiored:
    def test_resolves_absolute_char_offsets_across_passages(self):
        doc = {
            "passages": [
                {"offset": 0, "text": "Gene X causes disease.",
                 "annotations": [{"infons": {"type": "GeneOrGeneProduct", "identifier": "G1"},
                                   "text": "Gene X", "locations": [{"offset": 0, "length": 6}]}]},
                {"offset": 30, "text": "The disease is severe.",
                 "annotations": [{"infons": {"type": "DiseaseOrPhenotypicFeature", "identifier": "D1"},
                                   "text": "disease", "locations": [{"offset": 34, "length": 7}]}]},
            ],
            "relations": [{"infons": {"entity1": "G1", "entity2": "D1", "type": "Association"}}],
        }
        rec = convert_biored.convert_doc(doc)
        assert rec is not None
        tokens, ner, relations = rec["tokenized_text"], rec["ner"], rec["relations"]
        assert _decode(tokens, ner, 0) == ("Gene X", "GeneOrGeneProduct")
        assert _decode(tokens, ner, 1) == ("disease", "DiseaseOrPhenotypicFeature")
        assert relations == [[0, 1, "Association"]]

    def test_relation_uses_representative_first_mention_per_identifier(self):
        doc = {
            "passages": [
                {"offset": 0, "text": "X causes Y. X also causes Z.",
                 "annotations": [
                     {"infons": {"type": "GeneOrGeneProduct", "identifier": "G1"},
                      "text": "X", "locations": [{"offset": 0, "length": 1}]},
                     {"infons": {"type": "DiseaseOrPhenotypicFeature", "identifier": "D1"},
                      "text": "Y", "locations": [{"offset": 9, "length": 1}]},
                     {"infons": {"type": "GeneOrGeneProduct", "identifier": "G1"},
                      "text": "X", "locations": [{"offset": 12, "length": 1}]},
                 ]},
            ],
            "relations": [{"infons": {"entity1": "G1", "entity2": "D1", "type": "Association"}}],
        }
        rec = convert_biored.convert_doc(doc)
        head_idx = rec["relations"][0][0]
        # Representative mention is the *first* occurrence of G1, not the second.
        s, e, _ = rec["ner"][head_idx]
        assert s == 0

    def test_returns_none_without_annotations(self):
        doc = {"passages": [{"offset": 0, "text": "No entities.", "annotations": []}]}
        assert convert_biored.convert_doc(doc) is None
