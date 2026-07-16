"""Unit tests for GLiNER's document-level relation-extraction converters in data/.

Hermetic -- small synthetic inputs shaped like the real raw source schema.
Each converter was additionally run against 30 real documents streamed from
its source HuggingFace dataset during development.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_docred as docred  # noqa: E402
import convert_redocred as redocred  # noqa: E402


def _decode(tokens, ner, idx):
    s, e, label = ner[idx]
    return " ".join(tokens[s:e + 1]), label


_SENTS = [
    ["Micronesia", "is", "a", "country", "."],
    ["It", "is", "also", "known", "as", "the", "FSM", "."],
]

_VERTEX_SET = [
    [{"name": "Micronesia", "sent_id": 0, "pos": [0, 1], "type": "LOC"},
     {"name": "FSM", "sent_id": 1, "pos": [6, 7], "type": "LOC"}],  # cluster 0: two mentions, same entity
    [{"name": "country", "sent_id": 0, "pos": [3, 4], "type": "MISC"}],  # cluster 1
]


class TestConvertDocred:
    def test_flattens_sentences_and_resolves_cluster_mentions(self):
        row = {"sents": _SENTS, "vertexSet": _VERTEX_SET,
               "labels": {"head": [0], "tail": [1], "relation_text": ["instance of"]}}
        rec = docred.convert_row(row)
        assert rec is not None
        tokens, ner, relations = rec["tokenized_text"], rec["ner"], rec["relations"]
        # Both mentions of cluster 0 appear as separate ner entries at their real positions.
        assert _decode(tokens, ner, 0) == ("Micronesia", "LOC")
        assert _decode(tokens, ner, 1) == ("FSM", "LOC")
        assert _decode(tokens, ner, 2) == ("country", "MISC")
        # Relation links the cluster's *representative* (first) mention.
        assert relations == [[0, 2, "instance of"]]

    def test_drops_relation_with_unresolved_cluster(self):
        row = {"sents": _SENTS, "vertexSet": _VERTEX_SET,
               "labels": {"head": [0], "tail": [99], "relation_text": ["x"]}}
        rec = docred.convert_row(row)
        assert rec is not None
        assert rec["relations"] == []

    def test_returns_none_without_sentences(self):
        assert docred.convert_row({"sents": [], "vertexSet": []}) is None

    def test_mention_span_flattens_across_sentence_boundary(self):
        tokens, sent_starts = docred._flatten_sentences(_SENTS)
        assert sent_starts == [0, 5]  # sentence 0 has 5 tokens
        assert tokens[11] == "FSM"
        span = docred._mention_span(sent_starts, len(tokens), {"sent_id": 1, "pos": [6, 7]})
        assert span == (11, 11)


class TestConvertRedocred:
    def test_maps_wikidata_pid_to_relation_text(self):
        row = {"sents": _SENTS, "vertexSet": _VERTEX_SET,
               "labels": [{"h": 0, "t": 1, "r": "P31"}]}  # P31 = "instance of"
        rec = redocred.convert_row(row)
        assert rec is not None
        assert rec["relations"] == [[0, 2, "instance of"]]

    def test_unknown_pid_falls_back_to_raw_id(self):
        row = {"sents": _SENTS, "vertexSet": _VERTEX_SET,
               "labels": [{"h": 0, "t": 1, "r": "P99999"}]}
        rec = redocred.convert_row(row)
        assert rec["relations"] == [[0, 2, "P99999"]]

    def test_drops_self_relation(self):
        row = {"sents": _SENTS, "vertexSet": _VERTEX_SET,
               "labels": [{"h": 0, "t": 0, "r": "P31"}]}
        rec = redocred.convert_row(row)
        assert rec["relations"] == []
