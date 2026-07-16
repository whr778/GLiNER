"""Unit tests for data/convert_pile_ner_definition.py.

Hermetic -- a small synthetic ShareGPT-shaped conversation. Also run
against 20 real records streamed from HuggingFace during development
(20/20 converted).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_pile_ner_definition as pile  # noqa: E402


def _conversation(text, qas):
    turns = [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}]
    for definition, answer_json in qas:
        turns.append({"from": "human", "value": f"What describes {definition} in the text?"})
        turns.append({"from": "gpt", "value": answer_json})
    return {"conversations": turns}


class TestStripDefinition:
    def test_extracts_definition_between_fixed_prompt(self):
        assert pile.strip_definition("What describes a type of medication in the text?") == "a type of medication"

    def test_returns_none_for_non_matching_prompt(self):
        assert pile.strip_definition("something else entirely") is None


class TestConvertRecord:
    def test_uses_definition_directly_as_label(self):
        record = _conversation("Aspirin is a type of medication.", [("a type of medication", '["Aspirin"]')])
        rec = pile.convert_record(record)
        assert rec is not None
        s, e, label = rec["ner"][0]
        assert " ".join(rec["tokenized_text"][s:e + 1]) == "Aspirin"
        assert label == "a type of medication"
        assert rec["relations"] == []

    def test_empty_answer_list_yields_no_entities(self):
        record = _conversation("Nothing here.", [("a type of medication", "[]")])
        assert pile.convert_record(record) is None

    def test_drops_surface_not_verbatim_in_text(self):
        record = _conversation("Aspirin helps.", [("a type of medication", '["Tylenol"]')])
        assert pile.convert_record(record) is None

    def test_too_few_turns_returns_none(self):
        assert pile.convert_record({"conversations": [{"from": "human", "value": "Text: x"}]}) is None
