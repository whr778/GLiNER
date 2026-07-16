"""Unit tests for GLiNER's plain-NER (non-event) dataset converters in data/.

Hermetic -- small synthetic inputs shaped like the real raw source schema,
no network access. Each converter was additionally run against 100 real
rows streamed from its source HuggingFace dataset during development.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_gliner_multilingual as gm  # noqa: E402
import convert_knowledgator_gliner as kg  # noqa: E402


class TestConvertGlinerMultilingual:
    def test_unwraps_json_quoted_labels_and_keeps_token_indices(self):
        row = {
            "tokenized_text": ["Der", "Film", "wurde", "in", "Los", "Angeles", "gedreht", "."],
            "ner": [["4", "5", '"location"']],
        }
        rec = gm.convert_row(row)
        assert rec is not None
        assert rec["ner"] == [[4, 5, "location"]]
        assert rec["relations"] == []
        assert rec["tokenized_text"] == row["tokenized_text"]

    def test_drops_out_of_bounds_span(self):
        row = {"tokenized_text": ["a", "b"], "ner": [["0", "5", '"x"']]}
        assert gm.convert_row(row) is None

    def test_returns_none_without_any_ner(self):
        assert gm.convert_row({"tokenized_text": ["a", "b"], "ner": []}) is None

    def test_unwrap_label_handles_bare_and_json_quoted(self):
        assert gm.unwrap_label('"location"') == "location"
        assert gm.unwrap_label("bare") == "bare"
        assert gm.unwrap_label("") is None
        assert gm.unwrap_label(None) is None


class TestConvertKnowledgatorGliner:
    def test_strips_prompt_prefix_and_rebases_indices(self):
        row = {
            "tokenized_text": ["Identify", "entities", "Text", ":", "\n", "Gurgurnica", "is", "a", "village"],
            "ner": [["5", "5", '"Village"']],
        }
        rec = kg.convert_row(row)
        assert rec is not None
        assert rec["tokenized_text"] == ["Gurgurnica", "is", "a", "village"]
        assert rec["ner"] == [[0, 0, "Village"]]
        assert rec["relations"] == []

    def test_find_body_start_locates_the_text_colon_newline_trio(self):
        tokens = ["a", "b", "Text", ":", "\n", "c"]
        assert kg.find_body_start(tokens) == 5

    def test_returns_none_without_body_marker(self):
        row = {"tokenized_text": ["no", "marker", "here"], "ner": [["0", "0", '"x"']]}
        assert kg.convert_row(row) is None

    def test_drops_span_that_falls_inside_the_stripped_prompt(self):
        row = {
            "tokenized_text": ["prompt", "Text", ":", "\n", "body"],
            "ner": [["0", "0", '"x"']],  # points at "prompt", before body_start
        }
        assert kg.convert_row(row) is None
