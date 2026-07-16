"""Tests for training-time model-card generation."""
from types import SimpleNamespace

from gliner.training.model_card import (
    render_model_card,
    summarize_training_data,
    write_model_card,
)

RECORDS = [
    {"tokenized_text": ["John", "attacked", "Paris"],
     "ner": [[0, 0, "PER"], [1, 1, "Attack"], [2, 2, "LOC"]],
     "relations": [[1, 0, "Attacker"], [1, 2, "Place"]]},
    {"tokenized_text": ["Bombs", "exploded"],
     "ner": [[1, 1, "Attack"]],
     "relations": []},
]


def _event_config():
    return SimpleNamespace(
        model_name="microsoft/deberta-v3-small", event_mode=True, relations_layer="dot",
        trigger_types=["Attack"], max_len=512, max_width=12, max_types=100,
        span_mode="markerV0", triples_layer=None,
    )


def test_summarize_counts_documents_types_and_tokens():
    s = summarize_training_data(RECORDS)
    assert s["num_documents"] == 2
    assert s["num_entity_mentions"] == 4
    assert s["num_relations"] == 2
    assert s["entity_types"] == ["Attack", "LOC", "PER"]
    assert s["relation_types"] == ["Attacker", "Place"]
    assert s["tokens_min"] == 2 and s["tokens_max"] == 3


def test_summarize_returns_none_for_empty():
    assert summarize_training_data([]) is None
    assert summarize_training_data(None) is None


def test_event_card_has_purpose_schema_metrics_and_f1():
    card = render_model_card(_event_config(), summarize_training_data(RECORDS), best_f1=0.4123)
    assert card.startswith("---\n")                        # HF YAML frontmatter first
    assert "- event-extraction" in card                    # task tag
    assert "event-extraction** model" in card              # purpose
    assert "Best validation F1:** 0.4123" in card
    # label schema split into triggers vs. argument entities vs. roles
    assert "Event trigger types (1)" in card and "`Attack`" in card
    assert "Argument (entity) types (2)" in card and "`PER`" in card
    assert "Argument role types (2)" in card and "`Attacker`" in card
    # training-data metrics
    assert "Documents | 2" in card
    assert "predict_relations" in card                     # usage snippet


def test_ner_card_differs_from_event_card():
    ner_cfg = SimpleNamespace(model_name="bert-base", event_mode=False, relations_layer=None,
                              max_len=384, max_width=8, max_types=25, span_mode="markerV0", triples_layer=None)
    card = render_model_card(ner_cfg, summarize_training_data(RECORDS))
    assert "named-entity-recognition** model" in card
    assert "predict_entities" in card
    assert "Best validation F1" not in card                # omitted when best_f1 is None
    assert "Argument role types" not in card               # no event schema for NER


def test_write_model_card_creates_readme(tmp_path):
    path = write_model_card(tmp_path, _event_config(), summarize_training_data(RECORDS), best_f1=0.5)
    assert path == tmp_path / "README.md"
    assert path.read_text().startswith("---\n")
