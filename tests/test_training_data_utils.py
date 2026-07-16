import argparse
import json
import random
from types import SimpleNamespace

import pytest

from gliner.training.data_utils import (
    DEFAULT_THRESHOLD_GRID,
    BestModelTracker,
    annotate_languages,
    blind_test_by_language,
    detect_language,
    evaluate_and_extract_f1,
    flatten_namespace,
    format_evaluate_output,
    build_test_metrics,
    load_multi_dataset,
    print_blind_test,
    sweep_thresholds,
    warn_if_max_types_truncates,
    window_records,
)


def _write_json_array(path, records):
    path.write_text(json.dumps(records), encoding="utf-8")


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


class TestLoadMultiDataset:
    def test_none_returns_none(self):
        assert load_multi_dataset(None) is None

    def test_none_string_returns_none(self):
        assert load_multi_dataset("none") is None
        assert load_multi_dataset("None") is None
        assert load_multi_dataset("") is None

    def test_single_path_json_array(self, tmp_path):
        p = tmp_path / "a.json"
        _write_json_array(p, [{"id": 1}, {"id": 2}])
        result = load_multi_dataset(str(p))
        assert {r["id"] for r in result} == {1, 2}

    def test_single_path_jsonl(self, tmp_path):
        p = tmp_path / "a.jsonl"
        _write_jsonl(p, [{"id": 1}, {"id": 2}])
        result = load_multi_dataset(str(p))
        assert {r["id"] for r in result} == {1, 2}

    def test_list_of_paths_aggregates(self, tmp_path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.jsonl"
        _write_json_array(a, [{"id": 1}, {"id": 2}])
        _write_jsonl(b, [{"id": 3}, {"id": 4}])
        result = load_multi_dataset([str(a), str(b)])
        assert {r["id"] for r in result} == {1, 2, 3, 4}
        assert len(result) == 4

    def test_shuffle_is_deterministic_for_same_seed(self, tmp_path):
        p = tmp_path / "a.json"
        _write_json_array(p, [{"id": i} for i in range(50)])
        result1 = load_multi_dataset(str(p), seed=7)
        result2 = load_multi_dataset(str(p), seed=7)
        assert [r["id"] for r in result1] == [r["id"] for r in result2]

    def test_different_seeds_produce_different_order(self, tmp_path):
        p = tmp_path / "a.json"
        _write_json_array(p, [{"id": i} for i in range(50)])
        result1 = load_multi_dataset(str(p), seed=1)
        result2 = load_multi_dataset(str(p), seed=2)
        assert [r["id"] for r in result1] != [r["id"] for r in result2]

    def test_does_not_disturb_global_random_state(self, tmp_path):
        p = tmp_path / "a.json"
        _write_json_array(p, [{"id": i} for i in range(20)])
        random.seed(123)
        expected_next = random.random()
        random.seed(123)
        load_multi_dataset(str(p), seed=99)
        actual_next = random.random()
        assert actual_next == expected_next


def _fake_model(saved=None):
    """A stand-in model exposing save_pretrained + a config (both needed since
    maybe_save now also writes a model card from model.config)."""
    config = SimpleNamespace(
        model_name="bert-base", event_mode=False, relations_layer=None,
        max_len=384, max_width=8, max_types=25, span_mode="markerV0", triples_layer=None,
    )

    def save_pretrained(self, d):
        if saved is not None:
            saved.append(d)

    return type("M", (), {"save_pretrained": save_pretrained, "config": config})()


class TestBestModelTracker:
    def test_first_call_always_saves(self, tmp_path):
        tracker = BestModelTracker()
        saved = []
        improved = tracker.maybe_save(0.5, _fake_model(saved), tmp_path)
        assert improved is True
        assert saved == [str(tmp_path / "best")]

    def test_lower_f1_does_not_save(self, tmp_path):
        tracker = BestModelTracker()
        calls = []
        model = _fake_model(calls)
        tracker.maybe_save(0.8, model, tmp_path)
        improved = tracker.maybe_save(0.5, model, tmp_path)
        assert improved is False
        assert len(calls) == 1

    def test_higher_f1_saves_again(self, tmp_path):
        tracker = BestModelTracker()
        calls = []
        model = _fake_model(calls)
        tracker.maybe_save(0.5, model, tmp_path)
        improved = tracker.maybe_save(0.9, model, tmp_path)
        assert improved is True
        assert len(calls) == 2

    def test_saving_writes_a_model_card(self, tmp_path):
        stats = {
            "num_documents": 3, "num_entity_mentions": 5, "num_relations": 0,
            "entity_types": ["ORG", "PER"], "relation_types": [],
            "tokens_min": 2, "tokens_mean": 4.0, "tokens_max": 6,
        }
        tracker = BestModelTracker(card_data_stats=stats)
        tracker.maybe_save(0.73, _fake_model(), tmp_path)
        card = (tmp_path / "best" / "README.md").read_text()
        assert "library_name: gliner" in card       # HF frontmatter
        assert "Best validation F1:** 0.7300" in card
        assert "Documents | 3" in card               # training-data metric
        assert "`PER`" in card                        # label schema


class TestDetectLanguage:
    def test_empty_text_is_und(self):
        assert detect_language("") == "und"
        assert detect_language("   ") == "und"

    def test_english_text_detected(self):
        assert detect_language("The quick brown fox jumps over the lazy dog") == "en"

    def test_distinct_language_gets_different_code(self):
        en = detect_language("This is a sentence written in English for testing purposes.")
        fr = detect_language("Ceci est une phrase écrite en français à des fins de test.")
        assert en != fr


class TestAnnotateLanguages:
    def test_stamps_lang_field(self):
        records = [{"tokenized_text": ["Hello", "world", "this", "is", "English"]}]
        annotate_languages(records)
        assert records[0]["_lang"] == "en"

    def test_missing_tokenized_text_is_und(self):
        records = [{}]
        annotate_languages(records)
        assert records[0]["_lang"] == "und"


class TestEvaluateAndExtractF1:
    def test_ner_only_shape(self):
        model = type("M", (), {"evaluate": lambda self, records, **kw: ("report", 0.75)})()
        f1, output = evaluate_and_extract_f1(model, [])
        assert f1 == 0.75
        assert output == "report"

    def test_relex_shape_averages(self):
        model = type("M", (), {
            "evaluate": lambda self, records, **kw: (("ner_report", 0.6), ("rel_report", 0.4))
        })()
        f1, output = evaluate_and_extract_f1(model, [])
        assert f1 == pytest.approx(0.5)
        assert output == (("ner_report", 0.6), ("rel_report", 0.4))

    def test_kwargs_forwarded(self):
        captured = {}

        def fake_evaluate(self, records, **kw):
            captured.update(kw)
            return ("report", 1.0)

        model = type("M", (), {"evaluate": fake_evaluate})()
        evaluate_and_extract_f1(model, [], threshold=0.3, batch_size=4)
        assert captured == {"threshold": 0.3, "batch_size": 4}

    def test_rel_metric_weight_biases_toward_role_f1(self):
        model = type("M", (), {
            "evaluate": lambda self, records, **kw: (("ner", 0.9), ("rel", 0.1))
        })()
        # 0.5 = plain average; higher weights the (low) role F1 more, lowering the metric
        assert evaluate_and_extract_f1(model, [], rel_metric_weight=0.5)[0] == pytest.approx(0.5)
        assert evaluate_and_extract_f1(model, [], rel_metric_weight=0.75)[0] == pytest.approx(0.3)
        assert evaluate_and_extract_f1(model, [], rel_metric_weight=1.0)[0] == pytest.approx(0.1)

    def test_rel_metric_weight_ignored_for_ner_only(self):
        model = type("M", (), {"evaluate": lambda self, records, **kw: ("report", 0.75)})()
        assert evaluate_and_extract_f1(model, [], rel_metric_weight=0.9)[0] == 0.75

    def test_rel_metric_weight_not_forwarded_to_evaluate(self):
        captured = {}

        def fake_evaluate(self, records, **kw):
            captured.update(kw)
            return ("report", 1.0)

        model = type("M", (), {"evaluate": fake_evaluate})()
        evaluate_and_extract_f1(model, [], rel_metric_weight=0.8, threshold=0.3)
        assert "rel_metric_weight" not in captured and captured == {"threshold": 0.3}

    def test_entity_types_derived_from_full_records(self):
        captured = {}

        def fake_evaluate(self, records, **kw):
            captured.update(kw)
            return ("report", 1.0)

        model = type("M", (), {"evaluate": fake_evaluate})()
        records = [
            {"ner": [[0, 0, "PER"], [1, 1, "ORG"]]},
            {"ner": [[0, 0, "LOC"]]},
        ]
        evaluate_and_extract_f1(model, records)
        assert captured["entity_types"] == ["LOC", "ORG", "PER"]

    def test_explicit_entity_types_not_overridden(self):
        captured = {}

        def fake_evaluate(self, records, **kw):
            captured.update(kw)
            return ("report", 1.0)

        model = type("M", (), {"evaluate": fake_evaluate})()
        records = [{"ner": [[0, 0, "PER"]]}]
        evaluate_and_extract_f1(model, records, entity_types=["CUSTOM"])
        assert captured["entity_types"] == ["CUSTOM"]


class TestSweepThresholds:
    def test_picks_threshold_with_best_f1(self):
        f1_by_threshold = {0.1: 0.2, 0.3: 0.9, 0.5: 0.4}

        def fake_evaluate(self, records, threshold, **kw):
            return ("report", f1_by_threshold[threshold])

        model = type("M", (), {"evaluate": fake_evaluate})()
        best_threshold, best_f1, results = sweep_thresholds(model, [], thresholds=(0.1, 0.3, 0.5))

        assert best_threshold == 0.3
        assert best_f1 == 0.9
        assert results == f1_by_threshold

    def test_forwards_extra_evaluate_kwargs_to_every_candidate(self):
        captured = []

        def fake_evaluate(self, records, threshold, **kw):
            captured.append((threshold, kw))
            return ("report", 0.5)

        model = type("M", (), {"evaluate": fake_evaluate})()
        sweep_thresholds(model, [], thresholds=(0.1, 0.5), window_stride=128)

        assert captured == [(0.1, {"window_stride": 128}), (0.5, {"window_stride": 128})]

    def test_default_grid_is_five_coarse_points(self):
        assert DEFAULT_THRESHOLD_GRID == (0.1, 0.3, 0.5, 0.7, 0.9)


class TestBlindTestByLanguage:
    def test_groups_by_language_and_runs_all_pass(self, capsys):
        records = [
            {"tokenized_text": ["Hello", "world", "this", "is", "English", "text"]},
            {"tokenized_text": ["Ceci", "est", "une", "phrase", "écrite", "en", "français"]},
        ]
        calls = []

        def fake_evaluate(self, recs, **kw):
            calls.append(len(recs))
            return ("report", 0.5)

        model = type("M", (), {"evaluate": fake_evaluate})()
        f1, output = blind_test_by_language(model, records, {})

        # one call per distinct language + one for "all"
        assert len(calls) == len(set(r["_lang"] for r in records)) + 1
        assert calls[-1] == len(records)
        assert f1 == 0.5

        out = capsys.readouterr().out
        assert "All languages combined" in out
        assert "Blind test: all" in out

    def test_entity_types_derived_from_full_set_not_per_language_subset(self):
        records = [
            {"tokenized_text": ["Hello", "world", "this", "is", "English", "text"], "ner": [[0, 0, "PER"]]},
            {"tokenized_text": ["Ceci", "est", "une", "phrase", "écrite", "en", "français"], "ner": [[0, 0, "LOC"]]},
        ]
        captured_entity_types = []

        def fake_evaluate(self, recs, **kw):
            captured_entity_types.append(kw.get("entity_types"))
            return ("report", 0.5)

        model = type("M", (), {"evaluate": fake_evaluate})()
        blind_test_by_language(model, records, {})

        # Every call (per-language and the all-combined pass) sees the full
        # PER+LOC vocabulary, not just whatever that subset alone contains.
        assert all(types == ["LOC", "PER"] for types in captured_entity_types)


class TestPrintBlindTest:
    def test_prints_name_and_f1(self, capsys):
        print_blind_test("eng", 0.8765, "some report")
        out = capsys.readouterr().out
        assert "eng" in out
        assert "0.8765" in out
        assert "some report" in out


class TestFlattenNamespace:
    def test_merges_sections_into_one_namespace(self):
        cfg = argparse.Namespace(
            model=argparse.Namespace(model_name="bert", hidden_size=128),
            data=argparse.Namespace(train_data="train.json"),
            training=argparse.Namespace(num_steps=10, eval_every=2),
        )
        flat = flatten_namespace(cfg)
        assert flat.model_name == "bert"
        assert flat.hidden_size == 128
        assert flat.train_data == "train.json"
        assert flat.num_steps == 10
        assert flat.eval_every == 2

    def test_does_not_mutate_original_sections(self):
        cfg = argparse.Namespace(
            model=argparse.Namespace(a=1),
            data=argparse.Namespace(b=2),
            training=argparse.Namespace(c=3),
        )
        flatten_namespace(cfg)
        assert not hasattr(cfg.model, "b")
        assert not hasattr(cfg.data, "c")


class TestBuildTestMetrics:
    def test_ner_string_output_has_no_head_split(self):
        m = build_test_metrics(0.8, "Strict - P: 80%\tR: 80%\tF1: 80%", event_mode=False)
        assert m["overall_f1"] == 0.8
        assert "Strict" in m["report"]
        assert "entity_f1" not in m and "pair_f1" not in m

    def test_event_tuple_output_splits_head_f1s_and_labels(self):
        output = (("entity report", 0.6), ("role report", 0.4))
        m = build_test_metrics(0.5, output, event_mode=True)
        assert m["entity_f1"] == 0.6 and m["pair_f1"] == 0.4
        assert m["entity_label"] == "Trigger/entity" and m["pair_label"] == "Event role"
        assert "Trigger/entity" in m["report"] and "Event role" in m["report"]

    def test_relex_tuple_uses_entity_relation_labels(self):
        m = build_test_metrics(0.5, (("e", 0.6), ("r", 0.4)), event_mode=False)
        assert m["entity_label"] == "Entity" and m["pair_label"] == "Relation"


class TestWarnIfMaxTypesTruncates:
    RECORDS = [
        {"ner": [[0, 0, "A"], [1, 1, "B"], [2, 2, "C"]], "relations": [[0, 1, "r1"], [0, 2, "r2"]]},
    ]

    def test_warns_when_entity_types_exceed_max_types(self):
        with pytest.warns(UserWarning, match=r"3 distinct entity types"):
            warn_if_max_types_truncates(self.RECORDS, max_types=2)

    def test_warns_when_relation_types_exceed_max_types(self):
        with pytest.warns(UserWarning, match=r"2 distinct relation types"):
            warn_if_max_types_truncates(self.RECORDS, max_types=1)

    def test_silent_when_max_types_covers_all(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            warn_if_max_types_truncates(self.RECORDS, max_types=5)

    def test_silent_on_missing_inputs(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_max_types_truncates(None, 5)
            warn_if_max_types_truncates(self.RECORDS, None)


class TestWindowRecords:
    def test_short_records_pass_through_unchanged(self):
        records = [{"tokenized_text": ["a", "b"], "ner": []}]
        assert window_records(records, max_len=10) == records

    def test_long_record_expands_into_multiple_windows(self):
        tokens = [f"w{i}" for i in range(20)]
        records = [{"tokenized_text": tokens, "ner": [(19, 19, "PER")]}]

        windowed = window_records(records, max_len=10, stride=4)

        assert len(windowed) > 1
        assert any(any(lab == "PER" for _, _, lab in r["ner"]) for r in windowed)


class TestFormatEvaluateOutput:
    def test_ner_only_passthrough(self):
        assert format_evaluate_output("P: 10%\tR: 20%\tF1: 15%\n") == "P: 10%\tR: 20%\tF1: 15%"

    def test_relex_shape_labels_entity_and_relation(self):
        output = (("P: 70%\tR: 70%\tF1: 70%\n", 0.7), ("P: 10%\tR: 10%\tF1: 10%\n", 0.1))
        formatted = format_evaluate_output(output, event_mode=False)
        assert "Entity:" in formatted
        assert "Relation:" in formatted
        assert "Trigger" not in formatted
        assert "Event role" not in formatted

    def test_event_shape_labels_trigger_and_event_role(self):
        output = (("P: 70%\tR: 70%\tF1: 70%\n", 0.7), ("P: 0%\tR: 0%\tF1: 0%\n", 0.0))
        formatted = format_evaluate_output(output, event_mode=True)
        assert "Trigger/entity:" in formatted
        assert "Event role:" in formatted
        assert "Relation:" not in formatted


class TestPrintBlindTestEventLabeling:
    def test_relex_output_labeled_as_relation(self, capsys):
        output = (("P: 70%\tR: 70%\tF1: 70%\n", 0.7), ("P: 10%\tR: 10%\tF1: 10%\n", 0.1))
        print_blind_test("all", 0.4, output, event_mode=False)
        out = capsys.readouterr().out
        assert "Entity:" in out
        assert "Relation:" in out

    def test_event_output_labeled_as_event_role(self, capsys):
        output = (("P: 70%\tR: 70%\tF1: 70%\n", 0.7), ("P: 0%\tR: 0%\tF1: 0%\n", 0.0))
        print_blind_test("all", 0.35, output, event_mode=True)
        out = capsys.readouterr().out
        assert "Trigger/entity:" in out
        assert "Event role:" in out


class TestBlindTestByLanguageEventLabeling:
    def test_derives_event_mode_from_model_config(self, capsys):
        records = [
            {"tokenized_text": ["Hello", "world"], "ner": [[0, 0, "PER"]]},
        ]

        def fake_evaluate(self, recs, **kw):
            return (("P: 70%\tR: 70%\tF1: 70%\n", 0.7), ("P: 0%\tR: 0%\tF1: 0%\n", 0.0))

        config = type("Cfg", (), {"event_mode": True})()
        model = type("M", (), {"evaluate": fake_evaluate, "config": config})()

        blind_test_by_language(model, records, {})

        out = capsys.readouterr().out
        assert "Trigger/entity:" in out
        assert "Event role:" in out

    def test_defaults_to_relation_label_without_config(self, capsys):
        records = [
            {"tokenized_text": ["Hello", "world"], "ner": [[0, 0, "PER"]]},
        ]

        def fake_evaluate(self, recs, **kw):
            return (("P: 70%\tR: 70%\tF1: 70%\n", 0.7), ("P: 10%\tR: 10%\tF1: 10%\n", 0.1))

        model = type("M", (), {"evaluate": fake_evaluate})()

        blind_test_by_language(model, records, {})

        out = capsys.readouterr().out
        assert "Entity:" in out
        assert "Relation:" in out
