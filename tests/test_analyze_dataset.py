"""Unit tests for scripts/analyze_dataset.py.

Hermetic -- small synthetic records shaped like the real GLiNER schema.
Also run against real converted data during development (wikievents.train
correctly flagged as trigger/argument-shaped, docred.train correctly
flagged as a flat relation schema, nuner.train correctly hit the
no-relations branch) -- see the session notes, not re-asserted here since
that requires the real data files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_dataset as ad  # noqa: E402


class TestNumericSummary:
    def test_empty_returns_zero_count(self):
        assert ad._numeric_summary([]) == {"count": 0}

    def test_basic_stats(self):
        s = ad._numeric_summary([1, 2, 3, 4])
        assert s["count"] == 4
        assert s["min"] == 1
        assert s["max"] == 4
        assert s["mean"] == 2.5
        assert s["median"] == 2.5


class TestComputeTokenStats:
    def test_lengths_from_tokenized_text(self):
        records = [{"tokenized_text": ["a", "b", "c"]}, {"tokenized_text": ["a"]}]
        s = ad.compute_token_stats(records)
        assert s["min"] == 1
        assert s["max"] == 3

    def test_missing_tokenized_text_counts_as_zero(self):
        s = ad.compute_token_stats([{}])
        assert s["min"] == 0


class TestComputeEntityStats:
    def test_counts_entities_and_types(self):
        records = [
            {"ner": [[0, 0, "PER"], [2, 3, "ORG"]]},
            {"ner": [[0, 0, "PER"]]},
        ]
        s = ad.compute_entity_stats(records)
        assert s["total_entities"] == 3
        assert s["type_counts"]["PER"] == 2
        assert s["type_counts"]["ORG"] == 1
        assert s["distinct_types"] == 2

    def test_span_width_is_inclusive(self):
        # (start=2, end=3) is a 2-token span.
        records = [{"ner": [[2, 3, "ORG"]]}]
        s = ad.compute_entity_stats(records)
        assert s["span_width_tokens"]["min"] == 2

    def test_malformed_entries_counted_not_crashed(self):
        records = [{"ner": [[0, 0]]}]  # missing label
        s = ad.compute_entity_stats(records)
        assert s["malformed_entries"] == 1
        assert s["total_entities"] == 1  # still counted toward per-record total

    def test_missing_ner_key_treated_as_empty(self):
        s = ad.compute_entity_stats([{}])
        assert s["total_entities"] == 0


class TestComputeRelationStats:
    def test_counts_relations_and_types(self):
        records = [
            {"ner": [[0, 0, "PER"], [3, 3, "ORG"]], "relations": [[0, 1, "WORKS_FOR"]]},
        ]
        s = ad.compute_relation_stats(records)
        assert s["total_relations"] == 1
        assert s["type_counts"]["WORKS_FOR"] == 1

    def test_head_tail_type_pair_and_distance(self):
        records = [
            {"ner": [[0, 0, "PER"], [5, 5, "ORG"]], "relations": [[0, 1, "WORKS_FOR"]]},
        ]
        s = ad.compute_relation_stats(records)
        assert s["head_tail_type_pairs"]["PER -> ORG"] == 1
        assert s["head_tail_token_distance"]["min"] == 5

    def test_self_relation_flagged(self):
        records = [{"ner": [[0, 0, "PER"]], "relations": [[0, 0, "SELF"]]}]
        s = ad.compute_relation_stats(records)
        assert s["self_relations"] == 1

    def test_out_of_range_index_counted_as_malformed(self):
        records = [{"ner": [[0, 0, "PER"]], "relations": [[0, 5, "BAD"]]}]
        s = ad.compute_relation_stats(records)
        assert s["malformed_entries"] == 1

    def test_no_relations_gives_zero_totals(self):
        records = [{"ner": [[0, 0, "PER"]], "relations": []}]
        s = ad.compute_relation_stats(records)
        assert s["total_relations"] == 0


class TestComputeEventView:
    def test_trigger_types_derived_from_relation_heads(self):
        # attacked (idx 1) is always the head -> trigger type.
        records = [
            {"ner": [[0, 0, "PER"], [1, 1, "Attack"], [2, 2, "LOC"]],
             "relations": [[1, 0, "Attacker"], [1, 2, "Place"]]},
        ]
        ev = ad.compute_event_view(records)
        assert ev["trigger_types"] == ["Attack"]
        assert sorted(ev["argument_only_types"]) == ["LOC", "PER"]
        assert ev["looks_event_shaped"] is True

    def test_flat_relation_schema_not_event_shaped(self):
        # Both PER and ORG appear as heads somewhere -> every type is a "trigger".
        records = [
            {"ner": [[0, 0, "PER"], [1, 1, "ORG"]], "relations": [[0, 1, "WORKS_FOR"]]},
            {"ner": [[0, 0, "ORG"], [1, 1, "PER"]], "relations": [[0, 1, "EMPLOYS"]]},
        ]
        ev = ad.compute_event_view(records)
        assert ev["looks_event_shaped"] is False

    def test_no_relations_no_triggers(self):
        records = [{"ner": [[0, 0, "PER"]], "relations": []}]
        ev = ad.compute_event_view(records)
        assert ev["trigger_types"] == []
        assert ev["looks_event_shaped"] is False

    def test_out_of_range_head_index_skipped_not_crashed(self):
        records = [{"ner": [[0, 0, "PER"]], "relations": [[5, 0, "BAD"]]}]
        ev = ad.compute_event_view(records)
        assert ev["trigger_types"] == []


class TestComputeLanguageStats:
    def test_zero_sample_size_returns_none(self):
        assert ad.compute_language_stats([{"tokenized_text": ["a"]}], 0) is None

    def test_samples_up_to_limit(self):
        records = [{"tokenized_text": ["hello", "world"]} for _ in range(10)]
        stats = ad.compute_language_stats(records, 3)
        assert stats["sampled"] == 3


class TestAnalyzeIntegration:
    def test_no_relations_dataset_skips_relation_and_event_sections(self):
        records = [{"tokenized_text": ["a"], "ner": [[0, 0, "PER"]], "relations": []}]
        report = ad.analyze(records, lang_sample_size=0)
        assert report["has_relations"] is False
        assert "relations" not in report
        assert "events" not in report

    def test_relations_dataset_includes_both_sections(self):
        records = [
            {"tokenized_text": ["a", "b"], "ner": [[0, 0, "PER"], [1, 1, "ORG"]],
             "relations": [[0, 1, "WORKS_FOR"]]},
        ]
        report = ad.analyze(records, lang_sample_size=0)
        assert report["has_relations"] is True
        assert "relations" in report
        assert "events" in report


class TestFormatHelpers:
    def test_format_summary_handles_empty(self):
        assert ad._format_summary({"count": 0}) == "n/a (no data)"

    def test_format_top_counts_reports_remaining(self):
        counter = {"a": 5, "b": 3, "c": 1}
        out = ad._format_top_counts(counter, total=9, top_n=2)
        assert "a" in out
        assert "b" in out
        assert "1 more distinct label" in out


class TestCountersToDicts:
    def test_converts_nested_counters(self):
        from collections import Counter

        nested = {"outer": {"inner": Counter({"x": 2, "y": 1})}}
        result = ad._counters_to_dicts(nested)
        assert result == {"outer": {"inner": {"x": 2, "y": 1}}}
