import pytest

from gliner.data_processing.windowing import (
    default_stride,
    merge_windowed_relations,
    merge_windowed_spans,
    prepare_windowed_items,
    split_windows,
    window_tokens,
    window_training_record,
)
from gliner.decoding.decoder import Span


class TestSplitWindows:
    def test_short_input_returns_single_window(self):
        assert split_windows(10, max_len=384, stride=96) == [(0, 10)]

    def test_full_coverage_no_gaps(self):
        windows = split_windows(1000, max_len=384, stride=96)
        covered = set()
        for start, end in windows:
            covered.update(range(start, end))
        assert covered == set(range(1000))

    def test_consecutive_windows_overlap_by_stride(self):
        windows = split_windows(1000, max_len=384, stride=96)
        for (s1, e1), (s2, _) in zip(windows, windows[1:]):
            assert e1 - s2 == 96

    def test_last_window_not_needlessly_short(self):
        windows = split_windows(1000, max_len=384, stride=96)
        assert windows[-1][1] == 1000

    def test_degenerate_stride_does_not_infinite_loop(self):
        windows = split_windows(100, max_len=10, stride=10)
        assert windows[-1][1] == 100
        assert len(windows) < 100


class TestDefaultStride:
    def test_quarter_of_max_len(self):
        assert default_stride(384) == 96

    def test_floored_at_one(self):
        assert default_stride(2) == 1


class TestWindowTrainingRecord:
    def test_short_record_returned_unchanged(self):
        record = {"tokenized_text": ["a", "b", "c"], "ner": [(0, 0, "X")]}
        assert window_training_record(record, max_len=10, stride=2) == [record]

    def test_entity_past_cutoff_is_recovered(self):
        # 20 tokens, max_len=10, stride=4 -> windows (0,10),(6,16),(12,20).
        # An entity at word index 15 (past a naive tokens[:10] truncation)
        # must reappear, correctly reindexed, in some window.
        tokens = [f"w{i}" for i in range(20)]
        ner = [(15, 15, "PER")]
        record = {"tokenized_text": tokens, "ner": ner}

        windows = window_training_record(record, max_len=10, stride=4)

        recovered = False
        for w in windows:
            w_start = tokens.index(w["tokenized_text"][0])
            for e_start, e_end, label in w["ner"]:
                if w_start + e_start == 15 and w_start + e_end == 15 and label == "PER":
                    recovered = True
        assert recovered

    def test_entity_labeled_in_every_containing_window(self):
        # Entity at (6,6) sits in the overlap of windows (0,10) and (6,16).
        tokens = [f"w{i}" for i in range(20)]
        ner = [(6, 6, "PER")]
        record = {"tokenized_text": tokens, "ner": ner}

        windows = window_training_record(record, max_len=10, stride=4)

        containing = [w for w in windows if any(lab == "PER" for _, _, lab in w["ner"])]
        assert len(containing) == 2

    def test_relation_between_co_windowed_entities_survives(self):
        tokens = [f"w{i}" for i in range(20)]
        ner = [(1, 1, "PER"), (3, 3, "ORG")]
        relations = [(0, 1, "WORKS_FOR")]
        record = {"tokenized_text": tokens, "ner": ner, "relations": relations}

        windows = window_training_record(record, max_len=10, stride=4)

        first_window = windows[0]
        assert len(first_window["relations"]) == 1
        head_idx, tail_idx, label = first_window["relations"][0]
        assert first_window["ner"][head_idx][2] == "PER"
        assert first_window["ner"][tail_idx][2] == "ORG"
        assert label == "WORKS_FOR"

    def test_relation_across_non_cowindowed_entities_is_dropped(self):
        # Head at 0, tail at 19 -- 20 tokens apart, never co-contained in a
        # max_len=10 window regardless of stride.
        tokens = [f"w{i}" for i in range(20)]
        ner = [(0, 0, "PER"), (19, 19, "ORG")]
        relations = [(0, 1, "WORKS_FOR")]
        record = {"tokenized_text": tokens, "ner": ner, "relations": relations}

        windows = window_training_record(record, max_len=10, stride=4)

        assert all(len(w["relations"]) == 0 for w in windows)


class TestWindowTokens:
    def test_short_input_single_window_offset_zero(self):
        tokens = ["a", "b", "c"]
        assert window_tokens(tokens, max_len=10, stride=2) == [(tokens, 0)]

    def test_offsets_match_split_windows(self):
        tokens = [f"w{i}" for i in range(20)]
        chunks = window_tokens(tokens, max_len=10, stride=4)
        for chunk, offset in chunks:
            assert chunk == tokens[offset : offset + len(chunk)]


class TestPrepareWindowedItems:
    def test_owner_and_offset_bookkeeping(self):
        items = [
            {"tokenized_text": [f"a{i}" for i in range(20)]},
            {"tokenized_text": ["short", "doc"]},
        ]
        expanded, owner_idx, offsets = prepare_windowed_items(items, max_len=10, stride=4)

        assert owner_idx.count(0) > 1
        assert owner_idx.count(1) == 1
        assert len(expanded) == len(owner_idx) == len(offsets)
        # The short doc's single window has offset 0 and is unchanged.
        short_idx = owner_idx.index(1)
        assert offsets[short_idx] == 0
        assert expanded[short_idx]["tokenized_text"] == ["short", "doc"]


class TestMergeWindowedSpans:
    def test_duplicate_span_across_overlapping_windows_collapses(self):
        span_a = Span(start=2, end=2, entity_type="PER", score=0.7)
        span_b = Span(start=0, end=0, entity_type="PER", score=0.9)  # same doc position via offset
        windows = [([span_a], 0), ([span_b], 2)]

        merged, index_maps = merge_windowed_spans(windows)

        assert len(merged) == 1
        assert merged[0].start == 2 and merged[0].end == 2
        assert merged[0].score == 0.9  # highest score kept
        assert index_maps == [[0], [0]]

    def test_distinct_spans_both_kept(self):
        span_a = Span(start=0, end=0, entity_type="PER", score=0.7)
        span_b = Span(start=0, end=0, entity_type="ORG", score=0.9)
        windows = [([span_a], 0), ([span_b], 10)]

        merged, index_maps = merge_windowed_spans(windows)

        assert len(merged) == 2
        assert index_maps == [[0], [1]]


class TestMergeWindowedRelations:
    def test_relation_indices_remapped_through_index_maps(self):
        # Window 0 sees entities [PER, ORG] locally; window 1 (overlap) sees
        # the same two entities but in reverse local order.
        windows_rels = [[(0, "WORKS_FOR", 1, 0.6)], [(1, "WORKS_FOR", 0, 0.8)]]
        index_maps = [[0, 1], [1, 0]]  # window1 local0->merged1, local1->merged0

        merged = merge_windowed_relations(windows_rels, index_maps)

        assert merged == [(0, "WORKS_FOR", 1, 0.8)]  # deduped, higher score kept

    def test_distinct_relations_both_kept(self):
        windows_rels = [[(0, "WORKS_FOR", 1, 0.6)], [(0, "LIVES_IN", 1, 0.5)]]
        index_maps = [[0, 1], [0, 1]]

        merged = merge_windowed_relations(windows_rels, index_maps)

        assert len(merged) == 2

    def test_out_of_range_index_skipped(self):
        windows_rels = [[(0, "WORKS_FOR", 5, 0.6)]]
        index_maps = [[0]]

        merged = merge_windowed_relations(windows_rels, index_maps)

        assert merged == []
