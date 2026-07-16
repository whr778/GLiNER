from gliner.evaluation.utils import (
    _match_relaxed_sample,
    _spans_overlap,
    build_classification_report,
    compute_relaxed_prf,
    relaxed_label_counts,
    strict_label_counts,
)


class TestSpansOverlap:
    def test_identical_spans_overlap(self):
        assert _spans_overlap((5, 5), (5, 5))

    def test_shifted_spans_overlap(self):
        assert _spans_overlap((5, 7), (6, 8))

    def test_disjoint_spans_do_not_overlap(self):
        assert not _spans_overlap((5, 5), (6, 6))

    def test_touching_but_disjoint_single_token_spans_do_not_overlap(self):
        # (5,5) covers only token 5; (6,6) covers only token 6 -- adjacent, not overlapping.
        assert not _spans_overlap((5, 5), (6, 6))

    def test_relation_requires_both_head_and_tail_to_overlap(self):
        head_tail_a = (0, 0, 10, 10)
        head_overlaps_tail_does_not = (0, 0, 20, 20)
        assert not _spans_overlap(head_tail_a, head_overlaps_tail_does_not)

    def test_relation_matches_when_both_head_and_tail_overlap(self):
        head_tail_a = (0, 2, 10, 12)
        head_tail_b = (1, 3, 11, 13)
        assert _spans_overlap(head_tail_a, head_tail_b)


class TestMatchRelaxedSample:
    def test_exact_match_is_tp(self):
        tp, fp, fn = _match_relaxed_sample([["PER", (5, 5)]], [["PER", (5, 5)]])
        assert tp["PER"] == 1
        assert fp["PER"] == 0
        assert fn["PER"] == 0

    def test_overlap_only_prediction_is_tp(self):
        tp, fp, fn = _match_relaxed_sample([["PER", (5, 7)]], [["PER", (6, 8)]])
        assert tp["PER"] == 1
        assert fp["PER"] == 0
        assert fn["PER"] == 0

    def test_two_pass_prefers_exact_over_overlap(self):
        # Two gold spans; one pred exactly matches the first, and could also
        # overlap-match the second if pass 1 didn't claim it first. The
        # exact-first pass must reserve the exact pair so it isn't "stolen"
        # by a later overlap attempt against the wrong gold span.
        gold = [["A", (5, 5)], ["A", (6, 6)]]
        pred = [["A", (5, 5)]]
        tp, fp, fn = _match_relaxed_sample(gold, pred)
        assert tp["A"] == 1
        assert fn["A"] == 1  # (6, 6) stays unmatched, not stolen

    def test_duplicate_gold_spans_dedup_like_strict(self):
        # Strict counting treats gold as a set, so a duplicated (label, span)
        # entry must not inflate the relaxed false-negative count either.
        gold = [["PER", (5, 5)], ["PER", (5, 5)]]
        pred = [["PER", (5, 5)]]
        tp, fp, fn = _match_relaxed_sample(gold, pred)
        assert tp["PER"] == 1
        assert fp["PER"] == 0
        assert fn["PER"] == 0

    def test_label_mismatch_never_matches(self):
        tp, fp, fn = _match_relaxed_sample([["PER", (5, 5)]], [["ORG", (5, 5)]])
        assert tp["PER"] == 0
        assert tp["ORG"] == 0
        assert fp["ORG"] == 1
        assert fn["PER"] == 1


class TestComputeRelaxedPrf:
    def test_relaxed_never_scores_below_strict(self):
        # One sample where an exact match and a shifted-overlap match are
        # both available -- relaxed must recover at least what strict does.
        all_true = [[["PER", (5, 5)], ["ORG", (10, 12)]]]
        all_pred = [[["PER", (5, 5)], ["ORG", (11, 13)]]]

        relaxed = compute_relaxed_prf(all_true, all_pred)
        assert relaxed["f_score"] == 1.0
        assert relaxed["precision"] == 1.0
        assert relaxed["recall"] == 1.0

    def test_no_predictions_gives_zero(self):
        relaxed = compute_relaxed_prf([[["PER", (5, 5)]]], [[]])
        assert relaxed["precision"] == 0.0
        assert relaxed["recall"] == 0.0
        assert relaxed["f_score"] == 0.0

    def test_empty_gold_and_pred_gives_zero_not_nan(self):
        relaxed = compute_relaxed_prf([[]], [[]])
        assert relaxed["precision"] == 0.0
        assert relaxed["recall"] == 0.0
        assert relaxed["f_score"] == 0.0


class TestStrictLabelCounts:
    def test_per_label_tp_fp_fn(self):
        y_true = [[["PER", (0, 0)], ["ORG", (5, 7)]], [["PER", (2, 2)]]]
        y_pred = [[["PER", (0, 0)], ["ORG", (6, 8)]], [["LOC", (2, 2)]]]

        tp, fp, fn = strict_label_counts(y_true, y_pred)

        assert tp == {"PER": 1, "ORG": 0, "LOC": 0}
        assert fp == {"PER": 0, "ORG": 1, "LOC": 1}
        assert fn == {"PER": 1, "ORG": 1, "LOC": 0}

    def test_micro_sum_matches_compute_prf(self):
        # strict_label_counts must be consistent with compute_prf's own
        # micro sum -- both are derived from extract_tp_actual_correct, so
        # a per-label report row sum must equal the summary line's numbers.
        from gliner.evaluation.evaluator import BaseEvaluator

        y_true = [[["PER", (0, 0)], ["ORG", (5, 7)]]]
        y_pred = [[["PER", (0, 0)], ["ORG", (6, 8)]]]

        micro = BaseEvaluator.compute_prf(y_true, y_pred)
        tp, fp, fn = strict_label_counts(y_true, y_pred)

        total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
        expected_precision = total_tp / (total_tp + total_fp)
        expected_recall = total_tp / (total_tp + total_fn)
        assert expected_precision == micro["precision"]
        assert expected_recall == micro["recall"]


class TestBuildClassificationReport:
    def test_includes_per_label_rows_and_averages(self):
        tp = {"PER": 1, "ORG": 0}
        fp = {"PER": 0, "ORG": 1}
        fn = {"PER": 1, "ORG": 0}

        report = build_classification_report(tp, fp, fn)

        assert "PER" in report
        assert "ORG" in report
        assert "micro avg" in report
        assert "macro avg" in report

    def test_empty_labels_reports_placeholder(self):
        assert build_classification_report({}, {}, {}) == "(no labels)"

    def test_macro_is_unweighted_mean_of_per_label_f1(self):
        # PER: p=1,r=1,f=1 (tp=1). ORG: p=0,r=0,f=0 (all zero) -> macro f = 0.5.
        tp = {"PER": 1, "ORG": 0}
        fp = {"PER": 0, "ORG": 0}
        fn = {"PER": 0, "ORG": 1}

        report = build_classification_report(tp, fp, fn)
        macro_line = next(line for line in report.splitlines() if line.startswith("macro avg"))
        assert "0.5000" in macro_line


class TestRelaxedLabelCounts:
    def test_matches_compute_relaxed_prf_micro_sum(self):
        all_true = [[["PER", (5, 5)], ["ORG", (10, 12)]]]
        all_pred = [[["PER", (5, 5)], ["ORG", (11, 13)]]]

        tp, fp, fn = relaxed_label_counts(all_true, all_pred)
        micro = compute_relaxed_prf(all_true, all_pred)

        total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
        assert total_tp / (total_tp + total_fp) == micro["precision"]
        assert total_tp / (total_tp + total_fn) == micro["recall"]
