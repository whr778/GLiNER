from gliner.decoding.decoder import Span
from gliner.evaluation.evaluator import BaseNEREvaluator, BaseRelexEvaluator


class TestBaseRelexEvaluatorGetPredictions:
    def test_handles_span_objects(self):
        pred_ents = [
            Span(start=0, end=0, entity_type="PER", score=0.9),
            Span(start=3, end=3, entity_type="ORG", score=0.8),
        ]
        pred_rels = [(0, "WORKS_FOR", 1)]

        evaluator = BaseRelexEvaluator([], [])
        result = evaluator.get_predictions(pred_ents, pred_rels)

        assert result == [["WORKS_FOR", (0, 0, 3, 3)]]

    def test_handles_plain_tuples(self):
        pred_ents = [(0, 0, "PER"), (3, 3, "ORG")]
        pred_rels = [(0, "WORKS_FOR", 1)]

        evaluator = BaseRelexEvaluator([], [])
        result = evaluator.get_predictions(pred_ents, pred_rels)

        assert result == [["WORKS_FOR", (0, 0, 3, 3)]]


class TestBaseRelexEvaluatorEndToEnd:
    def test_exact_match_scores_perfect_f1(self):
        """A predicted (Span-shaped) entity/relation set identical to gold
        must register as a true positive -- i.e. gold's (start, end) tuples
        and predicted Span.start/.end, compared through get_predictions'
        h_bounds + t_bounds layout, must actually compare equal. This is
        the check test_handles_span_objects can't do on its own: that one
        only proves get_predictions doesn't crash on Span input, not that
        its output matches gold's format closely enough to ever produce
        a true positive.
        """
        gold_ents = [(0, 0, "PER"), (3, 3, "ORG")]
        gold_rels = [(0, 1, "WORKS_FOR")]

        pred_ents = [
            Span(start=0, end=0, entity_type="PER", score=0.9),
            Span(start=3, end=3, entity_type="ORG", score=0.8),
        ]
        pred_rels = [(0, "WORKS_FOR", 1, 0.95)]

        evaluator = BaseRelexEvaluator([(gold_ents, gold_rels)], [(pred_ents, pred_rels)])
        output_str, f1 = evaluator.evaluate()

        assert f1 == 1.0


class TestStrictRelaxedReport:
    """``evaluate()`` must keep returning (output_str, strict_f1) unchanged in
    shape -- evaluate_and_extract_f1's NER-vs-RelEx dispatch relies on
    output_str never itself being a tuple -- while output_str now surfaces
    both strict and relaxed numbers.
    """

    def test_ner_evaluate_returns_str_and_strict_f1(self):
        evaluator = BaseNEREvaluator([[(5, 7, "PER")]], [[(6, 8, "PER")]])
        output_str, f1 = evaluator.evaluate()

        assert isinstance(output_str, str)
        assert f1 == 0.0  # strict: no exact match
        assert "Strict" in output_str
        assert "Relaxed" in output_str
        assert "F1: 100.00%" in output_str  # relaxed recovers the overlap

    def test_relex_evaluate_return_shape_unchanged(self):
        gold_ents = [(0, 0, "PER"), (3, 3, "ORG")]
        gold_rels = [(0, 1, "WORKS_FOR")]
        pred_ents = [(0, 0, "PER"), (3, 3, "ORG")]
        pred_rels = [(0, "WORKS_FOR", 1)]

        evaluator = BaseRelexEvaluator([(gold_ents, gold_rels)], [(pred_ents, pred_rels)])
        output_str, f1 = evaluator.evaluate()

        assert isinstance(output_str, str)
        assert not isinstance(output_str, tuple)
        assert f1 == 1.0

    def test_output_includes_per_label_classification_report(self):
        evaluator = BaseNEREvaluator(
            [[(0, 0, "PER"), (5, 7, "ORG")]], [[(0, 0, "PER"), (6, 8, "ORG")]]
        )
        output_str, _ = evaluator.evaluate()

        assert "Strict classification report:" in output_str
        assert "Relaxed classification report:" in output_str
        assert "PER" in output_str
        assert "ORG" in output_str
        assert "micro avg" in output_str
        assert "macro avg" in output_str
