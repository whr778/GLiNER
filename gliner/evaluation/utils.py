import warnings
from typing import List, Tuple, Union, Literal
from collections import Counter, defaultdict

import numpy as np


class UndefinedMetricWarning(UserWarning):
    pass


def extract_tp_actual_correct(y_true, y_pred):
    elements_true = defaultdict(set)
    elements_pred = defaultdict(set)

    for type_name, el, idx in y_true:
        elements_true[type_name].add((el, idx))
    for type_name, el, idx in y_pred:
        elements_pred[type_name].add((el, idx))

    target_names = sorted(set(elements_true.keys()) | set(elements_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        elements_true_type = elements_true.get(type_name, set())
        elements_pred_type = elements_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(elements_true_type & elements_pred_type))
        pred_sum = np.append(pred_sum, len(elements_pred_type))
        true_sum = np.append(true_sum, len(elements_true_type))

    return pred_sum, tp_sum, true_sum, target_names


def _prf_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    metric: Literal["precision", "recall", "f-score"],
    modifier: str,
    average: str,
    warn_for: List[str],
    zero_division: Union[str, int] = "warn",
) -> np.ndarray:
    """Performs division and handles divide-by-zero with warnings."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(numerator, denominator)
        result[denominator == 0] = 0.0 if zero_division in ["warn", 0] else 1.0

    if denominator == 0 and zero_division == "warn" and metric in warn_for:
        msg_start = f"{metric.title()}"
        if "f-score" in warn_for:
            msg_start += " and F-score" if metric in warn_for else "F-score"
        msg_start += " are" if "f-score" in warn_for else " is"
        _warn_prf(
            average=average,
            modifier=modifier,
            msg_start=msg_start,
            result_size=len(result),
        )

    return result


def _warn_prf(average: str, modifier: str, msg_start: str, result_size: int):
    axis0, axis1 = ("label", "sample") if average == "samples" else ("sample", "label")
    if result_size == 1:
        msg = f"{msg_start} ill-defined and being set to 0.0 due to no {modifier} {axis0}."
    else:
        msg = f"{msg_start} ill-defined and being set to 0.0 in {axis1}s with no {modifier} {axis0}s."
    msg += " Use `zero_division` parameter to control this behavior."
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=3)


def flatten_for_eval(y_true, y_pred):
    all_true = []
    all_pred = []

    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        all_true.extend([[*t, i] for t in true])
        all_pred.extend([[*p, i] for p in pred])

    return all_true, all_pred


def _spans_overlap(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
    """True if every (start, end) pair in ``a`` overlaps the corresponding pair in ``b``.

    Both endpoints are inclusive word-token indices. ``a``/``b`` are 2-tuples
    for an entity span, or 4-tuples ``(h_start, h_end, t_start, t_end)`` for a
    relation -- in the latter case both the head and the tail span must
    overlap, matching the exact-match semantics they relax.
    """
    for i in range(0, len(a), 2):
        s1, e1 = a[i], a[i + 1]
        s2, e2 = b[i], b[i + 1]
        if not (s1 <= e2 and s2 <= e1):
            return False
    return True


def _match_relaxed_sample(true_items, pred_items) -> Tuple[Counter, Counter, Counter]:
    """Greedy two-pass match of one sample's ``[label, span]`` gold/pred lists.

    Pass 1 pairs exact-equal spans (so relaxed can never miss a match strict
    would find); pass 2 pairs remaining predictions to remaining gold spans
    by overlap, within the same label. Spans are deduped per label before
    matching so relaxed counts agree with strict's set-based counting.

    Returns per-label (tp, fp, fn) Counters for this sample.
    """
    true_by_label = defaultdict(list)
    pred_by_label = defaultdict(list)
    for label, span in true_items:
        true_by_label[label].append(span)
    for label, span in pred_items:
        pred_by_label[label].append(span)

    tp, fp, fn = Counter(), Counter(), Counter()
    for label in set(true_by_label) | set(pred_by_label):
        t_spans = sorted(set(true_by_label.get(label, [])))
        p_spans = sorted(set(pred_by_label.get(label, [])))
        t_matched = [False] * len(t_spans)
        p_matched = [False] * len(p_spans)

        for pi, p_span in enumerate(p_spans):
            for ti, t_span in enumerate(t_spans):
                if not t_matched[ti] and p_span == t_span:
                    t_matched[ti] = p_matched[pi] = True
                    break

        for pi, p_span in enumerate(p_spans):
            if p_matched[pi]:
                continue
            for ti, t_span in enumerate(t_spans):
                if not t_matched[ti] and _spans_overlap(p_span, t_span):
                    t_matched[ti] = p_matched[pi] = True
                    break

        matched = sum(p_matched)
        tp[label] = matched
        fp[label] = len(p_spans) - matched
        fn[label] = len(t_spans) - matched

    return tp, fp, fn


def relaxed_label_counts(all_true, all_outs) -> Tuple[Counter, Counter, Counter]:
    """Per-label TP/FP/FN Counters under relaxed (overlap) matching, summed over all samples."""
    total_tp, total_fp, total_fn = Counter(), Counter(), Counter()
    for true_items, pred_items in zip(all_true, all_outs):
        tp, fp, fn = _match_relaxed_sample(true_items, pred_items)
        total_tp.update(tp)
        total_fp.update(fp)
        total_fn.update(fn)
    return total_tp, total_fp, total_fn


def strict_label_counts(y_true, y_pred) -> Tuple[dict, dict, dict]:
    """Per-label TP/FP/FN dicts under strict (exact) matching.

    Reuses ``extract_tp_actual_correct``'s per-label arrays -- the same
    counts ``compute_prf``'s micro-average sums over -- so the per-label
    report is guaranteed consistent with the existing strict micro F1.
    """
    flat_true, flat_pred = flatten_for_eval(y_true, y_pred)
    pred_sum, tp_sum, true_sum, target_names = extract_tp_actual_correct(flat_true, flat_pred)
    tp = {name: int(t) for name, t in zip(target_names, tp_sum)}
    fp = {name: int(p - t) for name, p, t in zip(target_names, pred_sum, tp_sum)}
    fn = {name: int(tr - t) for name, tr, t in zip(target_names, true_sum, tp_sum)}
    return tp, fp, fn


def _pr_f1(tp_n: int, fp_n: int, fn_n: int) -> Tuple[float, float, float]:
    precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
    recall = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f_score


def build_classification_report(tp: dict, fp: dict, fn: dict) -> str:
    """Per-label precision/recall/F1/support table plus micro and macro avg rows."""
    labels = sorted(set(tp) | set(fp) | set(fn))
    if not labels:
        return "(no labels)"

    total_tp = sum(tp.get(label, 0) for label in labels)
    total_fp = sum(fp.get(label, 0) for label in labels)
    total_fn = sum(fn.get(label, 0) for label in labels)
    micro_p, micro_r, micro_f = _pr_f1(total_tp, total_fp, total_fn)

    rows = []
    macro_p_sum = macro_r_sum = macro_f_sum = 0.0
    for label in labels:
        p, r, f = _pr_f1(tp.get(label, 0), fp.get(label, 0), fn.get(label, 0))
        support = tp.get(label, 0) + fn.get(label, 0)
        rows.append((label, p, r, f, support))
        macro_p_sum += p
        macro_r_sum += r
        macro_f_sum += f
    n = len(labels)
    macro_p, macro_r, macro_f = macro_p_sum / n, macro_r_sum / n, macro_f_sum / n

    lines = [f"{'label':<40} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}", "-" * 82]
    for label, p, r, f, support in rows:
        display = label if len(label) <= 40 else label[:38] + ".."
        lines.append(f"{display:<40} {p:>10.4f} {r:>10.4f} {f:>10.4f} {support:>10d}")
    lines.append("-" * 82)
    overall_support = total_tp + total_fn
    lines.append(f"{'micro avg':<40} {micro_p:>10.4f} {micro_r:>10.4f} {micro_f:>10.4f} {overall_support:>10d}")
    lines.append(f"{'macro avg':<40} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f:>10.4f} {overall_support:>10d}")
    return "\n".join(lines)


def compute_relaxed_prf(all_true, all_outs) -> dict:
    """Micro precision/recall/F1 under relaxed (overlap) matching.

    ``all_true``/``all_outs`` are per-sample ``[[label, span], ...]`` lists,
    the same shape ``BaseEvaluator.transform_data()`` produces for
    ``compute_prf``. For the per-label/macro breakdown, see
    ``relaxed_label_counts`` + ``build_classification_report``.
    """
    total_tp, total_fp, total_fn = relaxed_label_counts(all_true, all_outs)
    tp_sum = sum(total_tp.values())
    pred_sum = tp_sum + sum(total_fp.values())
    true_sum = tp_sum + sum(total_fn.values())

    precision = tp_sum / pred_sum if pred_sum else 0.0
    recall = tp_sum / true_sum if true_sum else 0.0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f_score": f_score}
