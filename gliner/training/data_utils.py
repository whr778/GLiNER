"""Multi-dataset loading, best-model tracking, and blind-test-by-language
helpers shared between ``train.py`` and ``scripts/custom_train.py``.
"""

import argparse
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..data_processing.windowing import window_training_record
from .model_card import write_model_card

PathOrPaths = Union[str, List[str], None]


def flatten_namespace(cfg: argparse.Namespace) -> argparse.Namespace:
    """Merge a nested ``model``/``data``/``training`` config namespace into one flat namespace.

    ``scripts/custom_train.py`` predates the ``model:``/``data:``/``training:``
    section split in ``configs/*.yaml`` and reads every field as a top-level
    ``self.config.X`` attribute; this restores that flat shape from the
    current nested config files without touching their section layout.
    """
    merged: Dict[str, Any] = {}
    for section in ("model", "data", "training"):
        merged.update(vars(getattr(cfg, section)))
    return argparse.Namespace(**merged)


def _read_one(path: str) -> List[Dict]:
    """Read a single dataset file, auto-detecting JSON-array vs JSONL."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_multi_dataset(paths: PathOrPaths, seed: int = 42) -> Optional[List[Dict]]:
    """Load, concatenate, and shuffle one or more dataset files.

    ``paths`` may be a single path, a list of paths, or None/""/"none" (all
    of which return None, so an unset split can be skipped by the caller).
    Each file may be a single JSON array or JSONL (one record per line).
    Shuffling uses a local Random instance seeded by ``seed`` so it does not
    disturb the global ``random`` module state, and is reproducible.
    """
    if paths is None:
        return None
    if isinstance(paths, str):
        if paths.strip().lower() in ("", "none"):
            return None
        paths = [paths]

    records: List[Dict] = []
    for path in paths:
        records.extend(_read_one(path))

    random.Random(seed).shuffle(records)
    return records


def window_records(records: List[Dict], max_len: int, stride: Optional[int] = None) -> List[Dict]:
    """Expand records longer than ``max_len`` into overlapping training windows.

    Thin wrapper around ``window_training_record`` for the train split --
    without this, records past ``max_len`` are silently truncated (in both
    training and eval) and everything past the cutoff is invisible to the
    model. Records that already fit are returned unchanged. Val/test splits
    don't need this: eval-time windowing happens transparently inside
    ``model.evaluate()``.
    """
    windowed: List[Dict] = []
    for record in records:
        windowed.extend(window_training_record(record, max_len=max_len, stride=stride))
    return windowed


def _extract_entity_types(records: List[Dict]) -> List[str]:
    """Collect the sorted set of entity type labels present across ``records``."""
    types = {ent[2] for rec in records for ent in rec.get("ner", [])}
    return sorted(types)


def warn_if_max_types_truncates(records: Optional[List[Dict]], max_types: Optional[int]) -> None:
    """Warn when the training data has more distinct entity/relation types than ``max_types``.

    Per-example type sampling (``batch_generate_class_mappings``) pools each
    record's gold types with in-data negatives, shuffles, and keeps the first
    ``max_types`` -- so once that pool exceeds ``max_types`` gold types can be
    randomly dropped from an example's label set. Negatives are drawn only from
    in-data types, so ``total distinct types <= max_types`` guarantees no
    truncation and ``> max_types`` is the exact condition under which it can
    occur. For event models, dropping a trigger type removes that event's
    supervision, so this is worth surfacing once at startup.
    """
    if not records or not max_types:
        return
    max_types = int(max_types)
    for kind, key in (("entity", "ner"), ("relation", "relations")):
        types = {ann[-1] for rec in records for ann in (rec.get(key) or [])}
        total = len(types)
        if total > max_types:
            warnings.warn(
                f"max_types={max_types} but the training data has {total} distinct {kind} "
                f"types: per-example sampling can drop up to {total - max_types} of them "
                f"(gold types included) from each example's label set. Raise max_types to "
                f">= {total} to guarantee no gold {kind} type is dropped.",
                stacklevel=2,
            )


def evaluate_and_extract_f1(
    model, records: List[Dict], *, rel_metric_weight: float = 0.5, **evaluate_kwargs
) -> Tuple[float, Any]:
    """Run ``model.evaluate(records, **evaluate_kwargs)`` and extract a single F1.

    Normalizes the two return shapes GLiNER's ``evaluate()`` methods use:
    ``(output, f1)`` for NER-only models, or
    ``((ner_output, ner_f1), (rel_output, rel_f1))`` for RelEx models, in
    which case the tracked F1 is ``(1 - rel_metric_weight) * ner_f1 +
    rel_metric_weight * rel_f1``. The default 0.5 is the plain average; raising
    it biases checkpoint selection / early stopping toward the relation (event
    role) side, which matters for event models where the much larger entity-NER
    F1 otherwise dominates and plateaus early while roles are still learning.
    ``rel_metric_weight`` is ignored for NER-only models.

    Passes ``entity_types`` explicitly (derived from all of ``records``, not
    just ``evaluate()``'s default per-mini-batch inference) so a class the
    model predicts in one batch can still be resolved even if that specific
    batch's own records don't happen to include it.
    """
    if "entity_types" not in evaluate_kwargs:
        entity_types = _extract_entity_types(records)
        if entity_types:
            evaluate_kwargs["entity_types"] = entity_types
    result = model.evaluate(records, **evaluate_kwargs)
    first, second = result
    if isinstance(first, tuple) and isinstance(second, tuple):
        (ner_output, ner_f1), (rel_output, rel_f1) = first, second
        f1 = (1 - rel_metric_weight) * ner_f1 + rel_metric_weight * rel_f1
        return f1, result
    output, f1 = result
    return f1, output


DEFAULT_THRESHOLD_GRID: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)


def sweep_thresholds(
    model,
    eval_records: List[Dict],
    thresholds: Iterable[float] = DEFAULT_THRESHOLD_GRID,
    **evaluate_kwargs,
) -> Tuple[float, float, Dict[float, float]]:
    """Score ``eval_records`` at each candidate threshold; return the best.

    "Best" is the threshold maximizing ``evaluate_and_extract_f1``'s F1
    (already a single scalar -- the unweighted NER/relation average for
    RelEx models, so no further cross-category weighting is needed here).

    Each candidate re-runs ``model.evaluate()`` over ``eval_records`` (not
    threshold-cached), so keep the grid small -- this is a one-time
    post-training calibration step, not a fine search. It raises the
    *measured* F1 by finding a better decision threshold; it does not
    retrain the model. ``relation_threshold``/``adjacency_threshold`` are
    not swept independently -- they default-couple to ``threshold`` inside
    ``evaluate()`` unless ``evaluate_kwargs`` overrides them explicitly.

    Args:
        model: A loaded GLiNER model.
        eval_records: Held-out validation records (never the test split).
        thresholds: Candidate threshold values.
        **evaluate_kwargs: Forwarded to ``evaluate_and_extract_f1`` for every
            candidate (e.g. ``window_stride``); ``threshold`` is overridden
            per candidate and should not be passed here.

    Returns:
        ``(best_threshold, best_threshold's F1, {threshold: F1} for every
        candidate)``.
    """
    results: Dict[float, float] = {}
    for t in thresholds:
        f1, _ = evaluate_and_extract_f1(model, eval_records, threshold=t, **evaluate_kwargs)
        results[t] = f1

    best_threshold = max(results, key=results.get)
    return best_threshold, results[best_threshold], results


class BestModelTracker:
    """Tracks the best F1 seen so far and saves the model when it improves.

    ``card_data_stats`` (from ``summarize_training_data``) is computed once at
    training start and reused to write a model card next to each best save.
    """

    def __init__(self, card_data_stats: Optional[Dict[str, Any]] = None):
        self.best_f1: Optional[float] = None
        self.card_data_stats = card_data_stats

    def maybe_save(self, f1: float, model, output_dir: Union[str, Path]) -> bool:
        if self.best_f1 is not None and f1 <= self.best_f1:
            return False
        self.best_f1 = f1
        best_dir = Path(output_dir) / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(best_dir))
        write_model_card(best_dir, model.config, data_stats=self.card_data_stats, best_f1=f1)
        return True


def detect_language(text: str) -> str:
    """Return an ISO 639-1 language code for ``text``, or "und" if undetermined."""
    text = text.strip()
    if not text:
        return "und"
    from langdetect import LangDetectException, detect

    try:
        return detect(text)
    except LangDetectException:
        return "und"


def annotate_languages(records: List[Dict]) -> None:
    """Stamp each record dict with '_lang' in-place, detected from tokenized_text."""
    for rec in records:
        text = " ".join(rec.get("tokenized_text") or [])
        rec["_lang"] = detect_language(text)


def _label_block(label: str, text: str) -> str:
    """Prefix ``text`` with "label: " and indent any further lines to align under it.

    ``text`` may itself be multi-line (e.g. separate strict/relaxed report
    lines from ``BaseEvaluator.evaluate()``), which would otherwise dangle
    at column 0 under the label.
    """
    prefix = f"{label}: "
    indent = " " * len(prefix)
    return prefix + f"\n{indent}".join(str(text).strip().splitlines())


def format_evaluate_output(output: Any, event_mode: bool = False) -> str:
    """Label ``model.evaluate()``'s raw return value by task instead of printing it raw.

    NER-only models return a single (now strict+relaxed, two-line) P/R/F1
    report string, returned as-is. RelEx/event models return
    ``((entity_output, entity_f1), (pair_output, pair_f1))`` -- unlabeled,
    that pair reads as two anonymous report blocks with no indication of
    which is which, or whether the second is a relation or an event role.
    Labels the entity side "Trigger/entity" and the pair side "Event role"
    when ``event_mode`` is set (matching ``config.event_mode``), else
    "Entity" / "Relation".
    """
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[0], tuple):
        (entity_output, _), (pair_output, _) = output
        entity_label = "Trigger/entity" if event_mode else "Entity"
        pair_label = "Event role" if event_mode else "Relation"
        return f"{_label_block(entity_label, entity_output)}\n{_label_block(pair_label, pair_output)}"
    return str(output).strip()


def print_blind_test(name: str, f1: float, output: Any, event_mode: bool = False) -> None:
    print(f"\n===== Blind test: {name} =====")
    print(f"F1: {f1:.4f}")
    print(format_evaluate_output(output, event_mode=event_mode))


def build_test_metrics(f1: float, output: Any, event_mode: bool = False) -> Dict[str, Any]:
    """Normalize a blind-test ``(f1, output)`` into the model card's test-metrics dict.

    ``report`` is the same labeled strict/relaxed + per-label text
    ``print_blind_test`` shows, embedded verbatim in the card's evaluation
    section (no fragile re-parsing of the evaluator's report string). For
    RelEx/event models the ``((entity_output, entity_f1), (pair_output,
    pair_f1))`` shape also exposes the two head F1s separately as headline
    numbers.
    """
    metrics: Dict[str, Any] = {"overall_f1": f1, "report": format_evaluate_output(output, event_mode=event_mode)}
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[0], tuple):
        (_, entity_f1), (_, pair_f1) = output
        metrics["entity_label"] = "Trigger/entity" if event_mode else "Entity"
        metrics["pair_label"] = "Event role" if event_mode else "Relation"
        metrics["entity_f1"] = entity_f1
        metrics["pair_f1"] = pair_f1
    return metrics


def blind_test_by_language(model, test_records: List[Dict], evaluate_kwargs: Dict) -> Tuple[float, Any]:
    """Run the blind test per language, then once over all data combined.

    ``entity_types`` is derived once from the full ``test_records`` (not
    per-language subsets) so a type that's absent from one language's
    subset but present in another can still be resolved when the model
    predicts it. See ``evaluate_and_extract_f1``.

    Returns the (f1, output) of the all-combined pass.
    """
    evaluate_kwargs = dict(evaluate_kwargs)
    if "entity_types" not in evaluate_kwargs:
        entity_types = _extract_entity_types(test_records)
        if entity_types:
            evaluate_kwargs["entity_types"] = entity_types

    event_mode = bool(getattr(getattr(model, "config", None), "event_mode", False))

    annotate_languages(test_records)

    by_lang: Dict[str, List[Dict]] = defaultdict(list)
    for rec in test_records:
        by_lang[rec.get("_lang", "und")].append(rec)

    for lang in sorted(by_lang):
        subset = by_lang[lang]
        print(f"\n[blind test] Processing language: {lang} ({len(subset)} samples)")
        f1, output = evaluate_and_extract_f1(model, subset, **evaluate_kwargs)
        print_blind_test(lang, f1, output, event_mode=event_mode)

    print(f"\n[blind test] All languages combined ({len(test_records)} samples)")
    all_f1, all_output = evaluate_and_extract_f1(model, test_records, **evaluate_kwargs)
    print_blind_test("all", all_f1, all_output, event_mode=event_mode)

    return all_f1, all_output
