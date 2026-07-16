"""Analyze a GLiNER-schema dataset and report NER, relation, and event metrics.

Reads one or more ``{"tokenized_text": [...], "ner": [...], "relations": [...]}``
JSONL/JSON-array files (as produced by ``data/convert_*.py``, or already-split
train/val/test files) and prints:

1. **NER metrics** (always): document length, entity counts/types/span widths,
   language mix.
2. **Relation metrics** (only if any record has a non-empty ``relations``
   list): relation counts/types, head->tail entity-type pairs, and
   head-tail token distance. The distance stat matters beyond curiosity: a
   relation whose head and tail sit farther apart than one sliding window
   can't be recovered by windowing regardless of ``window_stride`` (see
   ``gliner/data_processing/windowing.py``), so a long tail on this
   distribution is a direct signal about how much relation recall a
   ``max_len``/``window_stride`` choice can realistically reach.
3. **Event metrics** (same relation records, reinterpreted): derives a
   trigger-type vocabulary the same way ``data/_trigger_types.py`` does --
   an entity type is a trigger candidate if it ever appears as a relation
   head -- and reports it against the full entity-type vocabulary. A small,
   distinct trigger set is consistent with a real trigger/argument event
   schema; a trigger set covering every entity type is consistent with a
   flat relation schema instead. This is a heuristic lens on the data, not
   a determination that the dataset IS (or isn't) event-shaped -- printed
   as such, with the deciding evidence shown alongside it.

Usage::

    uv run python scripts/analyze_dataset.py data/wikievents.train.jsonl
    uv run python scripts/analyze_dataset.py data/docred.train.jsonl data/docred.val.jsonl
    uv run python scripts/analyze_dataset.py data/nuner.train.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gliner.training.data_utils import detect_language, load_multi_dataset  # noqa: E402


def _numeric_summary(values: List[float]) -> Dict[str, Any]:
    """min/max/mean/median for ``values``, or ``{"count": 0}`` if empty."""
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def compute_token_stats(records: List[Dict]) -> Dict[str, Any]:
    lengths = [len(r.get("tokenized_text") or []) for r in records]
    return _numeric_summary(lengths)


def compute_entity_stats(records: List[Dict]) -> Dict[str, Any]:
    counts_per_record: List[int] = []
    widths: List[int] = []
    type_counts: Counter = Counter()
    malformed = 0

    for rec in records:
        entities = rec.get("ner") or []
        counts_per_record.append(len(entities))
        for ent in entities:
            if len(ent) < 3:
                malformed += 1
                continue
            start, end, label = ent[0], ent[1], ent[2]
            type_counts[label] += 1
            widths.append(end - start + 1)

    return {
        "total_entities": sum(counts_per_record),
        "per_record": _numeric_summary(counts_per_record),
        "span_width_tokens": _numeric_summary(widths),
        "type_counts": type_counts,
        "distinct_types": len(type_counts),
        "malformed_entries": malformed,
    }


def compute_relation_stats(records: List[Dict]) -> Dict[str, Any]:
    counts_per_record: List[int] = []
    type_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    distances: List[int] = []
    self_relations = 0
    malformed = 0

    for rec in records:
        entities = rec.get("ner") or []
        relations = rec.get("relations") or []
        counts_per_record.append(len(relations))
        for rel in relations:
            if len(rel) < 3:
                malformed += 1
                continue
            head_idx, tail_idx, label = rel[0], rel[1], rel[2]
            type_counts[label] += 1
            if head_idx == tail_idx:
                self_relations += 1
            if 0 <= head_idx < len(entities) and 0 <= tail_idx < len(entities) and \
                    len(entities[head_idx]) >= 3 and len(entities[tail_idx]) >= 3:
                head_type = entities[head_idx][2]
                tail_type = entities[tail_idx][2]
                pair_counts[f"{head_type} -> {tail_type}"] += 1
                distances.append(abs(entities[head_idx][0] - entities[tail_idx][0]))
            else:
                malformed += 1

    return {
        "total_relations": sum(counts_per_record),
        "per_record": _numeric_summary(counts_per_record),
        "type_counts": type_counts,
        "distinct_types": len(type_counts),
        "head_tail_type_pairs": pair_counts,
        "head_tail_token_distance": _numeric_summary(distances),
        "self_relations": self_relations,
        "malformed_entries": malformed,
    }


def compute_event_view(records: List[Dict]) -> Dict[str, Any]:
    """Reinterpret ``relations`` as trigger->argument event roles.

    Mirrors ``data/_trigger_types.py``'s ``derive_trigger_types`` rule (a
    type is a trigger candidate if it ever heads a relation), reimplemented
    here with bounds-checking since this tool must tolerate imperfect
    externally-provided data rather than assume a converter's guarantees.
    """
    all_types: set = set()
    for rec in records:
        for ent in rec.get("ner") or []:
            if len(ent) >= 3:
                all_types.add(ent[2])

    trigger_types: set = set()
    for rec in records:
        entities = rec.get("ner") or []
        for rel in rec.get("relations") or []:
            if len(rel) < 3:
                continue
            head_idx = rel[0]
            if 0 <= head_idx < len(entities) and len(entities[head_idx]) >= 3:
                trigger_types.add(entities[head_idx][2])

    trigger_counts: List[int] = []
    argument_counts: List[int] = []
    role_counts: Counter = Counter()
    for rec in records:
        entities = rec.get("ner") or []
        n_triggers = sum(1 for e in entities if len(e) >= 3 and e[2] in trigger_types)
        trigger_counts.append(n_triggers)
        argument_counts.append(len(entities) - n_triggers)
        for rel in rec.get("relations") or []:
            if len(rel) >= 3:
                role_counts[rel[2]] += 1

    return {
        "trigger_types": sorted(trigger_types),
        "argument_only_types": sorted(all_types - trigger_types),
        "num_trigger_types": len(trigger_types),
        "num_all_entity_types": len(all_types),
        "triggers_per_record": _numeric_summary(trigger_counts),
        "arguments_per_record": _numeric_summary(argument_counts),
        "role_counts": role_counts,
        "looks_event_shaped": bool(trigger_types) and trigger_types != all_types,
    }


def compute_language_stats(records: List[Dict], sample_size: int) -> Optional[Dict[str, Any]]:
    if sample_size <= 0:
        return None
    sample = records[:sample_size]
    counts: Counter = Counter()
    for rec in sample:
        text = " ".join(rec.get("tokenized_text") or [])
        counts[detect_language(text)] += 1
    return {"sampled": len(sample), "counts": counts}


def analyze(records: List[Dict], lang_sample_size: int) -> Dict[str, Any]:
    has_relations = any(rec.get("relations") for rec in records)
    report: Dict[str, Any] = {
        "num_records": len(records),
        "tokens_per_record": compute_token_stats(records),
        "entities": compute_entity_stats(records),
        "language": compute_language_stats(records, lang_sample_size),
        "has_relations": has_relations,
    }
    if has_relations:
        report["relations"] = compute_relation_stats(records)
        report["events"] = compute_event_view(records)
    return report


def _format_summary(s: Dict[str, Any], unit: str = "") -> str:
    if s["count"] == 0:
        return "n/a (no data)"
    return f"min={s['min']}{unit} max={s['max']}{unit} mean={s['mean']:.1f}{unit} median={s['median']:.1f}{unit}"


def _format_top_counts(counter: Dict[str, int], total: int, top_n: int) -> str:
    if not counter:
        return "    (none)"
    items = Counter(counter).most_common(top_n)
    width = max((len(str(label)) for label, _ in items), default=0)
    lines = []
    for label, count in items:
        pct = 100 * count / total if total else 0
        lines.append(f"    {str(label):<{width}}  {count:>8}  ({pct:5.1f}%)")
    remaining = len(counter) - min(top_n, len(counter))
    if remaining > 0:
        lines.append(f"    ... and {remaining} more distinct label(s)")
    return "\n".join(lines)


def print_report(report: Dict[str, Any], sources: List[str], top_n: int) -> None:
    print(f"===== Dataset Analysis: {', '.join(sources)} =====")
    print(f"Records: {report['num_records']}\n")

    print("--- NER ---")
    print(f"Tokens per record: {_format_summary(report['tokens_per_record'])}")
    ent = report["entities"]
    print(f"Total entities: {ent['total_entities']}")
    print(f"Entities per record: {_format_summary(ent['per_record'])}")
    print(f"Entity span width (tokens): {_format_summary(ent['span_width_tokens'])}")
    print(f"Distinct entity types: {ent['distinct_types']}")
    print("Top entity types:")
    print(_format_top_counts(ent["type_counts"], ent["total_entities"], top_n))
    if ent["malformed_entries"]:
        print(f"Malformed entity entries (skipped): {ent['malformed_entries']}")

    lang = report["language"]
    if lang is not None:
        print(f"\nLanguage mix (sampled {lang['sampled']} record(s)):")
        print(_format_top_counts(lang["counts"], lang["sampled"], top_n))

    if not report["has_relations"]:
        print("\n--- Relations & Events ---")
        print("No relations found in this dataset (every record's 'relations' "
              "list is empty) -- this looks like a plain NER dataset.")
        return

    rel = report["relations"]
    print("\n--- Relations ---")
    print(f"Total relations: {rel['total_relations']}")
    print(f"Relations per record: {_format_summary(rel['per_record'])}")
    print(f"Distinct relation types: {rel['distinct_types']}")
    print("Top relation types:")
    print(_format_top_counts(rel["type_counts"], rel["total_relations"], top_n))
    print(f"Head->Tail entity-type pairs (top {top_n}):")
    print(_format_top_counts(rel["head_tail_type_pairs"], rel["total_relations"], top_n))
    print(f"Head-tail token distance: {_format_summary(rel['head_tail_token_distance'])}")
    print(f"Self-relations (head==tail index): {rel['self_relations']}")
    if rel["malformed_entries"]:
        print(f"Malformed relation entries (skipped): {rel['malformed_entries']}")

    ev = report["events"]
    print("\n--- Events (heuristic view) ---")
    print("NOTE: reinterprets the same relations as trigger->argument event")
    print("roles, using the entity-type-appears-as-a-relation-head rule from")
    print("data/_trigger_types.py. This is a lens on the data, not a")
    print("determination that this IS an event dataset.")
    print(f"Derived trigger types: {ev['num_trigger_types']} of {ev['num_all_entity_types']} total entity types")
    if ev["looks_event_shaped"]:
        print("  -> Trigger vocabulary is a proper subset of all entity types --")
        print("     consistent with (but not proof of) a trigger/argument event schema.")
    else:
        print("  -> Every entity type appears as a relation head (or no relations")
        print("     exist) -- this looks like a flat relation schema, not a")
        print("     trigger/argument one.")
    print(f"Triggers per record: {_format_summary(ev['triggers_per_record'])}")
    print(f"Arguments per record: {_format_summary(ev['arguments_per_record'])}")
    print("Top event roles:")
    print(_format_top_counts(ev["role_counts"], sum(ev["role_counts"].values()), top_n))


def _counters_to_dicts(obj: Any) -> Any:
    """Recursively convert Counter values to plain, count-sorted dicts for JSON output."""
    if isinstance(obj, Counter):
        return dict(obj.most_common())
    if isinstance(obj, dict):
        return {k: _counters_to_dicts(v) for k, v in obj.items()}
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", nargs="+", help="One or more GLiNER-schema JSONL/JSON-array files to analyze.")
    parser.add_argument("--top-n", type=int, default=25,
                        help="Max distinct labels/pairs to print per distribution (default: 25).")
    parser.add_argument("--lang-sample-size", type=int, default=300,
                        help="Number of records to sample for language detection (default: 300).")
    parser.add_argument("--no-lang-detect", action="store_true", help="Skip language detection entirely.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON instead of formatted text.")
    args = parser.parse_args()

    records = load_multi_dataset(list(args.input))
    if not records:
        print(f"No records loaded from: {args.input}", file=sys.stderr)
        return 1

    lang_sample_size = 0 if args.no_lang_detect else args.lang_sample_size
    report = analyze(records, lang_sample_size)

    if args.json:
        report["sources"] = args.input
        print(json.dumps(_counters_to_dicts(report), indent=2, ensure_ascii=False))
    else:
        print_report(report, args.input, args.top_n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
