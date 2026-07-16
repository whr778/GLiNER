"""Convert knowledgator/events_classification_biotech to GLiNER event-training JSONL.

Despite the name, this dataset is pure multi-label text classification --
each article is tagged with 1-5 of 29 event-type categories (e.g. "funding
round", "m&a", "executive statement"). There are no trigger spans, no
argument roles, and -- unlike DocEE, which at least has a document-level
label to anchor a single synthetic trigger to -- no natural single anchor
point at all, since a document can carry several labels simultaneously.

This does not fit the trigger/argument event architecture at all: there is
no span-level signal anywhere in the source. To still route this data
through the event head (rather than dropping the dataset), one synthetic
trigger token ``[<label>]`` is prepended per true label -- so a
2-label document gets 2 synthetic trigger tokens, each its own event with
no arguments (``relations`` is always empty; there's nothing to link a
trigger to). This is a genuinely degenerate training signal -- the model
only ever learns to recognize a literal bracketed placeholder it was told
is a trigger, not any real-world event mention -- included because it was
explicitly requested, not because it is expected to generalize.

Source: reads GLiNER2's already-converted classification JSONL directly
(``{"input": ..., "output": {"classifications": [{"true_label": [...]}]}}``)
rather than re-downloading the raw CSV, since GLiNER's environment doesn't
carry the pandas/huggingface_hub dependencies GLiNER2's original converter
needs and a verified local conversion already exists.

Output (one record per source document)::

    {"tokenized_text": ["[m&a]", "[funding round]", "SPECIAL", "REPORT", ...],
     "ner": [[0, 0, "m&a"], [1, 1, "funding round"]],
     "relations": []}

Usage::

    uv run python data/convert_events_biotech.py \\
        --input /path/to/events_biotech.train.jsonl \\
        --out data/events_biotech.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _charspan import tokenize_with_offsets  # noqa: E402
from _jsonl import add_split_args, iter_jsonl, write_jsonl_split  # noqa: E402


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one classification-format record to a GLiNER event-training record."""
    text = row.get("input")
    output = row.get("output") or {}
    classifications = output.get("classifications") if isinstance(output, dict) else None
    if not isinstance(text, str) or not text.strip() or not isinstance(classifications, list):
        return None

    true_labels: List[str] = []
    for c in classifications:
        if not isinstance(c, dict):
            continue
        for label in c.get("true_label") or []:
            if isinstance(label, str) and label.strip() and label.strip() not in true_labels:
                true_labels.append(label.strip())
    if not true_labels:
        return None

    tokens, _ = tokenize_with_offsets(text)
    if not tokens:
        return None

    trigger_tokens = [f"[{label}]" for label in true_labels]
    ner = [[i, i, label] for i, label in enumerate(true_labels)]
    return {"tokenized_text": trigger_tokens + tokens, "ner": ner, "relations": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to GLiNER2's converted events_biotech "
                             "classification JSONL (input/output/classifications shape).")
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    add_split_args(parser)
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")

    def _records():
        n = 0
        with args.input.open(encoding="utf-8") as fh:
            for row in iter_jsonl(fh):
                rec = convert_row(row)
                if rec is None:
                    continue
                yield rec
                n += 1
                if args.max_records >= 0 and n >= args.max_records:
                    break

    counts = write_jsonl_split(_records(), args.out, ratios=args.split_ratios, seed=args.split_seed)
    print(f"wrote train={counts['train']} val={counts['val']} test={counts['test']} "
          f"-> {args.out} (.train/.val/.test.jsonl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
