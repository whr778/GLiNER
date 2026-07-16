"""Convert knowledgator/gliner-multilingual-synthetic to GLiNER NER-training JSONL.

Plain NER, no relations. The source ships each row already token-indexed --
unlike most datasets ported here, no offset mapping or surface search is
needed at all::

    tokenized_text: ["Der", "Film", "wurde", "in", "Los", "Angeles", "und",
                     "Santa", "Clarita", "gedreht", "."]
    ner:            [["7", "8", "\\"location\\""],
                     ["4", "5", "\\"location\\""]]

Indices are inclusive but string-typed, and labels are JSON-quoted
(``'"location"'``, not ``"location"``) -- both are unwrapped directly, no
substring/offset reconstruction required since the source is already the
gold token list.

Requires the optional ``data`` dependency group (``uv add --optional data
datasets``, or ``uv sync --extra data``).

Output (one record per source row)::

    {"tokenized_text": [...], "ner": [[4, 5, "location"], [7, 8, "location"]], "relations": []}

Usage::

    uv run python data/convert_gliner_multilingual.py --out data/gliner_multilingual.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import add_split_args, write_jsonl_split  # noqa: E402


def unwrap_label(raw: Any) -> Optional[str]:
    """Strip the JSON quoting around a label, e.g. '"location"' -> 'location'."""
    if not isinstance(raw, str):
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        decoded = raw.strip('"').strip()
    return decoded.strip() if isinstance(decoded, str) and decoded.strip() else None


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one multilingual row to a GLiNER NER-training record; None if unusable."""
    tokens = row.get("tokenized_text") or []
    ner_in = row.get("ner") or []
    if not tokens or not ner_in:
        return None
    num_tokens = len(tokens)

    ner: List[List[Any]] = []
    for span in ner_in:
        if not isinstance(span, (list, tuple)) or len(span) != 3:
            continue
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            continue
        label = unwrap_label(span[2])
        if label is None or not (0 <= start <= end < num_tokens):
            continue
        ner.append([start, end, label])

    if not ner:
        return None
    return {"tokenized_text": list(tokens), "ner": ner, "relations": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="knowledgator/gliner-multilingual-synthetic",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Streaming {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split, streaming=True)

    def _records():
        n = 0
        for row in ds:
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
