"""Convert stockmark/ner-wikipedia-dataset (Japanese NER) to GLiNER NER-training JSONL.

Plain NER, no relations. Each row is ``{text, entities: [{name, span, type}]}``
with ``span: [start, end]`` char offsets (end-exclusive -- verified against
real data: ``text[start:end] == name``). License: CC-BY-SA-3.0.

Japanese has no whitespace word boundaries, so -- matching
convert_klue_ner.py / convert_cmnee.py -- ``tokenized_text`` is
``list(text)`` (one character per token); a char span is then already a
token-index span, no offset mapping needed.

Usage::

    uv run python data/convert_stockmark_ner.py --out data/stockmark_ner.train.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import add_split_args, write_jsonl_split  # noqa: E402


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one stockmark NER row to a GLiNER NER-training record; None if unusable."""
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    tokens = list(text)
    num_tokens = len(tokens)

    ner: List[List[Any]] = []
    for ent in row.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        etype, span = ent.get("type"), ent.get("span")
        if not isinstance(etype, str) or not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        etype = etype.strip()
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            continue
        if not etype or not (0 <= start < end <= num_tokens):
            continue
        ner.append([start, end - 1, etype])

    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="stockmark/ner-wikipedia-dataset", help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split)

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
