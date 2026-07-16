"""Convert knowledgator/biomed_NER to GLiNER NER-training JSONL.

Plain NER, no relations. Source rows are flat ``{text, entities}`` with
character offsets (end-exclusive)::

    {"text": "Weed seed inactivation in soil mesocosms ...",
     "entities": [{"start": 0, "end": 4, "class": "ORGANISM"}, ...]}

Word-tokenizes ``text`` and maps each char span to a token span via
``data/_charspan.py`` rather than slicing+substring-searching -- offsets are
given directly so there's no ambiguity to resolve, just a mapping step.

Light cleanup carried over from GLiNER2's converter: strips whitespace from
class names (source has trailing-space duplicates like ``"ORGANISMS "`` vs
``"ORGANISMS"``) and skips the ``"Unlabelled"`` class (no training signal).

Requires the optional ``data`` dependency group.

Usage::

    uv run python data/convert_biomed_ner.py --out data/biomed_ner.train.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _charspan import char_span_to_token_span, tokenize_with_offsets  # noqa: E402
from _jsonl import add_split_args, write_jsonl_split  # noqa: E402

SKIP_CLASSES = {"Unlabelled"}


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one biomed_NER row to a GLiNER NER-training record; None if unusable."""
    text = row.get("text")
    spans_in = row.get("entities") or []
    if not isinstance(text, str) or not text.strip() or not spans_in:
        return None

    tokens, offsets = tokenize_with_offsets(text)
    if not tokens:
        return None
    num_tokens = len(tokens)

    ner: List[List[Any]] = []
    for span in spans_in:
        if not isinstance(span, dict):
            continue
        cls = span.get("class")
        if not isinstance(cls, str):
            continue
        cls = cls.strip()
        if not cls or cls in SKIP_CLASSES:
            continue
        try:
            start, end = int(span["start"]), int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end > len(text) or end <= start:
            continue
        tok_span = char_span_to_token_span(offsets, start, end)
        if tok_span is None or not (0 <= tok_span[1] < num_tokens):
            continue
        ner.append([tok_span[0], tok_span[1], cls])

    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="knowledgator/biomed_NER", help="HuggingFace dataset repo.")
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
