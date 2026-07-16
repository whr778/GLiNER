"""Convert numind/NuNER to GLiNER NER-training JSONL.

Plain NER, no relations. The NuNER ``full`` split stores each row as::

    input:  "<text>"
    output: "['<surface> <> <type> <> <description>', ...]"  (Python list literal)

(the ``entity`` split omits ``<description>``). The source is surface-only
-- no offsets at all -- so this word-tokenizes ``input`` and locates each
surface via ``data/_charspan.find_surface_span`` (first verbatim occurrence;
verified this doesn't silently drop everything by spot-checking real rows
during development -- most surfaces are unique per document). A surface
that isn't found verbatim is dropped, same as GLiNER2's own converter.
Entity descriptions have no home in GLiNER's schema and are dropped.

Requires the optional ``data`` dependency group.

Usage::

    uv run python data/convert_nuner.py --split full --out data/nuner.train.jsonl
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _charspan import find_surface_span, tokenize_with_offsets  # noqa: E402
from _jsonl import add_split_args, write_jsonl_split  # noqa: E402


def parse_items(output_field: str) -> List[List[str]]:
    """Parse the NuNER `output` string into a list of [surface, type, desc?] lists."""
    raw = ast.literal_eval(output_field)
    parsed = []
    for item in raw:
        parts = item.split(" <> ", 2)
        if len(parts) >= 2:
            parsed.append(parts)
    return parsed


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one NuNER row to a GLiNER NER-training record; None if unusable."""
    text = row.get("input")
    output_field = row.get("output")
    if not isinstance(text, str) or not text or not isinstance(output_field, str):
        return None
    try:
        items = parse_items(output_field)
    except (SyntaxError, ValueError):
        return None

    tokens, _ = tokenize_with_offsets(text)
    if not tokens:
        return None

    ner: List[List[Any]] = []
    seen: set = set()
    for parts in items:
        surface, etype = parts[0], parts[1]
        span = find_surface_span(tokens, surface)
        if span is None:
            continue
        key = (span[0], span[1], etype)
        if key in seen:
            continue
        seen.add(key)
        ner.append([span[0], span[1], etype])

    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="full", choices=["full", "entity"], help="NuNER split to convert.")
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="numind/NuNER", help="HuggingFace dataset repo.")
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
