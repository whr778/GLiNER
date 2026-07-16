"""Convert paraloq/json_data_extraction to GLiNER NER-training JSONL (schema-driven extraction).

Plain NER, no relations. Each source row is a ``(text, schema, item)``
triple: a natural-language document, a JSON Schema, and the extracted
structured object conforming to it. This walks the extracted ``item``
recursively -- dict keys become the label for their values, lists inherit
the parent label, leaf scalars are located in the text -- mirroring
GLiNER2's converter's recursive walk, but locating each leaf value's real
token span via ``data/_charspan.find_surface_span`` instead of only
recording surface strings (the source is surface-only, no offsets, so this
is the ceiling of achievable fidelity here).

License: Apache-2.0. Requires the optional ``data`` dependency group.

Usage::

    uv run python data/convert_paraloq_json.py --out data/paraloq_json.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _charspan import find_surface_span, tokenize_with_offsets  # noqa: E402
from _jsonl import add_split_args, write_jsonl_split  # noqa: E402

# Drop surfaces longer than this many whitespace tokens (long free-text
# fields blow past the span head's max_width and can never be matched).
MAX_SURFACE_WORDS = 50


def _coerce_surface(value: Any) -> Optional[str]:
    """Return a non-empty string surface for primitive scalars, else None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return None


def _walk(value: Any, label: Optional[str], tokens: List[str], ner: List[List[Any]], seen: set) -> None:
    """Recurse into the extracted object; map leaf scalars to token-span ner entries."""
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and k.strip():
                _walk(v, k.strip(), tokens, ner, seen)
    elif isinstance(value, list):
        for item in value:
            _walk(item, label, tokens, ner, seen)
    elif label is not None:
        surface = _coerce_surface(value)
        if surface is None or len(surface.split()) > MAX_SURFACE_WORDS:
            return
        span = find_surface_span(tokens, surface)
        if span is None:
            return
        key = (span[0], span[1], label)
        if key in seen:
            return
        seen.add(key)
        ner.append([span[0], span[1], label])


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one paraloq row to a GLiNER NER-training record; None if unusable."""
    text = row.get("text")
    item = row.get("item")
    if not isinstance(text, str) or not text.strip() or item is None:
        return None
    if isinstance(item, str):
        try:
            item = json.loads(item)
        except json.JSONDecodeError:
            return None
    if not isinstance(item, (dict, list)):
        return None

    tokens, _ = tokenize_with_offsets(text)
    if not tokens:
        return None

    ner: List[List[Any]] = []
    _walk(item, None, tokens, ner, set())
    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="paraloq/json_data_extraction", help="HuggingFace dataset repo.")
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
