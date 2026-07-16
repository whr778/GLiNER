"""Convert knowledgator/text2json-training-data to GLiNER NER-training JSONL.

Plain NER, no relations. The source repo holds several JSONL files with
inconsistent schemas, so this downloads one named file directly via
``huggingface_hub`` (default ``augmented_train.jsonl``, clean ``{text,
extracted}`` schema) rather than using ``datasets.load_dataset``.

``extracted`` comes in two shapes handled here (plus a long tail of deeply
nested objects skipped, matching GLiNER2's converter):

1. **Entity-list**: ``{"entities": [{"entity": ..., "type": ..., "description": ...}]}``
2. **Flat key->value**: ``{"tournament_code": "ROL-2024", "winner": "Sofia Petrova", ...}``

Source is surface-only (no offsets), so this word-tokenizes ``text`` and
locates each value via ``data/_charspan.find_surface_span`` instead of a
bare ``in text`` substring check -- the ceiling of achievable fidelity given
the source. Descriptions have no home in GLiNER's schema and are dropped.
Surfaces longer than ``MAX_SURFACE_WORDS`` are dropped -- text2json's flat
shape happily promotes long free-text fields (plot summaries, abstracts)
that blow past the span head's max_width and can never be matched anyway.

The optional ``data`` dependency group is *not* needed (uses
huggingface_hub directly, a core GLiNER dependency already).

Usage::

    uv run python data/convert_text2json.py --out data/text2json.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _charspan import find_surface_span, tokenize_with_offsets  # noqa: E402
from _jsonl import add_split_args, iter_jsonl, write_jsonl_split  # noqa: E402

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


def _add(tokens: List[str], ner: List[List[Any]], seen: set, label: str, surface: str) -> None:
    """Locate surface in tokens and append a ner entry if found and not a dupe."""
    if len(surface.split()) > MAX_SURFACE_WORDS:
        return
    span = find_surface_span(tokens, surface)
    if span is None:
        return
    key = (span[0], span[1], label)
    if key in seen:
        return
    seen.add(key)
    ner.append([span[0], span[1], label])


def _ingest_entity_list(items: list, tokens: List[str], ner: List[List[Any]], seen: set) -> None:
    """Process a list of {entity, type, description} dicts."""
    for item in items:
        if not isinstance(item, dict):
            continue
        etype = item.get("type")
        surface = _coerce_surface(item.get("entity"))
        if not isinstance(etype, str) or not etype.strip() or not surface:
            continue
        _add(tokens, ner, seen, etype.strip(), surface)


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one text2json row to a GLiNER NER-training record; None if unusable."""
    text = row.get("text")
    raw = row.get("extracted")
    if not isinstance(text, str) or not text.strip() or not raw:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    tokens, _ = tokenize_with_offsets(text)
    if not tokens:
        return None

    ner: List[List[Any]] = []
    seen: set = set()
    for key, value in data.items():
        if key == "entities" and isinstance(value, list):
            _ingest_entity_list(value, tokens, ner, seen)
            continue
        if not isinstance(key, str) or not key.strip():
            continue
        label = key.strip()
        if isinstance(value, list):
            for item in value:
                surface = _coerce_surface(item)
                if surface is not None:
                    _add(tokens, ner, seen, label, surface)
            continue
        surface = _coerce_surface(value)
        if surface is not None:
            _add(tokens, ner, seen, label, surface)

    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="knowledgator/text2json-training-data", help="HuggingFace dataset repo.")
    parser.add_argument("--file", default="augmented_train.jsonl", help="JSONL file inside the repo.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    add_split_args(parser)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    print(f"Downloading {args.repo}/{args.file}...")
    src_path = Path(hf_hub_download(args.repo, args.file, repo_type="dataset"))

    def _records():
        n = 0
        with src_path.open(encoding="utf-8") as fh:
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
