"""Convert knowledgator/PubMedAbstractsNER to GLiNER NER-training JSONL.

Plain NER, no relations. Source rows are already token-indexed, inclusive
spans, so no offset mapping is needed::

    tokenized_text: ["In", "this", "article", ",", "the", "development", ...]
    ner:            [[9, 9, "Body Regions - Anatomical areas of the body."], ...]

Each label string encodes ``"TypeName - long description"``; only the type
(the prefix before the first ``" - "``) is used as the ``ner`` label -- the
description has no home in GLiNER's ``{tokenized_text, ner, relations}``
schema (unlike GLiNER2's ``entity_descriptions`` side-channel) and is
dropped.

The HuggingFace ``datasets`` library fails to parse this repo's metadata,
so this downloads ``train.json`` directly via ``huggingface_hub`` (part of
GLiNER's core dependencies already, no optional group needed) and parses it
with the stdlib.

Usage::

    uv run python data/convert_pubmed_abstracts_ner.py --out data/pubmed_abstracts_ner.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import add_split_args, write_jsonl_split  # noqa: E402


def split_label(raw: Any) -> Tuple[str, Optional[str]]:
    """Split ``"TypeName - long description"`` into ``(type, description)``."""
    if not isinstance(raw, str):
        return "", None
    sep = " - "
    idx = raw.find(sep)
    if idx < 0:
        return raw.strip(), None
    return raw[:idx].strip(), raw[idx + len(sep):].strip() or None


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one PubMed row to a GLiNER NER-training record; None if unusable."""
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
        etype, _desc = split_label(span[2])
        if not etype or not (0 <= start <= end < num_tokens):
            continue
        ner.append([start, end, etype])

    if not ner:
        return None
    return {"tokenized_text": list(tokens), "ner": ner, "relations": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="knowledgator/PubMedAbstractsNER", help="HuggingFace dataset repo.")
    parser.add_argument("--file", default="train.json", help="JSON array file inside the repo.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    add_split_args(parser)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    print(f"Downloading {args.repo}/{args.file}...")
    src_path = Path(hf_hub_download(args.repo, args.file, repo_type="dataset"))
    with src_path.open(encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise SystemExit(f"expected a JSON array in {args.file}, got {type(rows).__name__}")
    print(f"Loaded {len(rows):,} rows")

    def _records():
        n = 0
        for row in rows:
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
