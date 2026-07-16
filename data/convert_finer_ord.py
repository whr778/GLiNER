"""Convert gtfintechlab/finer-ord (financial NER) to GLiNER NER-training JSONL.

Plain NER, no relations. Source is token-per-row: each row is
``{gold_token, gold_label, doc_idx, sent_idx}``. This regroups tokens into
sentences by ``(doc_idx, sent_idx)`` (in reading order) and decodes the
documented BIO label scheme directly into token-index spans::

    {0: O, 1: PER_B, 2: PER_I, 3: LOC_B, 4: LOC_I, 5: ORG_B, 6: ORG_I}

Since the source is already token-per-row, the sentence's own token list
*is* ``tokenized_text`` -- no offset mapping or surface search needed at
all, just BIO span decoding.

License: CC-BY-NC-4.0 (non-commercial).
Requires the optional ``data`` dependency group.

Usage::

    uv run python data/convert_finer_ord.py --out data/finer_ord.train.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import write_jsonl  # noqa: E402

# label id -> (entity_type, is_begin); anything else -> O
LABELS = {1: ("PER", True), 2: ("PER", False),
          3: ("LOC", True), 4: ("LOC", False),
          5: ("ORG", True), 6: ("ORG", False)}


def sentence_to_record(tokens: List[str], labels: List[Any]) -> Optional[Dict[str, Any]]:
    """Decode one sentence's (token, BIO-label-id) sequence into a GLiNER NER-training record."""
    spans: List[Tuple[str, int, int]] = []
    cur_type: Optional[str] = None
    start = 0
    for i, lid in enumerate(labels):
        info = LABELS.get(lid)
        if info is None:
            if cur_type is not None:
                spans.append((cur_type, start, i))
                cur_type = None
            continue
        typ, is_begin = info
        if is_begin or typ != cur_type:
            if cur_type is not None:
                spans.append((cur_type, start, i))
            cur_type, start = typ, i
    if cur_type is not None:
        spans.append((cur_type, start, len(tokens)))

    ner = [[s, e - 1, typ] for typ, s, e in spans]  # decode end is exclusive -> inclusive
    if not ner:
        return None
    return {"tokenized_text": list(tokens), "ner": ner, "relations": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="gtfintechlab/finer-ord", help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable sentences (-1 = all).")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split)

    sentences: "OrderedDict[tuple, dict]" = OrderedDict()
    for row in ds:
        tok = row.get("gold_token")
        if not isinstance(tok, str) or not tok.strip():
            continue
        key = (row.get("doc_idx"), row.get("sent_idx"))
        s = sentences.setdefault(key, {"tokens": [], "labels": []})
        s["tokens"].append(tok)
        s["labels"].append(row.get("gold_label"))

    def _records():
        n = 0
        for s in sentences.values():
            rec = sentence_to_record(s["tokens"], s["labels"])
            if rec is None:
                continue
            yield rec
            n += 1
            if args.max_records >= 0 and n >= args.max_records:
                break

    count = write_jsonl(_records(), args.out)
    print(f"wrote {count} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
