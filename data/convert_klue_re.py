"""Convert KLUE-RE (Korean) to GLiNER RE-training JSONL.

Fetched directly from the canonical KLUE-benchmark GitHub release (the
HuggingFace loader is broken). License: CC-BY-SA-4.0.

Unlike GLiNER2's converter (surface-string match: ``word in sentence``),
this uses the native ``start_idx``/``end_idx`` char offsets each entity
carries directly (both inclusive -- verified against real data:
``sentence[start_idx:end_idx+1] == word``). Korean has no whitespace word
boundaries, so ``tokenized_text`` is ``list(sentence)`` (one character per
token, matching convert_klue_ner.py and convert_cmnee.py), making a char
offset equal to its token index with no mapping step needed.

Records labeled ``no_relation`` are dropped -- no positive training signal.

Usage::

    uv run python data/convert_klue_re.py --out data/klue_re.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import add_split_args, write_jsonl_split  # noqa: E402

URL = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-re-v1.1/klue-re-v1.1_train.json"


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one KLUE-RE row to a GLiNER RE-training record; None if unusable."""
    sent = row.get("sentence")
    label = row.get("label")
    subj, obj = row.get("subject_entity") or {}, row.get("object_entity") or {}
    if not isinstance(sent, str) or not isinstance(label, str) or label == "no_relation":
        return None

    tokens = list(sent)
    num_tokens = len(tokens)
    ner: List[List[Any]] = []
    for ent in (subj, obj):
        typ, start, end = ent.get("type"), ent.get("start_idx"), ent.get("end_idx")
        if not isinstance(typ, str) or not isinstance(start, int) or not isinstance(end, int):
            return None
        if not (0 <= start <= end < num_tokens):
            return None
        ner.append([start, end, typ])

    if ner[0][0] == ner[1][0] and ner[0][1] == ner[1][1]:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": [[0, 1, label]]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--url", default=URL, help="Override the source JSON URL.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    add_split_args(parser)
    args = parser.parse_args()

    print(f"Fetching {args.url} ...")
    with urllib.request.urlopen(args.url, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    def _records():
        n = 0
        for row in data:
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
