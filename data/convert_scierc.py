"""Convert SciERC (scientific NER + relations) to GLiNER RE-training JSONL.

Canonical source: http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz
(~695 MB -- bundles ELMo embeddings we don't need). Pass ``--json`` to point
at an already-extracted ``processed_data/json/<split>.json`` and skip the
download.

Each doc has token ``sentences`` (flattened to one document-global token
list), ``ner`` entries ``[start, end, type]`` (inclusive, document-global
token indices), and ``relations`` ``[h_start, h_end, t_start, t_end, type]``
-- both NER and relations are natively token-span based, so unlike
DocRED/Re-DocRED this needs no coreference-cluster resolution or offset
mapping at all: a relation's head/tail spans are looked up directly against
the already-collected NER span index. License: research use (AI2 / SciERC).

Output (one record per source document)::

    {"tokenized_text": [...], "ner": [[3, 4, "Method"], ...], "relations": [[0, 1, "Used-for"], ...]}

Usage::

    uv run python data/convert_scierc.py --json processed_data/json/train.json --out data/scierc.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import write_jsonl  # noqa: E402

URL = "http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"


def convert_doc(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one SciERC document to a GLiNER RE-training record; None if unusable."""
    tokens = [t for sent in d.get("sentences", []) for t in sent]
    if not tokens:
        return None
    num_tokens = len(tokens)

    ner: List[List[Any]] = []
    span_to_idx: Dict[Tuple[int, int], int] = {}
    for sent in d.get("ner", []):
        for span in sent:
            if not isinstance(span, (list, tuple)) or len(span) != 3:
                continue
            s, e, typ = span
            if not isinstance(typ, str) or not (0 <= s <= e < num_tokens):
                continue
            key = (s, e)
            if key not in span_to_idx:
                span_to_idx[key] = len(ner)
                ner.append([s, e, typ])

    relations: List[List[Any]] = []
    for sent in d.get("relations", []):
        for r in sent:
            if not isinstance(r, (list, tuple)) or len(r) != 5:
                continue
            hs, he, ts, te, typ = r
            if not isinstance(typ, str):
                continue
            head_idx, tail_idx = span_to_idx.get((hs, he)), span_to_idx.get((ts, te))
            if head_idx is None or tail_idx is None or head_idx == tail_idx:
                continue
            relations.append([head_idx, tail_idx, typ])

    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": relations}


def _load_jsonl_text(args) -> str:
    if args.json:
        return Path(args.json).read_text(encoding="utf-8")
    member = f"processed_data/json/{args.split}.json"
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tf:
        print(f"Downloading {URL} (~695 MB)...")
        urllib.request.urlretrieve(URL, tf.name)
        with tarfile.open(tf.name) as tar:
            f = tar.extractfile(member)
            return f.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--json", default=None,
                        help="Local processed_data/json/<split>.json (skips the 695 MB download).")
    parser.add_argument("--split", default="train", help="Split to read when downloading.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    args = parser.parse_args()

    raw = _load_jsonl_text(args)

    def _records():
        n = 0
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            rec = convert_doc(json.loads(line))
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
