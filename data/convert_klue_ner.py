"""Convert KLUE-NER (Korean) to GLiNER NER-training JSONL.

Plain NER, no relations. Char-level BIO TSV, fetched directly from the
canonical KLUE-benchmark GitHub release (the HuggingFace loader is broken).
License: CC-BY-SA-4.0.

Korean, like Chinese, has no whitespace word boundaries, so -- matching
convert_cmnee.py's approach -- ``tokenized_text`` is ``list(chars)`` (one
character per token); a char-level BIO tag run then decodes directly into a
token-index span with no offset mapping needed at all.

Usage::

    uv run python data/convert_klue_ner.py --out data/klue_ner.train.jsonl
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import add_split_args, write_jsonl_split  # noqa: E402

URL = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-ner-v1.1/klue-ner-v1.1_train.tsv"


def _char_bio_spans(tags: List[str]) -> List[Tuple[str, int, int]]:
    """Fold a char-level BIO tag sequence into (type, start, end_inclusive) spans."""
    spans: List[Tuple[str, int, int]] = []
    cur: Optional[str] = None
    start = 0
    for i, tag in enumerate(tags):
        if not tag or tag == "O":
            if cur is not None:
                spans.append((cur, start, i - 1))
                cur = None
            continue
        prefix, _, typ = tag.partition("-") if "-" in tag else ("B", "", tag)
        typ = typ or prefix
        if prefix == "B" or typ != cur:
            if cur is not None:
                spans.append((cur, start, i - 1))
            cur, start = typ, i
    if cur is not None:
        spans.append((cur, start, len(tags) - 1))
    return spans


def sentence_to_record(chars: List[str], tags: List[str]) -> Optional[Dict[str, Any]]:
    """Convert one char-tokenized, BIO-tagged sentence to a GLiNER NER-training record."""
    spans = _char_bio_spans(tags)
    if not spans:
        return None
    return {"tokenized_text": list(chars), "ner": [[s, e, typ] for typ, s, e in spans], "relations": []}


def iter_records(tsv: str):
    """Yield GLiNER records from the raw KLUE-NER TSV (blank lines separate sentences)."""
    chars: List[str] = []
    tags: List[str] = []
    for line in tsv.split("\n"):
        if line.startswith("##"):
            continue
        if line.strip() == "" and "\t" not in line:
            if chars:
                rec = sentence_to_record(chars, tags)
                if rec is not None:
                    yield rec
                chars, tags = [], []
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        chars.append(parts[0])
        tags.append(parts[1])
    if chars:
        rec = sentence_to_record(chars, tags)
        if rec is not None:
            yield rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--url", default=URL, help="Override the source TSV URL.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    add_split_args(parser)
    args = parser.parse_args()

    print(f"Fetching {args.url} ...")
    with urllib.request.urlopen(args.url, timeout=180) as resp:
        raw = resp.read().decode("utf-8")

    def _records():
        n = 0
        for rec in iter_records(raw):
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
