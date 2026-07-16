"""Convert a HuggingFace token-classification NER dataset (tokens + BIO tags) to GLiNER NER-training JSONL.

Plain NER, no relations. Generic over the three tag encodings these
datasets use:

* BIO strings (``["B-LAW", "I-LAW", "O", ...]``)              -- e.g. kaznerd
* ClassLabel ints whose names live in the feature               -- e.g. bc4chemd
* bare ints needing an explicit label file (``--label-file``)   -- e.g. tner/bc5cdr

The source is already token-per-row, so ``tokens`` is used directly as
``tokenized_text``; each maximal ``B-/I-<type>`` run decodes straight into a
token-index span, no offset mapping or surface reconstruction needed.

Requires the optional ``data`` dependency group.

Usage::

    uv run python data/convert_hf_token_ner.py \\
        --repo yeshpanovrustem/kaznerd --out data/kaznerd.train.jsonl

    uv run python data/convert_hf_token_ner.py \\
        --repo tner/bc5cdr --revision refs/convert/parquet --tags-col tags \\
        --label-file dataset/label.json --out data/bc5cdr.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import write_jsonl  # noqa: E402


def bio_to_ner_spans(tags: List[str]) -> List[Tuple[str, int, int]]:
    """Fold a BIO ``<tag>`` sequence into (type, start, end_inclusive) spans."""
    spans: List[Tuple[str, int, int]] = []
    cur_type: Optional[str] = None
    start = 0
    for idx, tag in enumerate(tags):
        if not tag or tag == "O":
            if cur_type is not None:
                spans.append((cur_type, start, idx - 1))
                cur_type = None
            continue
        prefix, _, typ = tag.partition("-") if "-" in tag else ("B", "", tag)
        typ = typ or prefix
        if prefix == "B" or typ != cur_type:
            if cur_type is not None:
                spans.append((cur_type, start, idx - 1))
            cur_type, start = typ, idx
    if cur_type is not None:
        spans.append((cur_type, start, len(tags) - 1))
    return spans


def _id_to_name(features, tags_col, label_file_names):
    """Return an id->name list for int tags, or None if tags are already strings."""
    if label_file_names is not None:
        return label_file_names
    feat = features.get(tags_col)
    inner = getattr(feat, "feature", None)
    return getattr(inner, "names", None) or getattr(feat, "names", None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="HuggingFace dataset repo.")
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--revision", default=None,
                        help="Dataset revision (e.g. refs/convert/parquet for script datasets).")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--tokens-col", default="tokens", help="Token list column.")
    parser.add_argument("--tags-col", default="ner_tags", help="BIO tag column.")
    parser.add_argument("--label-file", default=None,
                        help="Repo file with a {label: id} map for bare-int tags "
                             "(e.g. tner's dataset/label.json).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    args = parser.parse_args()

    from datasets import load_dataset

    names = None
    if args.label_file:
        from huggingface_hub import hf_hub_download

        lp = hf_hub_download(args.repo, args.label_file, repo_type="dataset")
        label2id = json.loads(Path(lp).read_text())
        names = [None] * (max(label2id.values()) + 1)
        for label, i in label2id.items():
            names[i] = label

    print(f"Loading {args.repo} revision={args.revision} split={args.split}...")
    ds = load_dataset(args.repo, revision=args.revision, split=args.split)
    id2name = _id_to_name(ds.features, args.tags_col, names)

    def _records():
        n = 0
        for row in ds:
            tokens = row.get(args.tokens_col) or []
            tags = row.get(args.tags_col) or []
            m = min(len(tokens), len(tags))
            if m == 0:
                continue
            tokens, tags = tokens[:m], tags[:m]  # align on the common prefix
            if id2name is not None:
                tags = [id2name[t] if isinstance(t, int) and 0 <= t < len(id2name) else "O" for t in tags]
            spans = bio_to_ner_spans([str(t) for t in tags])
            if not spans:
                continue
            rec: Dict[str, Any] = {
                "tokenized_text": [str(t) for t in tokens],
                "ner": [[s, e, typ] for typ, s, e in spans],
                "relations": [],
            }
            yield rec
            n += 1
            if args.max_records >= 0 and n >= args.max_records:
                break

    count = write_jsonl(_records(), args.out)
    print(f"wrote {count} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
