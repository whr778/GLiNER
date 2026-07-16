"""Convert thunlp/docred (DocRED) to GLiNER RE-training JSONL.

DocRED is document-level relation extraction. Each row is::

    {"sents": [["Token", "list", "per", "sentence"], ...],
     "vertexSet": [[{"name": ..., "sent_id": 0, "pos": [start, end], "type": "ORG"},
                    ...more coref mentions...], ...one list per entity...],
     "labels": {"head": [e_i, ...], "tail": [e_j, ...], "relation_text": ["country", ...]}}

Each ``vertexSet`` entry is one entity as a coreference cluster of mentions
(``pos`` is a token range within its own sentence, end exclusive); ``labels``
gives cluster-to-cluster relations (document-level -- a relation holds
between two *entities*, not two specific mention occurrences).

Unlike GLiNER2's converter (which discards position info and buckets only
the first-seen surface string per entity type), this uses ``vertexSet``'s
own coreference clustering plus each mention's real ``sent_id``/``pos`` to
build one ``ner`` span per mention (flattened into whole-document token
indices), and links relations through a representative mention per cluster
(the first one) -- correct here, not a limitation: DocRED's relations are
inherently between entity clusters, not individual mention pairs, so a
representative mention is the right anchor regardless of how much position
fidelity is kept elsewhere. This is a stronger approach than surface-text
search: two mentions with different surface strings (e.g. an alias or
pronoun) in the same cluster are correctly recognized as the same entity via
structural identity, not string matching.

Requires the optional ``data`` dependency group. The HF dataset ships a
loader script newer ``datasets`` versions don't run; this reads the
auto-converted parquet revision instead (``refs/convert/parquet``).

Output (one record per source document)::

    {"tokenized_text": [...],
     "ner": [[3, 3, "ORG"], [3, 3, "ORG"], [40, 41, "LOC"], ...],  # one entry per mention
     "relations": [[0, 2, "country"], ...]}  # cluster-representative mention indices

Usage::

    uv run python data/convert_docred.py --out data/docred.train.jsonl --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import write_jsonl  # noqa: E402


def _flatten_sentences(sents: List[List[str]]) -> Tuple[List[str], List[int]]:
    """Flatten per-sentence token lists into one document token list, with cumulative sentence offsets."""
    tokens: List[str] = []
    sent_starts: List[int] = []
    for sent in sents:
        sent_starts.append(len(tokens))
        if isinstance(sent, list):
            tokens.extend(t for t in sent if isinstance(t, str))
    return tokens, sent_starts


def _mention_span(sent_starts: List[int], num_tokens: int, mention: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Flatten a mention's (sent_id, pos) into a whole-document inclusive token span."""
    sid, pos = mention.get("sent_id"), mention.get("pos")
    if not isinstance(sid, int) or not isinstance(pos, (list, tuple)) or len(pos) != 2:
        return None
    if not (0 <= sid < len(sent_starts)):
        return None
    start, end = sent_starts[sid] + pos[0], sent_starts[sid] + pos[1] - 1  # pos end is exclusive
    return (start, end) if 0 <= start <= end < num_tokens else None


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one DocRED document to a GLiNER RE-training record; None if unusable."""
    sents = row.get("sents")
    if not isinstance(sents, list) or not sents:
        return None
    tokens, sent_starts = _flatten_sentences(sents)
    if not tokens:
        return None
    num_tokens = len(tokens)

    ner: List[List[Any]] = []
    cluster_rep_idx: Dict[int, int] = {}  # vertexSet cluster index -> representative ner idx

    for cluster_idx, mentions in enumerate(row.get("vertexSet") or []):
        if not isinstance(mentions, list):
            continue
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            etype = mention.get("type")
            span = _mention_span(sent_starts, num_tokens, mention)
            if not isinstance(etype, str) or not etype.strip() or span is None:
                continue
            idx = len(ner)
            ner.append([span[0], span[1], etype.strip()])
            cluster_rep_idx.setdefault(cluster_idx, idx)

    if not ner:
        return None

    relations: List[List[Any]] = []
    labels = row.get("labels") or {}
    heads, tails, names = labels.get("head") or [], labels.get("tail") or [], labels.get("relation_text") or []
    for h, t, rname in zip(heads, tails, names):
        if not isinstance(rname, str) or not rname.strip():
            continue
        head_idx, tail_idx = cluster_rep_idx.get(h), cluster_rep_idx.get(t)
        if head_idx is None or tail_idx is None or head_idx == tail_idx:
            continue
        relations.append([head_idx, tail_idx, rname.strip()])

    return {"tokenized_text": tokens, "ner": ner, "relations": relations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="thunlp/docred", help="HuggingFace dataset repo.")
    parser.add_argument("--revision", default="refs/convert/parquet",
                        help="Dataset revision (default: the auto-converted parquet branch).")
    parser.add_argument("--split", default="train",
                        help="Parquet split: train (annotated), validation, or test (no labels).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.repo} revision={args.revision} split={args.split}...")
    ds = load_dataset(args.repo, revision=args.revision, split=args.split)

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

    count = write_jsonl(_records(), args.out)
    print(f"wrote {count} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
