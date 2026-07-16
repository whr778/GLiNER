"""Convert knowledgator/sentence_rex to GLiNER RE-training JSONL.

Source rows are sentence-level relation classification with the two
arguments marked **inline** by ``<e1>...</e1>`` / ``<e2>...</e2>`` tags::

    {"sentences": "<e1>Pope Pius XII</e1> re-opened the cause on 7 December 1954, "
                  "and Pope John Paul II proclaimed him <e2>Venerable</e2> on 6 July 1985.",
     "labels": "canonization status"}

This is the one dataset ported here with inline-tagged (not offset- or
surface-list-based) entity marking, and it has a real ordering trap: the
tags must be stripped *before* tokenizing, not after -- tokenizing the
tagged text first (or computing spans before stripping) silently shifts
every downstream token position by however many tag tokens preceded it.
This extracts the ``e1``/``e2`` surfaces from the tagged text, strips the
tags to get clean text, *then* word-tokenizes the clean text and locates
both surfaces in it via ``data/_charspan.find_surface_span``.

**Judgment call, made explicitly (there is no dataset-provided answer)**:
the source has no independent entity typing for ``e1``/``e2`` -- only the
relation label connecting them. Unlike RAMS (where the role name at least
distinguishes head from tail), there's nothing role-like here either: the
same two-entity structure repeats across ~850 different relation labels, so
using the relation label as an entity type would conflate the relation
vocabulary with the entity vocabulary. Both spans are given a single
placeholder NER type, ``"entity"`` -- explicit and honest about there being
no real typing signal in the source, rather than inventing a false
distinction. The relation classification signal itself (the actual point of
this dataset) is unaffected by this choice.

``--min-count`` drops the long singleton tail of the ~850-label relation
vocabulary if a cleaner training signal is wanted; default 1 keeps
everything, matching GLiNER2's converter.

Requires the optional ``data`` dependency group.

Usage::

    uv run python data/convert_sentence_rex.py --out data/sentence_rex.train.jsonl
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _charspan import find_surface_span, tokenize_with_offsets  # noqa: E402
from _jsonl import add_split_args, write_jsonl_split  # noqa: E402

E1_RE = re.compile(r"<e1>\s*(.*?)\s*</e1>", re.DOTALL)
E2_RE = re.compile(r"<e2>\s*(.*?)\s*</e2>", re.DOTALL)
TAG_RE = re.compile(r"</?e[12]>")

PLACEHOLDER_TYPE = "entity"


def parse_row(row: Dict[str, Any]) -> Optional[Tuple[str, str, str, str]]:
    """Return (clean_text, e1_surface, e2_surface, label) or None if unparseable."""
    sentence, label = row.get("sentences"), row.get("labels")
    if not isinstance(sentence, str) or not isinstance(label, str):
        return None
    label = label.strip()
    if not label:
        return None

    m1, m2 = E1_RE.search(sentence), E2_RE.search(sentence)
    if not m1 or not m2:
        return None
    e1, e2 = m1.group(1).strip(), m2.group(1).strip()
    if not e1 or not e2:
        return None

    # Tags stripped *before* tokenizing -- stripping after would leave stale
    # token positions computed against a string that no longer exists.
    clean = TAG_RE.sub("", sentence)
    clean = re.sub(r"\s+", " ", clean).strip()
    return (clean, e1, e2, label) if clean else None


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one sentence_rex row to a GLiNER RE-training record; None if unusable."""
    parsed = parse_row(row)
    if parsed is None:
        return None
    clean, e1, e2, label = parsed

    tokens, _ = tokenize_with_offsets(clean)
    if not tokens:
        return None

    head_span, tail_span = find_surface_span(tokens, e1), find_surface_span(tokens, e2)
    if head_span is None or tail_span is None or head_span == tail_span:
        return None

    ner = [[head_span[0], head_span[1], PLACEHOLDER_TYPE], [tail_span[0], tail_span[1], PLACEHOLDER_TYPE]]
    return {"tokenized_text": tokens, "ner": ner, "relations": [[0, 1, label]]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="knowledgator/sentence_rex", help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Minimum count for a relation label to be kept (default 1 = keep everything).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split)

    kept = None
    if args.min_count > 1:
        print("Counting relation labels...")
        label_counts: Counter = Counter()
        for row in ds:
            lbl = row.get("labels")
            if isinstance(lbl, str) and lbl.strip():
                label_counts[lbl.strip()] += 1
        kept = {label for label, n in label_counts.items() if n >= args.min_count}
        print(f"Keeping {len(kept)}/{len(label_counts)} labels (min_count={args.min_count})")

    def _records():
        n = 0
        for row in ds:
            rec = convert_row(row)
            if rec is None:
                continue
            if kept is not None and rec["relations"][0][2] not in kept:
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
