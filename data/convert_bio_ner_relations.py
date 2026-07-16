"""Convert knowledgator/bio-NER-relations to GLiNER RE-training JSONL.

Source rows are BioC-style, but simpler than BioRED's multi-passage layout
-- verified against 200 real rows, every document has exactly one passage
(``type: "full_text"``), so entity ``offsets`` are directly relative to that
single passage's own text with no cross-passage offset math needed::

    {"passages": [{"type": "full_text", "text": ["<full document text>"]}],
     "entities": [{"id": "18162134_T1", "type": "GeneProtein", "text": ["MST3"],
                   "offsets": [[23576, 23580]]}, ...],
     "relations": [{"type": "bind", "arg1_id": "...T1", "arg2_id": "...T3"}, ...]}

Documents with zero or more than one passage are dropped rather than
guessed at (unverified against real data -- none observed in 200 sampled
rows). Relations resolve via ``arg{1,2}_id`` lookup into the entity table
built from real offsets, not surface text.

``--skip-types`` drops noisy entity buckets (default ``umlsterm`` --
auto-extracted UMLS concept matches that dominate entity assignments); pass
``--skip-types ''`` to keep them.

Requires the optional ``data`` dependency group.

Usage::

    uv run python data/convert_bio_ner_relations.py --out data/bio_ner_relations.train.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _charspan import char_span_to_token_span, tokenize_with_offsets  # noqa: E402
from _jsonl import write_jsonl  # noqa: E402

DEFAULT_SKIP_TYPES = {"umlsterm"}


def convert_row(row: Dict[str, Any], skip_types: set) -> Optional[Dict[str, Any]]:
    """Convert one bio-NER-relations row to a GLiNER RE-training record; None if unusable."""
    passages = row.get("passages") or []
    if len(passages) != 1:
        return None
    text_field = passages[0].get("text")
    if not isinstance(text_field, list) or len(text_field) != 1 or not isinstance(text_field[0], str):
        return None
    text = text_field[0]
    if not text.strip():
        return None

    tokens, offsets = tokenize_with_offsets(text)
    if not tokens:
        return None
    num_tokens = len(tokens)

    ner: List[List[Any]] = []
    id_to_idx: Dict[str, int] = {}
    for ent in row.get("entities") or []:
        eid, etype = ent.get("id"), ent.get("type")
        span_list = ent.get("offsets") or []
        if not isinstance(eid, str) or not isinstance(etype, str) or not span_list:
            continue
        etype = etype.strip()
        if not etype or etype in skip_types:
            continue
        char_start, char_end = span_list[0]
        tok_span = char_span_to_token_span(offsets, char_start, char_end)
        if tok_span is None or not (0 <= tok_span[1] < num_tokens):
            continue
        id_to_idx[eid] = len(ner)
        ner.append([tok_span[0], tok_span[1], etype])

    if not ner:
        return None

    relations: List[List[Any]] = []
    for rel in row.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        rname = rel.get("type")
        if not isinstance(rname, str) or not rname.strip():
            continue
        head_idx, tail_idx = id_to_idx.get(rel.get("arg1_id")), id_to_idx.get(rel.get("arg2_id"))
        if head_idx is None or tail_idx is None or head_idx == tail_idx:
            continue
        relations.append([head_idx, tail_idx, rname.strip()])

    return {"tokenized_text": tokens, "ner": ner, "relations": relations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="knowledgator/bio-NER-relations", help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--skip-types", default="umlsterm",
                        help="Comma-separated entity types to drop (default: 'umlsterm'). "
                             "Pass '' to keep everything.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    args = parser.parse_args()

    from datasets import load_dataset

    skip_types = {t.strip() for t in args.skip_types.split(",") if t.strip()}
    print(f"Loading {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split)

    def _records():
        n = 0
        for row in ds:
            rec = convert_row(row, skip_types)
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
