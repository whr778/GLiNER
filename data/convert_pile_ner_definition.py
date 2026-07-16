"""Convert Universal-NER/Pile-NER-definition to GLiNER NER-training JSONL.

Plain NER, no relations. Each record is a ShareGPT-style conversation::

    turns[0]: human  -> "Text: <passage>"
    turns[1]: gpt    -> "I've read this text."
    turns[2k+0]: human -> "What describes <DEFINITION> in the text?"
    turns[2k+1]: gpt   -> '["surface_1", "surface_2", ...]'   (or "[]")

Each (human, gpt) pair after the first two is one entity-type query whose
"type" is a natural-language definition (e.g. "a type of medication").
GLiNER2 mints a synthetic short key (``e_0``, ``e_1``, ...) per query and
routes the definition into a separate description side-channel to keep its
own schema tokens short. GLiNER has no such side-channel -- and doesn't
need one, since arbitrary-length natural-language labels are exactly what
GLiNER's zero-shot entity-typing is designed for -- so this uses the
definition itself as the ``ner`` label directly.

Source is surface-only (no offsets); locates each answer surface via
``data/_charspan.find_surface_span``. Empty answers (``[]``) are negative
samples for the source GPT prompt and are dropped, matching GLiNER2's
converter.

Requires the optional ``data`` dependency group.

Usage::

    uv run python data/convert_pile_ner_definition.py --out data/pile_ner_definition.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _charspan import find_surface_span, tokenize_with_offsets  # noqa: E402
from _jsonl import add_split_args, write_jsonl_split  # noqa: E402


def strip_definition(human_value: str) -> Optional[str]:
    """Extract the definition from a 'What describes X in the text?' turn."""
    prefix, suffix = "What describes ", " in the text?"
    if not (human_value.startswith(prefix) and human_value.endswith(suffix)):
        return None
    return human_value[len(prefix):-len(suffix)].strip()


def convert_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one Pile-NER record to a GLiNER NER-training record; None if unusable."""
    turns = record.get("conversations") or []
    if len(turns) < 3:
        return None
    text_turn = turns[0].get("value", "")
    if not text_turn.startswith("Text: "):
        return None
    text = text_turn[len("Text: "):]

    tokens, _ = tokenize_with_offsets(text)
    if not tokens:
        return None

    ner: List[List[Any]] = []
    seen: set = set()
    qa_turns = turns[2:]
    for i in range(0, len(qa_turns) - 1, 2):
        human, gpt = qa_turns[i].get("value", ""), qa_turns[i + 1].get("value", "")
        definition = strip_definition(human)
        if not definition:
            continue
        try:
            spans = json.loads(gpt)
        except json.JSONDecodeError:
            continue
        if not isinstance(spans, list):
            continue

        for surface in spans:
            if not isinstance(surface, str):
                continue
            span = find_surface_span(tokens, surface)
            if span is None:
                continue
            key = (span[0], span[1], definition)
            if key in seen:
                continue
            seen.add(key)
            ner.append([span[0], span[1], definition])

    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="Universal-NER/Pile-NER-definition", help="HuggingFace dataset repo.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Streaming {args.repo}...")
    ds = load_dataset(args.repo, split="train", streaming=True)

    def _records():
        n = 0
        for record in ds:
            rec = convert_record(record)
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
