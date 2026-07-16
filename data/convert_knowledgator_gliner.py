"""Convert knowledgator/GLINER-multi-task-synthetic-data to GLiNER NER-training JSONL.

Plain NER, no relations. Each row carries an "Identify ... Text:" prompt
prefix before the actual body::

    tokenized_text: ["Identify", "the", "following", ..., "Text", ":", "\\n",
                     "Gurgurnica", "(", ",", ")", "is", ...]
    ner:            [["19", "19", "\\"Village\\""], ["30", "30", "\\"Municipality\\""], ...]

The body starts right after the first ``("Text", ":", "\\n")`` trio; NER
span indices are re-based into the body's own token numbering (dropping the
prefix entirely) rather than kept relative to the full prompt+body sequence,
so the model isn't trained on the synthetic prompt template. Indices are
already token-based (inclusive, string-typed) -- no offset mapping needed.

Requires the optional ``data`` dependency group (``uv add --optional data
datasets``, or ``uv sync --extra data``).

Output (one record per source row, prompt prefix stripped)::

    {"tokenized_text": ["Gurgurnica", "(", ...], "ner": [[0, 0, "Village"], ...], "relations": []}

Usage::

    uv run python data/convert_knowledgator_gliner.py --out data/knowledgator_gliner.train.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import add_split_args, write_jsonl_split  # noqa: E402
from convert_gliner_multilingual import unwrap_label  # noqa: E402


def find_body_start(tokens: List[str]) -> Optional[int]:
    """Return the index right after the first ``"Text", ":", "\\n"`` trio."""
    for i in range(len(tokens) - 2):
        if tokens[i] == "Text" and tokens[i + 1] == ":" and tokens[i + 2] == "\n":
            return i + 3
    return None


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one knowledgator row to a GLiNER NER-training record; None if unusable."""
    tokens = row.get("tokenized_text") or []
    ner_in = row.get("ner") or []
    if not tokens or not ner_in:
        return None

    body_start = find_body_start(tokens)
    if body_start is None:
        return None
    body_tokens = tokens[body_start:]
    if not body_tokens:
        return None
    num_tokens = len(body_tokens)

    ner: List[List[Any]] = []
    for span in ner_in:
        if not isinstance(span, (list, tuple)) or len(span) != 3:
            continue
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            continue
        label = unwrap_label(span[2])
        if label is None:
            continue
        rel_start, rel_end = start - body_start, end - body_start
        if not (0 <= rel_start <= rel_end < num_tokens):
            continue
        ner.append([rel_start, rel_end, label])

    if not ner:
        return None
    return {"tokenized_text": list(body_tokens), "ner": ner, "relations": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="knowledgator/GLINER-multi-task-synthetic-data",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Streaming {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split, streaming=True)

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

    counts = write_jsonl_split(_records(), args.out, ratios=args.split_ratios, seed=args.split_seed)
    print(f"wrote train={counts['train']} val={counts['val']} test={counts['test']} "
          f"-> {args.out} (.train/.val/.test.jsonl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
