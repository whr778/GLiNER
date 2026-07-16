"""Convert CMNEE (Zhu et al., LREC-COLING 2024) to GLiNER event-training JSONL.

CMNEE -- Chinese Military News Event Extraction -- is a 17,000-document
event-extraction corpus with both triggers and typed arguments (8 event
types / 11 argument roles), document-level, multi-event-per-document,
character-offset annotated.

CMNEE has no independent argument entity typing (only a ``role``, like
RAMS) -- the role name doubles as the argument's NER label since there is
nothing else to use.

Unlike GLiNER2's own converter (substring search: ``trigger_text not in
text``), this uses the native ``offset: [start, end]`` char spans CMNEE
ships on both trigger and arguments directly -- Chinese has no whitespace
word boundaries, so ``tokenized_text`` here is simply ``list(text)`` (one
character per token), which makes a char offset *equal* to its token index;
no offset-to-token mapping step is needed at all, unlike the CASIE/DocEE
converters which word-tokenize English text.

Source layout (Google Drive, see upstream README; ``gdown`` wraps it)::

    {"id": "...", "text": "<Chinese document>",
     "event_list": [
         {"event_type": "Manoeuvre",
          "trigger":   {"text": "...", "offset": [18, 20]},
          "arguments": [{"role": "Subject", "text": "...", "offset": [6, 10]}, ...]}
     ]}

Output (one record per source document)::

    {"tokenized_text": [...],  # list of individual characters
     "ner": [[18, 19, "Manoeuvre"], [6, 9, "Subject"], ...],
     "relations": [[<trigger_ner_idx>, <arg_ner_idx>, "Subject"], ...]}

CMNEE ships canonical train/valid/test splits -- run this once per split.

Usage::

    uv run python data/convert_cmnee.py \\
        --input data/cmnee/CMNEE/train.json --out data/cmnee.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import write_jsonl  # noqa: E402


def _offset_span(obj: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Read an ``offset: [start, end]`` pair; end is treated as exclusive."""
    offset = obj.get("offset")
    if not isinstance(offset, list) or len(offset) != 2:
        return None
    try:
        s, e = int(offset[0]), int(offset[1])
    except (TypeError, ValueError):
        return None
    return (s, e) if s < e else None


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one CMNEE document to a GLiNER event-training record; None if unusable."""
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    tokens = list(text)
    num_tokens = len(tokens)

    ner: List[List[Any]] = []
    relations: List[List[Any]] = []
    arg_span_to_ner_idx: Dict[Tuple[int, int, str], int] = {}

    for ev in row.get("event_list") or []:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("event_type")
        trigger = ev.get("trigger") or {}
        if not isinstance(etype, str) or not isinstance(trigger, dict):
            continue
        etype = etype.strip()
        trig_span = _offset_span(trigger)
        if not etype or trig_span is None or not (0 <= trig_span[0] < trig_span[1] <= num_tokens):
            continue

        trigger_idx = len(ner)
        ner.append([trig_span[0], trig_span[1] - 1, etype])

        for arg in ev.get("arguments") or []:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role")
            if not isinstance(role, str):
                continue
            role = role.strip()
            arg_span = _offset_span(arg)
            if not role or arg_span is None or not (0 <= arg_span[0] < arg_span[1] <= num_tokens):
                continue

            arg_key = (arg_span[0], arg_span[1] - 1, role)
            arg_idx = arg_span_to_ner_idx.get(arg_key)
            if arg_idx is None:
                arg_idx = len(ner)
                ner.append([arg_span[0], arg_span[1] - 1, role])
                arg_span_to_ner_idx[arg_key] = arg_idx

            relations.append([trigger_idx, arg_idx, role])

    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": relations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path,
                        help="Local path to a CMNEE split file "
                             "(train.json / valid.json / test.json).")
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")

    with args.input.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise SystemExit(f"expected a JSON array of records, got {type(data).__name__}")

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

    count = write_jsonl(_records(), args.out)
    print(f"wrote {count} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
