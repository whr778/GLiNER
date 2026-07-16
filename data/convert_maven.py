"""Convert MAVEN (Wang et al., EMNLP 2020) to GLiNER event-training JSONL.

MAVEN is trigger-and-type detection only -- no argument annotations, so
every converted record has an empty ``relations`` list; only the trigger
half of the event architecture gets supervision from this dataset (pair it
with RAMS/WikiEvents/CASIE for argument-role supervision). Keeps MAVEN's
native per-sentence (start, end) token offsets (end exclusive) rather than
GLiNER2's surface-form-only schema -- see data/convert_wikievents.py for
why offsets are preferred when the source actually has them.

Source layout (https://github.com/THU-KEG/MAVEN-dataset)::

    {"content": [{"tokens": [...]}, ...],
     "events": [{"type": "...",
                 "mention": [{"sent_id": 0, "offset": [6, 7]}, ...]}]}

Output (one record per source document)::

    {"tokenized_text": [...], "ner": [[start, end, event_type], ...], "relations": []}

Usage::

    uv run python data/convert_maven.py \\
        --input /path/to/maven/train.jsonl \\
        --out data/maven.train.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import iter_jsonl, write_jsonl  # noqa: E402


def _flatten_tokens(sentences: List[Dict[str, Any]]) -> List[str]:
    tokens: List[str] = []
    for sent in sentences:
        toks = sent.get("tokens") if isinstance(sent, dict) else None
        if isinstance(toks, list):
            tokens.extend(t for t in toks if isinstance(t, str))
    return tokens


def _sentence_token_offsets(sentences: List[Dict[str, Any]]) -> List[int]:
    """Cumulative start index of each sentence in the flat token list."""
    offsets: List[int] = []
    acc = 0
    for sent in sentences:
        offsets.append(acc)
        toks = sent.get("tokens") if isinstance(sent, dict) else None
        if isinstance(toks, list):
            acc += sum(1 for t in toks if isinstance(t, str))
    return offsets


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one MAVEN document to a GLiNER event-training record; None if unusable."""
    sentences = row.get("content")
    events_in = row.get("events")
    if not isinstance(sentences, list) or not sentences:
        return None
    tokens = _flatten_tokens(sentences)
    if not tokens:
        return None
    num_tokens = len(tokens)
    sent_starts = _sentence_token_offsets(sentences)

    ner: List[List[Any]] = []
    if isinstance(events_in, list):
        for evt in events_in:
            if not isinstance(evt, dict):
                continue
            etype = evt.get("type")
            mentions = evt.get("mention")
            if not isinstance(etype, str) or not isinstance(mentions, list):
                continue
            etype = etype.strip()
            if not etype:
                continue
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                sid = m.get("sent_id")
                offset = m.get("offset")
                if not isinstance(sid, int) or not isinstance(offset, list) or len(offset) != 2:
                    continue
                if not (0 <= sid < len(sent_starts)):
                    continue
                try:
                    s = sent_starts[sid] + int(offset[0])
                    e = sent_starts[sid] + int(offset[1])  # exclusive
                except (TypeError, ValueError):
                    continue
                if not (0 <= s < e <= num_tokens):
                    continue
                ner.append([s, e - 1, etype])

    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True,
                        help="Path to a MAVEN jsonl file (train.jsonl / valid.jsonl).")
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    args = parser.parse_args()

    p = Path(args.input)
    if not p.is_file():
        raise SystemExit(f"input not found: {args.input}")

    def _records():
        n = 0
        with p.open(encoding="utf-8") as fh:
            for row in iter_jsonl(fh):
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
