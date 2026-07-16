"""Convert WikiEvents (Li et al., NAACL 2021) to GLiNER event-training JSONL.

Unlike GLiNER2's tools/data/convert_wikievents.py, which discards WikiEvents'
native word-token offsets in favor of a surface-form-only schema (resolved
by fuzzy matching downstream), this converter reads the same raw source but
keeps the offsets -- WikiEvents already ships pre-tokenized ``tokens`` plus
(start, end) word-token spans for entity mentions and event triggers (end is
exclusive), so there is no ambiguity to resolve. Output records carry
(start, end_inclusive, label) NER tuples -- entities and event triggers in
one list -- plus (trigger_idx, arg_idx, role) relation tuples, matching what
EventExtractionSpanProcessor.preprocess_example consumes (indices reference
positions in the emitted ``ner`` list, in original/pre-sort order).

Source layout -- one JSONL per split, hosted on a public S3 bucket from
https://github.com/raspberryice/gen-arg::

    https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/train.jsonl
    https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/dev.jsonl
    https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/test.jsonl

Per record::

    {"tokens": [...],
     "entity_mentions": [{"id": ..., "entity_type": "PER", "text": ...,
                          "start": 11, "end": 15}, ...],
     "event_mentions": [{"event_type": "Life.Injure.Unspecified",
                         "trigger": {"start": 62, "end": 63, "text": "injured"},
                         "arguments": [{"entity_id": ..., "role": "Victim",
                                        "text": "Terry Duffield"}]}]}

Output (one record per source document)::

    {"tokenized_text": [...],
     "ner": [[11, 14, "PER"], ..., [62, 62, "Life.Injure.Unspecified"]],
     "relations": [[<trigger_ner_idx>, <arg_ner_idx>, "Victim"], ...]}

Usage::

    uv run python data/convert_wikievents.py --split train --out data/wikievents.train.jsonl
    uv run python data/convert_wikievents.py --split dev --out data/wikievents.dev.jsonl
    uv run python data/convert_wikievents.py --split test --out data/wikievents.test.jsonl
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import iter_jsonl, write_jsonl  # noqa: E402


S3_BASE = "https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data"


def _iter_records(source: str) -> Iterable[Dict[str, Any]]:
    if source.startswith("http://") or source.startswith("https://"):
        req = urllib.request.Request(source, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=120) as r:
            yield from iter_jsonl(line.decode("utf-8") for line in r)
    else:
        p = Path(source)
        if not p.is_file():
            raise SystemExit(f"input not found: {source}")
        with p.open(encoding="utf-8") as fh:
            yield from iter_jsonl(fh)


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one WikiEvents document to a GLiNER event-training record; None if unusable."""
    tokens = row.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        return None
    num_tokens = len(tokens)

    ner: List[List[Any]] = []
    entity_id_to_ner_idx: Dict[str, int] = {}

    for mention in row.get("entity_mentions") or []:
        if not isinstance(mention, dict):
            continue
        mid = mention.get("id")
        etype = mention.get("entity_type")
        start, end = mention.get("start"), mention.get("end")
        if not isinstance(mid, str) or not isinstance(etype, str):
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if not (0 <= start < end <= num_tokens):
            continue
        entity_id_to_ner_idx[mid] = len(ner)
        ner.append([start, end - 1, etype.strip()])

    relations: List[List[Any]] = []

    for ev in row.get("event_mentions") or []:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("event_type")
        trigger = ev.get("trigger") or {}
        if not isinstance(etype, str) or not isinstance(trigger, dict):
            continue
        t_start, t_end = trigger.get("start"), trigger.get("end")
        if not isinstance(t_start, int) or not isinstance(t_end, int):
            continue
        if not (0 <= t_start < t_end <= num_tokens):
            continue

        trigger_ner_idx = len(ner)
        ner.append([t_start, t_end - 1, etype.strip()])

        for arg in ev.get("arguments") or []:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role")
            entity_id = arg.get("entity_id")
            if not isinstance(role, str) or not isinstance(entity_id, str):
                continue
            arg_ner_idx = entity_id_to_ner_idx.get(entity_id)
            if arg_ner_idx is None:
                continue
            relations.append([trigger_ner_idx, arg_ner_idx, role.strip()])

    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": relations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--input",
        help="Local path or URL of a WikiEvents jsonl file (train.jsonl / dev.jsonl / test.jsonl).",
    )
    src_group.add_argument(
        "--split", choices=("train", "dev", "test"),
        help=f"Convenience: download the named split from the canonical S3 bucket ({S3_BASE}).",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    args = parser.parse_args()

    source = args.input if args.input else f"{S3_BASE}/{args.split}.jsonl"

    def _records():
        n = 0
        for row in _iter_records(source):
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
