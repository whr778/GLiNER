"""Convert RAMS (Ebner et al., ACL 2020) to GLiNER event-training JSONL.

RAMS annotates one event (trigger + roles) per document, with arguments
possibly in neighboring sentences. Like convert_wikievents.py, this reads
the same raw source as GLiNER2's tools/data/convert_rams.py but keeps the
native (start, end) flattened-token indices -- both inclusive in RAMS, so
unlike WikiEvents no end-1 adjustment is needed here.

RAMS has no independent generic entity typing (PER/GPE/...) for arguments
-- only the role identifier (e.g. "evt089arg02victim") -- so the parsed
role name doubles as both the argument's NER type and the relation's role
label; there is nothing else to use.

Source layout (https://nlp.jhu.edu/rams/, RAMS_1.0c.tar.gz)::

    {"sentences": [["tok", ...], ...],
     "evt_triggers": [[start, end, [["life.die.deathcaused...", 1.0]]]],
     "gold_evt_links": [[[trig_start, trig_end], [arg_start, arg_end],
                          "evt090arg02victim"], ...]}

Output (one record per source document)::

    {"tokenized_text": [...],
     "ner": [[69, 69, "life.die.deathcausedbyviolentevents"], ..., [42, 43, "victim"]],
     "relations": [[<trigger_ner_idx>, <arg_ner_idx>, "victim"], ...]}

Usage::

    uv run python data/convert_rams.py \\
        --input /path/to/RAMS_1.0c/data/train.jsonlines \\
        --out data/rams.train.jsonl
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import iter_jsonl, write_jsonl  # noqa: E402


ROLE_RE = re.compile(r"^evt\d+arg\d+([A-Za-z]+)$")


def _parse_role(raw: Any) -> Optional[str]:
    """Extract the role name from a RAMS link id like ``evt089arg02victim``."""
    if not isinstance(raw, str):
        return None
    m = ROLE_RE.match(raw.strip())
    return m.group(1) if m else None


def _event_type_of(evt_trigger: Any) -> Optional[str]:
    if not isinstance(evt_trigger, list) or len(evt_trigger) < 3:
        return None
    type_list = evt_trigger[2]
    if not isinstance(type_list, list) or not type_list:
        return None
    first = type_list[0]
    if not isinstance(first, list) or not first:
        return None
    etype = first[0]
    return etype.strip() if isinstance(etype, str) and etype.strip() else None


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one RAMS document to a GLiNER event-training record; None if unusable."""
    sentences = row.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        return None
    tokens: List[str] = [t for sent in sentences if isinstance(sent, list) for t in sent if isinstance(t, str)]
    if not tokens:
        return None
    num_tokens = len(tokens)

    ner: List[List[Any]] = []
    trigger_span_to_ner_idx: Dict[tuple, int] = {}
    for trig in row.get("evt_triggers") or []:
        if not isinstance(trig, list) or len(trig) < 2:
            continue
        try:
            s, e = int(trig[0]), int(trig[1])
        except (TypeError, ValueError):
            continue
        etype = _event_type_of(trig)
        if etype is None or not (0 <= s <= e < num_tokens):
            continue
        trigger_span_to_ner_idx[(s, e)] = len(ner)
        ner.append([s, e, etype])

    relations: List[List[Any]] = []
    arg_span_to_ner_idx: Dict[tuple, int] = {}
    for link in row.get("gold_evt_links") or []:
        if not isinstance(link, list) or len(link) != 3:
            continue
        trig_span, arg_span, raw_role = link
        if not (isinstance(trig_span, list) and len(trig_span) == 2):
            continue
        if not (isinstance(arg_span, list) and len(arg_span) == 2):
            continue
        try:
            ts, te = int(trig_span[0]), int(trig_span[1])
            as_, ae = int(arg_span[0]), int(arg_span[1])
        except (TypeError, ValueError):
            continue

        trigger_idx = trigger_span_to_ner_idx.get((ts, te))
        role = _parse_role(raw_role)
        if trigger_idx is None or role is None or not (0 <= as_ <= ae < num_tokens):
            continue

        arg_key = (as_, ae, role)
        arg_idx = arg_span_to_ner_idx.get(arg_key)
        if arg_idx is None:
            arg_idx = len(ner)
            ner.append([as_, ae, role])
            arg_span_to_ner_idx[arg_key] = arg_idx

        relations.append([trigger_idx, arg_idx, role])

    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": relations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True,
                        help="Path to a RAMS jsonlines file (train/dev/test.jsonlines).")
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
