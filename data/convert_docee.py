"""Convert DocEE (Tong et al., NAACL 2022) to GLiNER event-training JSONL.

DocEE is the largest publicly-available document-level event extraction
corpus: ~27k documents, 59 event types, 356 argument-role types. It follows
a strict one-event-per-document paradigm -- the event type is a
document-level label, not a span -- and **does not annotate triggers**.

This does not fit the trigger/argument event architecture natively (there
is no trigger span to select). To still get event-role supervision out of
this corpus, a synthetic trigger token ``[<event_type>]`` is prepended to
every document (mirroring GLiNER2's own optional ``--emit-events`` flag,
which does the same prepend-to-text trick) and becomes ``ner[0]`` / the
head of every relation in that document. Every real argument keeps its
native span -- DocEE's ``annotations`` entries carry ``start``/``end`` char
offsets against the body text (GLiNER2's own converter ignores these and
does a substring search instead, which mis-locates any surface string that
repeats at multiple positions); this converter uses the offsets directly
and validates each one against ``text`` (falls back to a surface search
only when the offset doesn't roundtrip).

Source layout (one canonical split file, or the DocEE-en.json all-data
file -- Google-Drive-distributed, see https://github.com/tongmeihan1995/docee)::

    [[title, text, event_type, annotations], ...]

    annotations: [{"start": 278, "end": 294, "type": "Deceased", "text": "..."}, ...]

Output (one record per source document)::

    {"tokenized_text": ["[Famous Person - Death]", "MONTERREY", ",", ...],
     "ner": [[0, 0, "Famous Person - Death"], [51, 52, "Deceased"], ...],
     "relations": [[0, 1, "Deceased"], ...]}

Usage::

    uv run python data/convert_docee.py \\
        --input DocEE-en/normal_setting/train.json --out data/docee.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _charspan import char_span_to_token_span, find_surface_span, tokenize_with_offsets  # noqa: E402
from _jsonl import write_jsonl  # noqa: E402

TEXT_KEYS = ("text", "content", "body", "passage")
EVENT_TYPE_KEYS = ("event_type", "type", "label", "event")
ANNOTATIONS_KEYS = ("annotations", "args", "arguments", "labels", "meta", "metadata", "event_arguments")


def _normalise_record(raw: Any) -> Optional[Dict[str, Any]]:
    """Coerce a raw DocEE record (4-element list, or dict) into a uniform dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        title, text, event_type, annotations = raw[0], raw[1], raw[2], raw[3]
        if isinstance(annotations, str):
            try:
                annotations = json.loads(annotations)
            except json.JSONDecodeError:
                annotations = []
        return {
            "title": title if isinstance(title, str) else None,
            "text": text if isinstance(text, str) else None,
            "event_type": event_type if isinstance(event_type, str) else None,
            "annotations": annotations if isinstance(annotations, list) else [],
        }
    return None


def _first_value(rec: Dict[str, Any], keys) -> Any:
    for k in keys:
        v = rec.get(k)
        if v not in (None, "", [], {}):
            return v
    return None


def _get_annotations(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = _first_value(rec, ANNOTATIONS_KEYS)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    return [a for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []


def convert_row(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one DocEE document to a GLiNER event-training record; None if unusable."""
    text = _first_value(rec, TEXT_KEYS)
    event_type = _first_value(rec, EVENT_TYPE_KEYS)
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(event_type, str) or not event_type.strip():
        return None
    event_type = event_type.strip()

    tokens, offsets = tokenize_with_offsets(text)
    if not tokens:
        return None
    num_tokens = len(tokens)

    # Trigger is synthetic and occupies index 0 in the shifted token list;
    # real argument spans are computed against the unshifted tokens, then +1.
    ner: List[List[Any]] = [[0, 0, event_type]]
    relations: List[List[Any]] = []
    arg_span_to_ner_idx: Dict[Tuple[int, int, str], int] = {}

    for ann in _get_annotations(rec):
        role = ann.get("type")
        surface = ann.get("text")
        if not isinstance(role, str) or not isinstance(surface, str):
            continue
        role, surface = role.strip(), surface.strip()
        if not role or not surface:
            continue

        tok_span = None
        start, end = ann.get("start"), ann.get("end")
        if isinstance(start, int) and isinstance(end, int) and start < end:
            candidate = char_span_to_token_span(offsets, start, end)
            if candidate is not None:
                decoded = text[offsets[candidate[0]][0]:offsets[candidate[1]][1]]
                if decoded.strip() == surface.strip():
                    tok_span = candidate
        if tok_span is None:
            tok_span = find_surface_span(tokens, surface)
        if tok_span is None or not (0 <= tok_span[1] < num_tokens):
            continue

        shifted = (tok_span[0] + 1, tok_span[1] + 1)
        arg_key = (shifted[0], shifted[1], role)
        arg_idx = arg_span_to_ner_idx.get(arg_key)
        if arg_idx is None:
            arg_idx = len(ner)
            ner.append([shifted[0], shifted[1], role])
            arg_span_to_ner_idx[arg_key] = arg_idx
        relations.append([0, arg_idx, role])

    if not relations:
        return None
    return {"tokenized_text": [f"[{event_type}]"] + tokens, "ner": ner, "relations": relations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to a DocEE JSON file (canonical split, or DocEE-en.json).")
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")

    with args.input.open(encoding="utf-8") as fh:
        data = json.load(fh)
    items = data if isinstance(data, list) else next(
        (data[k] for k in ("data", "records", "examples", "items") if isinstance(data.get(k), list)), []
    )
    if not items:
        raise SystemExit(f"could not find a list of records in {args.input}")

    def _records():
        n = 0
        for raw in items:
            rec = _normalise_record(raw)
            if rec is None:
                continue
            out = convert_row(rec)
            if out is None:
                continue
            yield out
            n += 1
            if args.max_records >= 0 and n >= args.max_records:
                break

    count = write_jsonl(_records(), args.out)
    print(f"wrote {count} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
