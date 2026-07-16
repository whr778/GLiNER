"""Convert ACE 2005 English annotations to GLiNER event-training JSONL.

ACE 2005 (LDC2006T06) is the canonical event-extraction benchmark: 33 event
subtypes, 7 entity types, 6 relation types, with event-argument fillers
themselves drawn from the entity-mention table. LDC-licensed -- this port
mirrors GLiNER2's tools/data/convert_ace2005.py logic but has **not** been
run or verified against real data, since no locally-licensed copy is
available in this environment. Treat it as a code port pending verification
against an actual LDC2006T06 delivery.

Structural note on why this diverges from the WikiEvents/RAMS/CASIE/CMNEE
converters, which all use native char/token offsets: ACE's ``charseq``
offsets are relative to the *original* SGM file, but both this converter
and GLiNER2's strip SGML markup and collapse whitespace before matching
(``_strip_sgml``), which invalidates those offsets. GLiNER2's converter
works around this with a substring search (``extent_text not in text``);
this port does the same (``_charspan.find_surface_span`` on the stripped,
word-tokenized text) for consistency with the only implementation that has
actually been run against this corpus. The known limitation this carries
over: a surface string repeated verbatim at multiple positions in a
document resolves to its *first* occurrence only (see convert_docee.py /
convert_casie.py for the offset-based alternative used where the source
actually ships usable offsets).

Only entities and events are ported -- ACE's separate non-event relation
annotations (ORG-AFF, PART-WHOLE, ...) have no trigger and don't fit the
trigger/argument event schema; they belong to a plain NER/RE converter
instead, out of scope here. The mention-type filter machinery
(NAM/NOM/PRO) from GLiNER2's converter is also dropped -- unverifiable
config-driven filtering adds risk without adding capability over the
default (keep all mention types).

Event arguments: when an ``event_mention_argument``'s REFID resolves to an
already-collected entity mention, that mention's own TYPE.SUBTYPE becomes
the argument's ``ner`` label (rich typing, e.g. "PER.Individual"). When it
doesn't (value/time arguments -- Money, Time-Within, Job-Title, ... are not
entity mentions), the argument's own extent text is used with the role
name as its label, matching convert_rams.py's fallback ("role doubles as
type" when nothing else is available).

Source layout -- typical LDC delivery, only the ``adj`` (adjudicated gold)
annotation-pass folders are converted (the other passes duplicate the same
documents)::

    ace_2005_td_v7/data/English/<genre>/adj/<doc>.sgm
                                             <doc>.apf.xml

Output (one record per source document)::

    {"tokenized_text": [...],
     "ner": [[3, 4, "PER.Individual"], [12, 12, "Conflict.Attack"], ...],
     "relations": [[<trigger_ner_idx>, <arg_ner_idx>, "Attacker"], ...]}

Usage::

    uv run python data/convert_ace2005.py \\
        --input /path/to/ace_2005_td_v7/data/English --out data/ace2005.jsonl
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _charspan import find_surface_span, tokenize_with_offsets  # noqa: E402
from _jsonl import add_split_args, write_jsonl_split  # noqa: E402

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def _strip_sgml(sgm_text: str) -> str:
    m = re.search(r"<TEXT>(.*?)</TEXT>", sgm_text, re.DOTALL | re.IGNORECASE)
    body = m.group(1) if m else sgm_text
    stripped = TAG_RE.sub(" ", body)
    return WS_RE.sub(" ", stripped).strip()


def _first_charseq_text(parent: ET.Element, sub_tag: str) -> Optional[str]:
    sub = parent.find(sub_tag)
    if sub is None:
        return None
    cs = sub.find("charseq")
    if cs is None or cs.text is None:
        return None
    return WS_RE.sub(" ", cs.text).strip() or None


def _pair_sgm(apf_path: Path) -> Optional[Path]:
    stem = apf_path.name
    sgm = apf_path.with_name(stem[:-len(".apf.xml")] + ".sgm") if stem.endswith(".apf.xml") \
        else apf_path.with_suffix(".sgm")
    return sgm if sgm.is_file() else None


def _add_ner(ner: List[List[Any]], tokens: List[str], surface: str, label: str,
             span_cache: Dict[str, Optional[Tuple[int, int]]]) -> Optional[int]:
    """Locate ``surface`` in ``tokens`` (cached per surface string) and append a ner entry."""
    if surface not in span_cache:
        span_cache[surface] = find_surface_span(tokens, surface)
    span = span_cache[surface]
    if span is None:
        return None
    idx = len(ner)
    ner.append([span[0], span[1], label])
    return idx


def parse_apf(apf_path: Path, keep_subtypes: bool) -> Optional[Dict[str, Any]]:
    """Parse one .apf.xml + .sgm pair into a GLiNER event-training record; None if unusable."""
    sgm_path = _pair_sgm(apf_path)
    if sgm_path is None:
        return None
    try:
        root = ET.parse(apf_path).getroot()
    except ET.ParseError:
        return None
    try:
        sgm_text = sgm_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = _strip_sgml(sgm_text)
    if not text:
        return None
    tokens, _ = tokenize_with_offsets(text)
    if not tokens:
        return None

    ner: List[List[Any]] = []
    span_cache: Dict[str, Optional[Tuple[int, int]]] = {}
    mention_to_ner_idx: Dict[str, int] = {}

    for entity in root.iter("entity"):
        etype = (entity.get("TYPE") or "").strip()
        esub = (entity.get("SUBTYPE") or "").strip()
        if not etype:
            continue
        full_type = f"{etype}.{esub}" if keep_subtypes and esub else etype
        for emention in entity.iter("entity_mention"):
            mid = emention.get("ID")
            extent_text = _first_charseq_text(emention, "extent")
            if not mid or not extent_text:
                continue
            idx = _add_ner(ner, tokens, extent_text, full_type, span_cache)
            if idx is not None:
                mention_to_ner_idx[mid] = idx

    relations: List[List[Any]] = []
    for evt in root.iter("event"):
        etype = (evt.get("TYPE") or "").strip()
        esub = (evt.get("SUBTYPE") or "").strip()
        if not etype:
            continue
        event_type = f"{etype}.{esub}" if keep_subtypes and esub else etype
        for emention in evt.iter("event_mention"):
            anchor_text = _first_charseq_text(emention, "anchor")
            if not anchor_text:
                continue
            trigger_idx = _add_ner(ner, tokens, anchor_text, event_type, span_cache)
            if trigger_idx is None:
                continue

            seen_args: set = set()
            for arg in emention.iter("event_mention_argument"):
                role = (arg.get("ROLE") or "").strip()
                arg_text = _first_charseq_text(arg, "extent")
                if not role or not arg_text:
                    continue
                refid = arg.get("REFID")
                arg_idx = mention_to_ner_idx.get(refid) if refid else None
                if arg_idx is None:
                    arg_idx = _add_ner(ner, tokens, arg_text, role, span_cache)
                if arg_idx is None:
                    continue
                key = (trigger_idx, arg_idx, role)
                if key in seen_args:
                    continue
                seen_args.add(key)
                relations.append([trigger_idx, arg_idx, role])

    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": relations}


def iter_apf_files(root: Path):
    """Yield .apf.xml files from the adjudicated ("adj") annotation folders only."""
    for path in root.rglob("*.apf.xml"):
        if "adj" in path.relative_to(root).parts:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path,
                        help="Root directory of the ACE 2005 corpus (typically "
                             "ace_2005_td_v7/data/English).")
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    parser.add_argument("--no-subtypes", action="store_true",
                        help="Use only top-level event/entity types (drop SUBTYPE everywhere).")
    add_split_args(parser)
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"input directory not found: {args.input}")

    def _records():
        n = 0
        for apf_path in iter_apf_files(args.input):
            rec = parse_apf(apf_path, keep_subtypes=not args.no_subtypes)
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
