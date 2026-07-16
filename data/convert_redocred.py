"""Convert tonytan48/Re-DocRED to GLiNER RE-training JSONL.

Re-DocRED is a revised, higher-quality version of DocRED with corrected
entity and relation annotations (Tan et al., 2022). Same document schema as
DocRED (``sents`` + ``vertexSet`` mention clusters with real ``sent_id``/
``pos`` offsets) -- reuses convert_docred.py's mention-flattening and span
logic directly. Two differences:

* ``labels`` is a list of dicts ``{h, t, r, evidence}`` (not DocRED's
  parallel arrays), and ``r`` is a Wikidata property ID (``"P17"``) mapped
  here to the same human-readable string DocRED's ``relation_text`` uses
  (``"country"``), so a model co-trained on both sees one label per relation.
* Canonical train/validation/test splits (3053/500/500 docs) -- no random
  partitioning needed, call once per split.

Requires the optional ``data`` dependency group.

Usage::

    uv run python data/convert_redocred.py --split train --out data/redocred.train.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import write_jsonl  # noqa: E402
from convert_docred import _flatten_sentences, _mention_span  # noqa: E402

# Relation-text strings match DocRED's relation_text field verbatim so a
# model co-trained on both datasets shares one label per Wikidata P-ID.
_RELATION_TEXT: Dict[str, str] = {
    "P6": "head of government", "P17": "country", "P19": "place of birth",
    "P20": "place of death", "P22": "father", "P25": "mother", "P26": "spouse",
    "P27": "country of citizenship", "P30": "continent", "P31": "instance of",
    "P35": "head of state", "P36": "capital", "P37": "official language",
    "P39": "position held", "P40": "child", "P50": "author",
    "P54": "member of sports team", "P57": "director", "P58": "screenwriter",
    "P69": "educated at", "P86": "composer", "P102": "member of political party",
    "P108": "employer", "P112": "founded by", "P118": "league", "P123": "publisher",
    "P127": "owned by", "P131": "located in the administrative territorial entity",
    "P136": "genre", "P137": "operator", "P140": "religion",
    "P150": "contains administrative territorial entity", "P155": "follows",
    "P156": "followed by", "P159": "headquarters location", "P161": "cast member",
    "P162": "producer", "P166": "award received", "P170": "creator",
    "P171": "parent taxon", "P172": "ethnic group", "P175": "performer",
    "P176": "manufacturer", "P178": "developer", "P179": "series",
    "P190": "sister city", "P194": "legislative body", "P205": "basin country",
    "P206": "located in or next to body of water", "P241": "military branch",
    "P264": "record label", "P272": "production company", "P276": "location",
    "P279": "subclass of", "P355": "subsidiary", "P361": "part of",
    "P364": "original language of work", "P400": "platform",
    "P403": "mouth of the watercourse", "P449": "original network",
    "P463": "member of", "P488": "chairperson", "P495": "country of origin",
    "P527": "has part", "P551": "residence", "P569": "date of birth",
    "P570": "date of death", "P571": "inception",
    "P576": "dissolved, abolished or demolished", "P577": "publication date",
    "P580": "start time", "P582": "end time", "P585": "point in time",
    "P607": "conflict", "P674": "characters", "P676": "lyrics by",
    "P706": "located on terrain feature", "P710": "participant",
    "P737": "influenced by", "P740": "location of formation",
    "P749": "parent organization", "P800": "notable work", "P807": "separated from",
    "P840": "narrative location", "P937": "work location",
    "P1001": "applies to jurisdiction", "P1056": "product or material produced",
    "P1198": "unemployment rate", "P1336": "territory claimed by",
    "P1344": "participant of", "P1365": "replaces", "P1366": "replaced by",
    "P1376": "capital of", "P1412": "languages spoken, written or signed",
    "P1441": "present in work", "P3373": "sibling",
}


def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one Re-DocRED document to a GLiNER RE-training record; None if unusable."""
    sents = row.get("sents")
    if not isinstance(sents, list) or not sents:
        return None
    tokens, sent_starts = _flatten_sentences(sents)
    if not tokens:
        return None
    num_tokens = len(tokens)

    ner: List[List[Any]] = []
    cluster_rep_idx: Dict[int, int] = {}

    for cluster_idx, mentions in enumerate(row.get("vertexSet") or []):
        if not isinstance(mentions, list):
            continue
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            etype = mention.get("type")
            span = _mention_span(sent_starts, num_tokens, mention)
            if not isinstance(etype, str) or not etype.strip() or span is None:
                continue
            idx = len(ner)
            ner.append([span[0], span[1], etype.strip()])
            cluster_rep_idx.setdefault(cluster_idx, idx)

    if not ner:
        return None

    relations: List[List[Any]] = []
    for label in row.get("labels") or []:
        if not isinstance(label, dict):
            continue
        pid = label.get("r")
        if not pid:
            continue
        rname = _RELATION_TEXT.get(pid, pid)
        head_idx, tail_idx = cluster_rep_idx.get(label.get("h")), cluster_rep_idx.get(label.get("t"))
        if head_idx is None or tail_idx is None or head_idx == tail_idx:
            continue
        relations.append([head_idx, tail_idx, rname])

    return {"tokenized_text": tokens, "ner": ner, "relations": relations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"],
                        help="Which canonical split to convert.")
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--repo", default="tonytan48/Re-DocRED", help="HuggingFace dataset repo.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split)

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

    count = write_jsonl(_records(), args.out)
    print(f"wrote {count} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
