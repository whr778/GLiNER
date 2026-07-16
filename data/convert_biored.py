"""Convert BioRED (biomedical NER + relations) to GLiNER RE-training JSONL.

Canonical source: https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip (~2 MB,
read from the BioC.JSON inside). Pass ``--zip`` to use a local copy.

Standard BioC format: a document is a list of ``passages``, each with its
own ``offset`` (absolute char position within the document) and ``text``;
each passage's ``annotations`` carry ``locations: [{offset, length}]`` that
are themselves **absolute document offsets**, not passage-relative (verified
against a real downloaded document -- passage 2's own offset is 159, and its
first annotation's location offset 184 slices correctly against passage 2's
own text only after subtracting the passage offset, confirming both are on
the same absolute scale).

This reconstructs one full document text by placing each passage's text at
its own absolute offset (gaps between passages, if any, filled with spaces
to keep every other offset valid), then maps every annotation's absolute
char span to a token span via ``data/_charspan.py`` -- unlike GLiNER2's
converter, which discards offsets and works from surface strings only.

Entity types: GeneOrGeneProduct, DiseaseOrPhenotypicFeature, ChemicalEntity,
SequenceVariant, CellLine, OrganismTaxon. Relations are document-level and
link normalized entity *identifiers* (Association, Positive_Correlation,
...), not specific mention positions -- so, like convert_docred.py, each
relation is anchored to a representative mention (the first-seen span for
that identifier) rather than a specific occurrence.

License: NCBI / U.S. National Library of Medicine (see the bundled
README.txt in the zip).

The optional ``data`` dependency group is *not* needed here (no
HuggingFace datasets involved) -- only stdlib urllib/zipfile.

Usage::

    uv run python data/convert_biored.py --out data/biored.train.jsonl --split train
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _charspan import char_span_to_token_span, tokenize_with_offsets  # noqa: E402
from _jsonl import write_jsonl  # noqa: E402

URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip"
MEMBER = {"train": "BioRED/Train.BioC.JSON", "dev": "BioRED/Dev.BioC.JSON", "test": "BioRED/Test.BioC.JSON"}


def _reconstruct_text(passages: List[Dict[str, Any]]) -> str:
    """Place each passage's text at its own absolute offset; fill gaps with spaces."""
    pieces = []
    for p in passages:
        offset, text = p.get("offset"), p.get("text") or ""
        if isinstance(offset, int) and text:
            pieces.append((offset, text))
    if not pieces:
        return ""
    end = max(off + len(text) for off, text in pieces)
    buf = [" "] * end
    for offset, text in pieces:
        buf[offset:offset + len(text)] = text
    return "".join(buf)


def convert_doc(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one BioRED document to a GLiNER RE-training record; None if unusable."""
    passages = doc.get("passages") or []
    text = _reconstruct_text(passages)
    if not text.strip():
        return None
    tokens, offsets = tokenize_with_offsets(text)
    if not tokens:
        return None
    num_tokens = len(tokens)

    ner: List[List[Any]] = []
    id_to_rep_idx: Dict[str, int] = {}  # normalized identifier -> representative (first-seen) ner idx

    for passage in passages:
        for ann in passage.get("annotations", []):
            infons = ann.get("infons") or {}
            typ, ident = infons.get("type"), infons.get("identifier")
            locations = ann.get("locations") or []
            if not isinstance(typ, str) or not locations:
                continue
            loc = locations[0]
            char_start, length = loc.get("offset"), loc.get("length")
            if not isinstance(char_start, int) or not isinstance(length, int) or length <= 0:
                continue
            tok_span = char_span_to_token_span(offsets, char_start, char_start + length)
            if tok_span is None or not (0 <= tok_span[1] < num_tokens):
                continue
            idx = len(ner)
            ner.append([tok_span[0], tok_span[1], typ])
            if isinstance(ident, str) and ident:
                id_to_rep_idx.setdefault(ident, idx)

    if not ner:
        return None

    relations: List[List[Any]] = []
    for rel in doc.get("relations", []):
        infons = rel.get("infons") or {}
        typ = infons.get("type")
        if not isinstance(typ, str):
            continue
        head_idx = id_to_rep_idx.get(infons.get("entity1"))
        tail_idx = id_to_rep_idx.get(infons.get("entity2"))
        if head_idx is None or tail_idx is None or head_idx == tail_idx:
            continue
        relations.append([head_idx, tail_idx, typ])

    return {"tokenized_text": tokens, "ner": ner, "relations": relations}


def _load_documents(args) -> list:
    if args.zip:
        data = Path(args.zip).read_bytes()
    else:
        print(f"Downloading {URL} ...")
        with urllib.request.urlopen(URL, timeout=180) as resp:
            data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open(MEMBER[args.split]) as f:
            return json.load(f)["documents"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--zip", default=None, help="Local BIORED.zip (skips the download).")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    args = parser.parse_args()

    docs = _load_documents(args)

    def _records():
        n = 0
        for doc in docs:
            rec = convert_doc(doc)
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
