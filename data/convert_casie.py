"""Convert CASIE (Satyapanich et al., AAAI 2020) to GLiNER event-training JSONL.

CASIE is a 1,000-document cybersecurity-event corpus with 5 event subtypes
(``Databreach``, ``Phishing``, ``Ransom``, ``Vulnerability-Discover``,
``Vulnerability-Patch``) and typed arguments. Each argument carries both an
event-role label (``Compromised-Data``, ``Attacker``, ``Place``, ...) and an
independent entity ``type`` (``PII``, ``Person``, ``Organization``,
``Device``, ...) -- so, unlike RAMS, the role name and the argument's NER
label are genuinely different things here; the argument's ``ner`` entry uses
``type`` and the role is carried separately in ``relations``.

Unlike GLiNER2's own converter (which does substring search: ``trigger not
in text``), this reads the native ``startOffset``/``endOffset`` char spans
that CASIE ships on both the trigger ``nugget`` and every ``argument``, and
maps them to word-token spans via ``data/_charspan.py``. This is required
for correctness, not just precision -- CASIE documents frequently repeat a
trigger/argument surface string at multiple positions, so a bare substring
search (as GLiNER2's converter does) resolves to the *first* occurrence
only, silently mislabeling every other real occurrence.

Source layout (https://github.com/Ebiquity/CASIE)::

    data/annotation/<doc_id>.json:
        content: clean document body (matched to char offsets)
        cyberevent.hopper[].events[]:
            nugget: {startOffset, endOffset, text}      -- trigger
            subtype: "Databreach" | "Phishing" | ...    -- event type
            argument: [{startOffset, endOffset, text, type, role: {type}}]

Output (one record per source document)::

    {"tokenized_text": [...],
     "ner": [[12, 12, "Cyber.Databreach"], [4, 6, "PII"], ...],
     "relations": [[<trigger_ner_idx>, <arg_ner_idx>, "Compromised-Data"], ...]}

No manual download step -- the converter fetches the ~10 MB tarball from
GitHub once, unless ``--input`` points at an already-extracted repo root.

Usage::

    uv run python data/convert_casie.py --out data/casie.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _charspan import char_span_to_token_span, tokenize_with_offsets  # noqa: E402
from _jsonl import add_split_args, write_jsonl_split  # noqa: E402

CASIE_TARBALL_URL = "https://github.com/Ebiquity/CASIE/archive/refs/heads/master.tar.gz"


def _download_tarball(url: str, dest: Path) -> None:
    """Fetch the CASIE tarball into ``dest`` and extract it."""
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    print(f"  downloaded {len(raw) / 1e6:.1f} MB, extracting...")
    tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    tf.extractall(dest)


def _find_data_root(extract_root: Path) -> Optional[Path]:
    """Locate the ``data/`` directory inside the extracted tarball."""
    for child in extract_root.iterdir():
        candidate = child / "data"
        if candidate.is_dir():
            return candidate
    return None


def _offset_span(obj: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Read a ``{startOffset, endOffset}`` pair; end is treated as exclusive."""
    s, e = obj.get("startOffset"), obj.get("endOffset")
    if not isinstance(s, int) or not isinstance(e, int) or s >= e:
        return None
    return s, e


def parse_annotation(annotation_path: Path, prefix_event: bool) -> Optional[Dict[str, Any]]:
    """Parse one CASIE annotation JSON into a GLiNER event-training record; None if unusable."""
    try:
        with annotation_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    text = data.get("content")
    if not isinstance(text, str) or not text.strip():
        return None
    tokens, offsets = tokenize_with_offsets(text)
    if not tokens:
        return None
    num_tokens = len(tokens)

    ner: List[List[Any]] = []
    relations: List[List[Any]] = []
    arg_span_to_ner_idx: Dict[Tuple[int, int, str], int] = {}

    cyberevent = data.get("cyberevent") or {}
    for hopper in cyberevent.get("hopper") or []:
        if not isinstance(hopper, dict):
            continue
        for ev in hopper.get("events") or []:
            if not isinstance(ev, dict):
                continue
            subtype = ev.get("subtype")
            nugget = ev.get("nugget") or {}
            if not isinstance(subtype, str) or not isinstance(nugget, dict):
                continue
            subtype = subtype.strip()
            char_span = _offset_span(nugget)
            if not subtype or char_span is None:
                continue
            tok_span = char_span_to_token_span(offsets, *char_span)
            if tok_span is None or not (0 <= tok_span[1] < num_tokens):
                continue

            event_type = f"Cyber.{subtype}" if prefix_event else subtype
            trigger_idx = len(ner)
            ner.append([tok_span[0], tok_span[1], event_type])

            for arg in ev.get("argument") or []:
                if not isinstance(arg, dict):
                    continue
                ent_type = arg.get("type")
                role_obj = arg.get("role") or {}
                role = role_obj.get("type") if isinstance(role_obj, dict) else None
                arg_char_span = _offset_span(arg)
                if not isinstance(ent_type, str) or not isinstance(role, str) or arg_char_span is None:
                    continue
                ent_type, role = ent_type.strip(), role.strip()
                if not ent_type or not role:
                    continue
                arg_tok_span = char_span_to_token_span(offsets, *arg_char_span)
                if arg_tok_span is None or not (0 <= arg_tok_span[1] < num_tokens):
                    continue

                arg_key = (arg_tok_span[0], arg_tok_span[1], ent_type)
                arg_idx = arg_span_to_ner_idx.get(arg_key)
                if arg_idx is None:
                    arg_idx = len(ner)
                    ner.append([arg_tok_span[0], arg_tok_span[1], ent_type])
                    arg_span_to_ner_idx[arg_key] = arg_idx

                relations.append([trigger_idx, arg_idx, role])

    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner, "relations": relations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="Output GLiNER JSONL path.")
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to a pre-extracted CASIE repo root "
                             "(containing data/annotation/). If omitted, "
                             "the tarball is downloaded from GitHub.")
    parser.add_argument("--url", default=CASIE_TARBALL_URL,
                        help="Tarball URL when --input is not provided.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Stop after this many usable records (-1 = all).")
    parser.add_argument("--no-prefix-event", action="store_true",
                        help="Keep event types as bare 'Databreach' / 'Phishing' / "
                             "... instead of prefixing with 'Cyber.'.")
    add_split_args(parser)
    args = parser.parse_args()

    if args.input is not None:
        if not args.input.is_dir():
            raise SystemExit(f"--input not a directory: {args.input}")
        data_root = args.input / "data" if (args.input / "data").is_dir() else args.input
        annotation_dir = data_root / "annotation"
    else:
        tmp_root = Path(tempfile.mkdtemp(prefix="casie_"))
        _download_tarball(args.url, tmp_root)
        data_root = _find_data_root(tmp_root)
        if data_root is None:
            raise SystemExit(f"no data/ directory found under {tmp_root}")
        annotation_dir = data_root / "annotation"

    if not annotation_dir.is_dir():
        raise SystemExit(f"could not find data/annotation/ under {annotation_dir.parent}")

    def _records():
        n = 0
        for ann_path in sorted(annotation_dir.glob("*.json")):
            rec = parse_annotation(ann_path, prefix_event=not args.no_prefix_event)
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
