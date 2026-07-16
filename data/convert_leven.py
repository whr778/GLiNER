"""Convert LEVEN (Yao et al., Findings of ACL 2022) to GLiNER event-training JSONL.

LEVEN is the largest Chinese event detection dataset (108 legal event
types) and, like MAVEN, is trigger-and-type detection only -- no argument
annotations, so every converted record has an empty ``relations`` list.
Schema is identical to MAVEN's (same "content"/"tokens"/"events"/"mention"
field names), so this reuses convert_maven.py's conversion logic directly
rather than duplicating it.

LEVEN's ``tokens`` are already word-segmented (not characters) -- GLiNER's
``tokenized_text`` uses them as-is, same as any other dataset; the only
Chinese-specific wrinkle (relevant to GLiNER2's own converter, which
reconstructs a single ``input`` text string by joining tokens with "" since
Chinese has no whitespace word boundaries) doesn't apply here, since GLiNER
keeps the token list rather than flattening to a text string.

Source layout (https://github.com/thunlp/LEVEN) -- same shape as MAVEN::

    {"content": [{"tokens": [...]}, ...],
     "events": [{"type": "...",
                 "mention": [{"sent_id": 0, "offset": [14, 15]}, ...]}]}

test.jsonl ships ``candidates`` instead of ``events`` (annotations
removed) and converts to nothing, so only train/valid are usable.

Usage::

    uv run python data/convert_leven.py \\
        --input /path/to/LEVEN/train.jsonl \\
        --out data/leven.train.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import iter_jsonl, write_jsonl  # noqa: E402
from convert_maven import convert_row  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True,
                        help="Path to a LEVEN jsonl file (train.jsonl / valid.jsonl).")
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
