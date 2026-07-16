"""Split an already-converted GLiNER JSONL file into train/val/test.

For datasets already converted to GLiNER schema (e.g. from before the
SplitWriter mechanism existed, or a converter whose raw source needs
manual preparation and isn't handy to re-run) that don't yet have a
train/val/test split. Wraps the same SplitWriter/write_jsonl_split
mechanism data/convert_*.py converters use internally -- see data/_jsonl.py.

Usage::

    uv run python data/split_jsonl.py --input data/events_biotech.jsonl --out data/events_biotech.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jsonl import add_split_args, iter_jsonl, write_jsonl_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="Already-converted GLiNER JSONL to split.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output base path; produces <base>.train/.val/.test.jsonl.")
    add_split_args(parser)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as fh:
        records = list(iter_jsonl(fh))

    counts = write_jsonl_split(records, args.out, ratios=args.split_ratios, seed=args.split_seed)
    print(f"read {len(records)} records from {args.input}")
    print(f"wrote train={counts['train']} val={counts['val']} test={counts['test']} "
          f"-> {args.out} (.train/.val/.test.jsonl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
