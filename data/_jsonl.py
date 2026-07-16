"""Shared JSONL read/write helpers for GLiNER data converters.

Mirrors GLiNER2's tools/data/_split.py: the normalization convention (NFKC
plus stray Unicode line-separator stripping, so records never fragment
across physical lines) and, for datasets whose source has no canonical
train/val/test split of its own, the same deterministic 80/10/10
``SplitWriter`` -- a seeded RNG draws one ``random()`` per record and
routes it by cumulative ratio, so re-running a converter with the same seed
reproduces the same partition.
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from pathlib import Path
from typing import Any, Dict, IO, Iterable, Sequence, Tuple

# Stray Unicode line/paragraph separators that json.dumps writes literally
# (they are >= U+0020, so not escaped) yet str.splitlines() treats as line
# breaks. Left in place they fragment a JSONL record across physical lines,
# breaking any splitlines()-based reader. Map them to a plain space.
_LINE_SEPARATORS = str.maketrans({"\x85": " ", " ": " ", " ": " "})


def clean_text(s: str) -> str:
    """NFKC-normalize a string and strip stray Unicode line separators."""
    return unicodedata.normalize("NFKC", s).translate(_LINE_SEPARATORS)


def normalize_record(obj: Any) -> Any:
    """Recursively NFKC-normalize every string in a nested JSON-like structure."""
    if isinstance(obj, str):
        return clean_text(obj)
    if isinstance(obj, dict):
        return {normalize_record(k): normalize_record(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_record(x) for x in obj]
    return obj


def dumps_record(record: Dict) -> str:
    """Serialize a record to one JSONL line: normalized, non-ASCII kept."""
    return json.dumps(normalize_record(record), ensure_ascii=False)


def write_jsonl(records: Iterable[Dict], out_path: Path) -> int:
    """Write records to out_path as normalized JSONL. Returns the count written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(dumps_record(record))
            fh.write("\n")
            n += 1
    return n


SPLIT_NAMES = ("train", "val", "test")


def parse_ratios(spec: str) -> Tuple[float, float, float]:
    """Parse a 'train,val,test' string and validate it sums to ~1.0."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--split-ratios needs 3 comma-separated values, got {spec!r}")
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--split-ratios must be numeric, got {spec!r}") from e
    if any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError(f"--split-ratios cannot contain negative values, got {vals}")
    if abs(sum(vals) - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(f"--split-ratios must sum to 1.0, got {sum(vals):.4f}")
    return vals  # type: ignore[return-value]


def add_split_args(parser: argparse.ArgumentParser) -> None:
    """Attach the standard --split-ratios / --split-seed flags to a parser."""
    parser.add_argument(
        "--split-ratios", type=parse_ratios, default=(0.8, 0.1, 0.1),
        help="Comma-separated train,val,test ratios (default: 0.8,0.1,0.1).",
    )
    parser.add_argument(
        "--split-seed", type=int, default=42,
        help="Random seed for the train/val/test partition (default: 42).",
    )


def derive_split_paths(base: Path) -> Dict[str, Path]:
    """Return {split: path} for the three sibling files.

    If ``base`` ends in ``.jsonl`` the suffix is stripped before appending
    the per-split suffix; otherwise ``base`` is used as-is.
    """
    stem = base.with_suffix("") if base.suffix == ".jsonl" else base
    return {s: Path(f"{stem}.{s}.jsonl") for s in SPLIT_NAMES}


class SplitWriter:
    """JSONL writer that routes each record into train/val/test deterministically.

    For datasets whose source has no canonical split of its own. Args:
        base: Output base path (``data/foo.jsonl`` or ``data/foo``).
        ratios: Three-tuple (train, val, test) summing to 1.0. Default (0.8, 0.1, 0.1).
        seed: Seed for the per-record routing RNG.
    """

    def __init__(self, base: Path, ratios: Sequence[float] = (0.8, 0.1, 0.1), seed: int = 42) -> None:
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"ratios must be a 3-tuple summing to 1.0, got {ratios!r}")
        self._paths: Dict[str, Path] = derive_split_paths(base)
        self._files: Dict[str, IO[str]] = {}
        self._counts: Dict[str, int] = {s: 0 for s in SPLIT_NAMES}
        # Cumulative thresholds, e.g. (0.8, 0.9, 1.0).
        self._cum = []
        acc = 0.0
        for r in ratios:
            acc += r
            self._cum.append(acc)
        self._rng = random.Random(seed)

    def __enter__(self) -> "SplitWriter":
        for name, path in self._paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._files[name] = path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for fh in self._files.values():
            fh.close()
        self._files = {}

    def _route(self) -> str:
        x = self._rng.random()
        for i, threshold in enumerate(self._cum):
            if x < threshold:
                return SPLIT_NAMES[i]
        return SPLIT_NAMES[-1]

    def write(self, record: Dict) -> str:
        """Write ``record`` to the chosen split and return the split name."""
        split = self._route()
        self._files[split].write(dumps_record(record) + "\n")
        self._counts[split] += 1
        return split

    @property
    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    @property
    def paths(self) -> Dict[str, Path]:
        return dict(self._paths)

    def summary(self) -> str:
        c = self._counts
        return (
            f"train={c['train']} val={c['val']} test={c['test']} "
            f"-> {self._paths['train']}, {self._paths['val']}, {self._paths['test']}"
        )


def write_jsonl_split(
    records: Iterable[Dict], out_base: Path, ratios: Sequence[float] = (0.8, 0.1, 0.1), seed: int = 42
) -> Dict[str, int]:
    """Drop-in split counterpart to ``write_jsonl``: route records into train/val/test.

    Returns the per-split counts written.
    """
    with SplitWriter(out_base, ratios=ratios, seed=seed) as writer:
        for record in records:
            writer.write(record)
        return writer.counts


def iter_jsonl(source: IO) -> Iterable[Dict]:
    """Yield parsed JSON objects from an open text-mode file-like, skipping blank/invalid lines."""
    for line in source:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
