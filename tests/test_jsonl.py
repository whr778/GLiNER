"""Unit tests for data/_jsonl.py, focused on the SplitWriter mechanism.

Hermetic -- no network access, no real dataset needed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import pytest

from _jsonl import (  # noqa: E402
    SplitWriter,
    derive_split_paths,
    parse_ratios,
    write_jsonl_split,
)


class TestDeriveSplitPaths:
    def test_strips_jsonl_suffix_before_appending(self, tmp_path):
        paths = derive_split_paths(tmp_path / "foo.jsonl")
        assert paths["train"] == tmp_path / "foo.train.jsonl"
        assert paths["val"] == tmp_path / "foo.val.jsonl"
        assert paths["test"] == tmp_path / "foo.test.jsonl"

    def test_appends_directly_when_no_jsonl_suffix(self, tmp_path):
        paths = derive_split_paths(tmp_path / "foo")
        assert paths["train"] == tmp_path / "foo.train.jsonl"


class TestParseRatios:
    def test_parses_valid_spec(self):
        assert parse_ratios("0.8,0.1,0.1") == (0.8, 0.1, 0.1)

    def test_rejects_wrong_count(self):
        with pytest.raises(Exception):
            parse_ratios("0.8,0.2")

    def test_rejects_non_numeric(self):
        with pytest.raises(Exception):
            parse_ratios("a,b,c")

    def test_rejects_negative(self):
        with pytest.raises(Exception):
            parse_ratios("-0.1,0.6,0.5")

    def test_rejects_sum_not_one(self):
        with pytest.raises(Exception):
            parse_ratios("0.5,0.2,0.2")


class TestSplitWriter:
    def test_every_record_written_exactly_once(self, tmp_path):
        records = [{"id": i} for i in range(200)]
        with SplitWriter(tmp_path / "data.jsonl", seed=1) as writer:
            for r in records:
                writer.write(r)

        seen_ids = set()
        for path in writer.paths.values():
            with path.open() as fh:
                for line in fh:
                    seen_ids.add(json.loads(line)["id"])
        assert seen_ids == {r["id"] for r in records}
        assert sum(writer.counts.values()) == 200

    def test_ratios_are_approximately_respected(self, tmp_path):
        records = [{"id": i} for i in range(5000)]
        with SplitWriter(tmp_path / "data.jsonl", ratios=(0.8, 0.1, 0.1), seed=1) as writer:
            for r in records:
                writer.write(r)

        counts = writer.counts
        assert 3850 < counts["train"] < 4150
        assert 350 < counts["val"] < 650
        assert 350 < counts["test"] < 650

    def test_same_seed_is_deterministic(self, tmp_path):
        records = [{"id": i} for i in range(300)]

        with SplitWriter(tmp_path / "a.jsonl", seed=7) as writer:
            for r in records:
                writer.write(r)
        train_ids_1 = _read_ids(writer.paths["train"])

        with SplitWriter(tmp_path / "b.jsonl", seed=7) as writer:
            for r in records:
                writer.write(r)
        train_ids_2 = _read_ids(writer.paths["train"])

        assert train_ids_1 == train_ids_2

    def test_different_seed_gives_different_split(self, tmp_path):
        records = [{"id": i} for i in range(300)]

        with SplitWriter(tmp_path / "a.jsonl", seed=1) as writer:
            for r in records:
                writer.write(r)
        train_ids_1 = _read_ids(writer.paths["train"])

        with SplitWriter(tmp_path / "b.jsonl", seed=2) as writer:
            for r in records:
                writer.write(r)
        train_ids_2 = _read_ids(writer.paths["train"])

        assert train_ids_1 != train_ids_2

    def test_rejects_ratios_not_summing_to_one(self, tmp_path):
        with pytest.raises(ValueError):
            SplitWriter(tmp_path / "data.jsonl", ratios=(0.5, 0.2, 0.2))

    def test_records_are_normalized_like_write_jsonl(self, tmp_path):
        with SplitWriter(tmp_path / "data.jsonl", seed=1) as writer:
            writer.write({"text": "café line"})

        for path in writer.paths.values():
            if path.stat().st_size:
                content = path.read_text(encoding="utf-8")
                assert " " not in content


class TestWriteJsonlSplit:
    def test_returns_per_split_counts(self, tmp_path):
        records = ({"id": i} for i in range(100))
        counts = write_jsonl_split(records, tmp_path / "data.jsonl", seed=1)
        assert set(counts) == {"train", "val", "test"}
        assert sum(counts.values()) == 100


def _read_ids(path: Path) -> set:
    ids = set()
    if not path.exists():
        return ids
    with path.open() as fh:
        for line in fh:
            ids.add(json.loads(line)["id"])
    return ids
