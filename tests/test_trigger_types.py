"""Tests for event trigger-type derivation and the training-time wiring.

The event head splits each record's labels into triggers vs. arguments via
``config.trigger_types``. The shipped event configs leave it empty, and the
training scripts must derive it from the data -- otherwise every label is
treated as an argument, no triggers are selected, and the event head silently
produces zero events (while NER still trains). These tests pin that wiring.
"""
import argparse
import sys
from pathlib import Path

import pytest

from gliner.data_processing.trigger_types import (
    apply_derived_trigger_types,
    derive_trigger_types,
    derive_trigger_types_if_needed,
)

# Relations point trigger -> argument, so trigger types are the labels that
# appear as a relation head: Attack and Die below.
RECORDS = [
    {"ner": [[0, 0, "Person"], [1, 1, "Attack"], [2, 2, "Location"]],
     "relations": [[1, 0, "Attacker"], [1, 2, "Place"]]},
    {"ner": [[1, 1, "Die"], [2, 2, "Person"]],
     "relations": [[0, 1, "Victim"]]},
]


def test_derive_from_relation_heads():
    assert derive_trigger_types(RECORDS) == {"Attack", "Die"}


def test_derive_explicit_passthrough():
    assert derive_trigger_types(RECORDS, explicit=["X", "Y"]) == {"X", "Y"}


def test_derive_tolerates_records_without_relations():
    # A NER-only record contributes no trigger heads and must not KeyError.
    assert derive_trigger_types([{"ner": [[0, 0, "Person"]]}]) == set()


def test_if_needed_derives_when_event_and_empty():
    assert derive_trigger_types_if_needed(True, [], RECORDS) == ["Attack", "Die"]
    assert derive_trigger_types_if_needed(True, None, RECORDS) == ["Attack", "Die"]


def test_if_needed_is_noop_when_already_set():
    assert derive_trigger_types_if_needed(True, ["Attack"], RECORDS) is None


def test_if_needed_is_noop_when_not_event_mode():
    assert derive_trigger_types_if_needed(False, [], RECORDS) is None


def test_if_needed_raises_on_relationless_event_dataset():
    with pytest.raises(ValueError, match="trigger_types"):
        derive_trigger_types_if_needed(True, [], [{"ner": [[0, 0, "A"]], "relations": []}])


# --- apply_derived_trigger_types: the exact wiring both training scripts run.
# train.py passes a dict (model_cfg); custom_train.py passes a flattened
# argparse.Namespace (self.config). Test both shapes against the real function.

def test_apply_mutates_dict_config_like_train_py():
    model_cfg = {"event_mode": True, "trigger_types": []}
    applied = apply_derived_trigger_types(model_cfg, RECORDS)
    assert applied == ["Attack", "Die"]
    assert model_cfg["trigger_types"] == ["Attack", "Die"]


def test_apply_mutates_namespace_config_like_custom_train_py():
    config = argparse.Namespace(event_mode=True, trigger_types=[])
    applied = apply_derived_trigger_types(config, RECORDS)
    assert applied == ["Attack", "Die"]
    assert config.trigger_types == ["Attack", "Die"]


def test_apply_is_noop_when_trigger_types_already_set():
    model_cfg = {"event_mode": True, "trigger_types": ["Custom"]}
    assert apply_derived_trigger_types(model_cfg, RECORDS) is None
    assert model_cfg["trigger_types"] == ["Custom"]  # left untouched


def test_apply_is_noop_for_non_event_config():
    model_cfg = {"event_mode": False, "trigger_types": []}
    assert apply_derived_trigger_types(model_cfg, RECORDS) is None
    assert model_cfg["trigger_types"] == []


def test_apply_raises_on_relationless_event_dataset():
    with pytest.raises(ValueError, match="trigger_types"):
        apply_derived_trigger_types(
            {"event_mode": True, "trigger_types": []}, [{"ner": [[0, 0, "A"]], "relations": []}]
        )


def test_data_shim_reexport_still_importable():
    """data/_trigger_types.py is imported by the converters and older tests via
    a sys.path insert; it must keep re-exporting derive_trigger_types."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from _trigger_types import derive_trigger_types as shim
    assert shim(RECORDS) == {"Attack", "Die"}


def test_train_main_derives_trigger_types_end_to_end(tmp_path, monkeypatch):
    """Execution-level guard for the wiring itself: driving train.main() on the
    event fixture must leave the built model with non-empty trigger_types. If
    the apply_derived_trigger_types call is removed from train.py, the model is
    built with trigger_types=[] and this fails. Training is monkeypatched to a
    no-op so only the config-load -> derive -> build_model path actually runs."""
    yaml = pytest.importorskip("yaml")
    repo_root = Path(__file__).resolve().parents[1]
    fixture = repo_root / "tests" / "fixtures" / "wikievents_sample.jsonl"
    if not fixture.exists():
        pytest.skip("wikievents fixture not available")

    sys.path.insert(0, str(repo_root))
    import train as train_script
    from gliner.model import BaseGLiNER

    cfg = yaml.safe_load((repo_root / "configs" / "wikievents.yaml").read_text())
    cfg["model"]["model_name"] = "prajjwal1/bert-tiny"
    cfg["model"]["hidden_size"] = 128
    cfg["model"]["trigger_types"] = []
    cfg["data"]["root_dir"] = str(tmp_path / "out")
    cfg["data"]["train_data"] = str(fixture)
    cfg["data"]["val_data"] = None
    cfg["data"]["test_data"] = None
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    captured = {}

    def fake_train_model(self, *args, **kwargs):
        captured["trigger_types"] = list(getattr(self.config, "trigger_types", []) or [])
        return None

    monkeypatch.setattr(BaseGLiNER, "train_model", fake_train_model)

    train_script.main(str(cfg_path))

    assert captured.get("trigger_types"), "train.main() built the model without deriving trigger_types"
    # WikiEvents trigger labels are dotted event types (e.g. Life.Die.Unspecified).
    assert all("." in t for t in captured["trigger_types"])
