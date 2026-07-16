"""Unit tests for the GLiNER event-dataset converters in data/.

These exercise each converter's pure conversion logic (convert_row /
parse_annotation / parse_apf) against small synthetic inputs shaped like
the real raw source schema -- hermetic, no network access and no
dependency on a locally-cached corpus. Each converter was additionally
run against real source data by hand during development (see the module
docstrings for what was checked); these tests keep that logic covered by
CI going forward without re-downloading anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import convert_cmnee  # noqa: E402
import convert_docee  # noqa: E402
import convert_events_biotech  # noqa: E402
import convert_ace2005  # noqa: E402
import convert_casie  # noqa: E402
from _trigger_types import derive_trigger_types  # noqa: E402


def _decode(tokens, ner, idx):
    s, e, label = ner[idx]
    return " ".join(tokens[s:e + 1]), label


class TestConvertCmnee:
    def test_converts_trigger_and_arguments_with_native_offsets(self):
        row = {
            "text": "俄罗斯海军总司令部对外表示试验",
            "event_list": [{
                "event_type": "Experiment",
                "trigger": {"text": "试验", "offset": [13, 15]},
                "arguments": [{"role": "Subject", "text": "俄罗斯海军总司令部", "offset": [0, 9]}],
            }],
        }
        rec = convert_cmnee.convert_row(row)
        assert rec is not None
        tokens, ner, relations = rec["tokenized_text"], rec["ner"], rec["relations"]
        assert tokens == list(row["text"])

        def _decode_cn(idx):
            s, e, label = ner[idx]
            return "".join(tokens[s:e + 1]), label

        assert _decode_cn(0) == ("试验", "Experiment")
        assert _decode_cn(1) == ("俄罗斯海军总司令部", "Subject")
        assert relations == [[0, 1, "Subject"]]

    def test_returns_none_when_no_events(self):
        assert convert_cmnee.convert_row({"text": "hello", "event_list": []}) is None

    def test_drops_out_of_bounds_offsets(self):
        row = {
            "text": "短文本",
            "event_list": [{
                "event_type": "X",
                "trigger": {"text": "短文本", "offset": [0, 99]},
                "arguments": [],
            }],
        }
        assert convert_cmnee.convert_row(row) is None


class TestConvertDocee:
    def test_prepends_synthetic_trigger_and_resolves_argument_offsets(self):
        rec_in = {
            "text": "Edelmiro Cavazos was the mayor. Edelmiro Cavazos died.",
            "event_type": "Famous Person - Death",
            "annotations": [
                {"start": 0, "end": 16, "type": "Deceased", "text": "Edelmiro Cavazos"},
                {"start": 33, "end": 49, "type": "Deceased", "text": "Edelmiro Cavazos"},
            ],
        }
        out = convert_docee.convert_row(rec_in)
        assert out is not None
        tokens, ner, relations = out["tokenized_text"], out["ner"], out["relations"]
        assert tokens[0] == "[Famous Person - Death]"
        assert ner[0] == [0, 0, "Famous Person - Death"]
        # Two distinct mentions of the same surface string resolve to two distinct spans.
        spans = {(s, e) for s, e, label in ner[1:]}
        assert len(spans) == 2
        assert len(relations) == 2
        assert all(r[0] == 0 and r[2] == "Deceased" for r in relations)

    def test_returns_none_without_any_resolved_argument(self):
        rec_in = {
            "text": "Some document with no matching annotation offsets.",
            "event_type": "X",
            "annotations": [{"start": 0, "end": 4, "type": "Role", "text": "nonexistent phrase"}],
        }
        assert convert_docee.convert_row(rec_in) is None

    def test_falls_back_to_surface_search_when_offset_mismatches(self):
        rec_in = {
            "text": "The victim was John Doe from the report.",
            "event_type": "X",
            # Offsets deliberately wrong; surface text is still findable.
            "annotations": [{"start": 0, "end": 3, "type": "Person", "text": "John Doe"}],
        }
        out = convert_docee.convert_row(rec_in)
        assert out is not None
        tokens, ner = out["tokenized_text"], out["ner"]
        assert _decode(tokens, ner, 1) == ("John Doe", "Person")


class TestConvertEventsBiotech:
    def test_emits_one_synthetic_trigger_per_true_label(self):
        row = {
            "input": "Acme raises Series B funding round.",
            "output": {"classifications": [{
                "task": "biotech_event",
                "labels": ["funding round", "m&a", "other"],
                "true_label": ["funding round", "m&a"],
            }]},
        }
        rec = convert_events_biotech.convert_row(row)
        assert rec is not None
        assert rec["relations"] == []
        assert rec["ner"] == [[0, 0, "funding round"], [1, 1, "m&a"]]
        assert rec["tokenized_text"][:2] == ["[funding round]", "[m&a]"]

    def test_returns_none_without_true_labels(self):
        row = {"input": "text", "output": {"classifications": [{"true_label": []}]}}
        assert convert_events_biotech.convert_row(row) is None


class TestConvertAce2005:
    SGM = "<DOC><TEXT>\nJohn Smith attacked the embassy in Baghdad yesterday.\n</TEXT></DOC>"
    APF = """<?xml version="1.0"?>
<source_file URI="doc1.sgm" SOURCE="newswire">
<document DOCID="doc1">
<entity ID="doc1-E1" TYPE="PER" SUBTYPE="Individual">
  <entity_mention ID="doc1-E1-1" TYPE="NAM">
    <extent><charseq START="0" END="9">John Smith</charseq></extent>
  </entity_mention>
</entity>
<entity ID="doc1-E2" TYPE="FAC" SUBTYPE="Building-Grounds">
  <entity_mention ID="doc1-E2-1" TYPE="NOM">
    <extent><charseq START="24" END="31">embassy</charseq></extent>
  </entity_mention>
</entity>
<event ID="doc1-EV1" TYPE="Conflict" SUBTYPE="Attack">
  <event_mention ID="doc1-EV1-1">
    <anchor><charseq START="11" END="18">attacked</charseq></anchor>
    <event_mention_argument REFID="doc1-E1-1" ROLE="Attacker">
      <extent><charseq START="0" END="9">John Smith</charseq></extent>
    </event_mention_argument>
    <event_mention_argument REFID="doc1-E2-1" ROLE="Target">
      <extent><charseq START="24" END="31">embassy</charseq></extent>
    </event_mention_argument>
    <event_mention_argument REFID="doc1-T1" ROLE="Time-Within">
      <extent><charseq START="43" END="52">yesterday</charseq></extent>
    </event_mention_argument>
  </event_mention>
</event>
</document>
</source_file>"""

    def _write_fixture(self, tmp_path):
        adj = tmp_path / "adj"
        adj.mkdir()
        (adj / "doc1.sgm").write_text(self.SGM, encoding="utf-8")
        (adj / "doc1.apf.xml").write_text(self.APF, encoding="utf-8")
        return adj / "doc1.apf.xml"

    def test_resolves_entity_typed_arguments_via_refid(self, tmp_path):
        apf_path = self._write_fixture(tmp_path)
        rec = convert_ace2005.parse_apf(apf_path, keep_subtypes=True)
        assert rec is not None
        tokens, ner, relations = rec["tokenized_text"], rec["ner"], rec["relations"]
        by_role = {role: _decode(tokens, ner, a) for _, a, role in relations}
        assert by_role["Attacker"] == ("John Smith", "PER.Individual")
        assert by_role["Target"] == ("embassy", "FAC.Building-Grounds")

    def test_falls_back_to_role_as_type_when_refid_unresolved(self, tmp_path):
        apf_path = self._write_fixture(tmp_path)
        rec = convert_ace2005.parse_apf(apf_path, keep_subtypes=True)
        tokens, ner, relations = rec["tokenized_text"], rec["ner"], rec["relations"]
        by_role = {role: _decode(tokens, ner, a) for _, a, role in relations}
        assert by_role["Time-Within"] == ("yesterday", "Time-Within")

    def test_trigger_uses_event_type_as_label(self, tmp_path):
        apf_path = self._write_fixture(tmp_path)
        rec = convert_ace2005.parse_apf(apf_path, keep_subtypes=True)
        tokens, ner, relations = rec["tokenized_text"], rec["ner"], rec["relations"]
        trigger_idx = relations[0][0]
        assert _decode(tokens, ner, trigger_idx) == ("attacked", "Conflict.Attack")

    def test_no_subtypes_strips_to_top_level_types(self, tmp_path):
        apf_path = self._write_fixture(tmp_path)
        rec = convert_ace2005.parse_apf(apf_path, keep_subtypes=False)
        labels = {label for _, _, label in rec["ner"]}
        assert "Conflict" in labels
        assert "PER" in labels
        assert "Conflict.Attack" not in labels

    def test_missing_sgm_returns_none(self, tmp_path):
        adj = tmp_path / "adj"
        adj.mkdir()
        apf_path = adj / "doc1.apf.xml"
        apf_path.write_text(self.APF, encoding="utf-8")
        assert convert_ace2005.parse_apf(apf_path, keep_subtypes=True) is None


class TestDeriveTriggerTypes:
    def test_derives_from_relation_heads(self):
        rec = {
            "ner": [[0, 0, "Manoeuvre"], [1, 2, "Subject"], [3, 3, "Date"]],
            "relations": [[0, 1, "Subject"], [0, 2, "Date"]],
        }
        assert derive_trigger_types([rec]) == {"Manoeuvre"}

    def test_bare_event_type_label_is_not_confused_with_a_role(self):
        # CMNEE trigger types are bare words like "Manoeuvre" -- a
        # shape-based heuristic (e.g. "contains a dot") would misclassify
        # this as an argument type; head-of-relation derivation gets it right.
        rec = {
            "ner": [[0, 0, "Manoeuvre"], [1, 1, "Subject"]],
            "relations": [[0, 1, "Subject"]],
        }
        trigger_types = derive_trigger_types([rec])
        assert "Manoeuvre" in trigger_types
        assert "Subject" not in trigger_types

    def test_explicit_override_for_relation_less_datasets(self):
        # MAVEN/LEVEN/events_biotech have no relations to derive from.
        rec = {"ner": [[0, 0, "Know"]], "relations": []}
        assert derive_trigger_types([rec]) == set()
        assert derive_trigger_types([rec], explicit={"Know", "Catastrophe"}) == {"Know", "Catastrophe"}

    def test_argument_less_trigger_type_is_silently_not_derived(self):
        # Known incompleteness (documented in docs/events.md and the module
        # docstring, not a bug): a real trigger type that happens to have no
        # linked argument in the given records never appears as a relation
        # head, so it's absent from the derived set even though it's a
        # legitimate trigger label present in `ner`. Sound (never wrong),
        # not complete -- callers with a known fixed vocabulary should pass
        # it explicitly instead of relying on derivation alone.
        rec = {
            "ner": [[0, 0, "Medical.Intervention.Unspecified"], [1, 1, "Conflict.Attack"], [2, 2, "Victim"]],
            "relations": [[1, 2, "Victim"]],
        }
        derived = derive_trigger_types([rec])
        assert derived == {"Conflict.Attack"}
        assert "Medical.Intervention.Unspecified" not in derived


class TestConvertCasie:
    def test_offset_based_matching_disambiguates_repeated_surface_text(self, tmp_path):
        text = "Attacker stole data. Later, attacker fled with data again."
        annotation = {
            "content": text,
            "cyberevent": {"hopper": [{"events": [
                {
                    "subtype": "Databreach",
                    "nugget": {"text": "stole", "startOffset": 9, "endOffset": 14},
                    "argument": [
                        {"text": "data", "type": "Data", "startOffset": 15, "endOffset": 19,
                         "role": {"type": "Compromised-Data"}},
                    ],
                },
                {
                    "subtype": "Databreach",
                    "nugget": {"text": "fled", "startOffset": 38, "endOffset": 42},
                    "argument": [
                        {"text": "data", "type": "Data", "startOffset": 48, "endOffset": 52,
                         "role": {"type": "Compromised-Data"}},
                    ],
                },
            ]}]},
        }
        ann_path = tmp_path / "doc1.json"
        import json
        ann_path.write_text(json.dumps(annotation), encoding="utf-8")

        rec = convert_casie.parse_annotation(ann_path, prefix_event=True)
        assert rec is not None
        tokens, ner, relations = rec["tokenized_text"], rec["ner"], rec["relations"]
        assert len(relations) == 2
        # Both "data" arguments resolve to their own distinct positions, not both to the first.
        arg_spans = {(ner[a][0], ner[a][1]) for _, a, _ in relations}
        assert len(arg_spans) == 2

    def test_unusable_annotation_returns_none(self, tmp_path):
        import json
        ann_path = tmp_path / "doc1.json"
        ann_path.write_text(json.dumps({"content": ""}), encoding="utf-8")
        assert convert_casie.parse_annotation(ann_path, prefix_event=True) is None
