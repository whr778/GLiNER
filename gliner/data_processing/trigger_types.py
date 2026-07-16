"""Derive the trigger-type label set for event-extraction training.

``EventExtractionSpanProcessor`` needs an explicit ``trigger_types`` set to
split each record's ``ner`` labels into triggers vs. arguments
(``trigger_class_mask``) -- the converter output alone doesn't self-describe
which labels are which. Without it, every label is treated as an argument, no
triggers are ever selected, and the event head silently produces no events
(NER still trains). So event training must resolve ``trigger_types`` before
building the model; ``derive_trigger_types_if_needed`` is the wiring hook the
training scripts use for that.

A label-shape heuristic (e.g. "contains a dot") is a trap: it happens to hold
for WikiEvents (``Life.Die.Unspecified``) but is wrong for CMNEE, whose
trigger types are bare words (``Manoeuvre``) matching the same shape as some
argument roles.

The robust, dataset-agnostic rule: a record's relations always point
trigger -> argument (``[trigger_ner_idx, arg_ner_idx, role]``, see every
convert_*.py under ``data/``), so a *sound* trigger-type vocabulary is the set
of ``ner`` labels that ever appear as a relation *head*. This never
misclassifies an argument type as a trigger for any converter that has both
triggers and relations (WikiEvents, RAMS, CASIE, CMNEE, ACE2005) -- head and
tail label sets are disjoint by construction (event types vs. entity/role
types).

It is not guaranteed *complete*: an event type that never co-occurs with any
argument in the records derived from won't appear as a relation head and is
silently dropped. Trigger-only datasets with no relations at all (MAVEN,
LEVEN; events_biotech -- synthetic trigger-per-label, no arguments) have
nothing to derive from; set their ``trigger_types`` explicitly in the config.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set


def derive_trigger_types(records: Iterable[Dict[str, Any]], explicit: Optional[Iterable[str]] = None) -> Set[str]:
    """Return the trigger-type label set for a batch of converted records.

    With ``explicit`` given, returns ``set(explicit)`` unchanged (use this
    for relation-less datasets: MAVEN, LEVEN, events_biotech). Otherwise
    derives it as the set of ``ner`` labels appearing as a relation head.
    """
    if explicit is not None:
        return set(explicit)
    trigger_types: Set[str] = set()
    for rec in records:
        ner = rec.get("ner") or []
        for trigger_idx, _arg_idx, _role in rec.get("relations") or []:
            trigger_types.add(ner[trigger_idx][2])
    return trigger_types


def derive_trigger_types_if_needed(
    event_mode: bool, trigger_types: Optional[Iterable[str]], records: Iterable[Dict[str, Any]]
) -> Optional[List[str]]:
    """Resolve ``trigger_types`` for an event model, deriving when unset.

    Returns the sorted trigger-type list to apply, or ``None`` when nothing
    needs to change (not an event model, or ``trigger_types`` already set).
    Callers assign the returned list onto their own config object.

    Raises:
        ValueError: If this is an event model with no ``trigger_types`` set
            and none could be derived from ``records`` (e.g. a relation-less
            trigger-only dataset) -- set ``trigger_types`` in the config.
    """
    if not event_mode or trigger_types:
        return None
    derived = sorted(derive_trigger_types(records))
    if not derived:
        raise ValueError(
            "event_mode is set but trigger_types is empty and none could be derived "
            "from the training data's relation heads (a trigger-only dataset with no "
            "relations, e.g. MAVEN/LEVEN). Set trigger_types explicitly in the config."
        )
    return derived


def apply_derived_trigger_types(config, records) -> Optional[List[str]]:
    """Set derived ``trigger_types`` on an event training config, in place.

    The single wiring hook shared by ``train.py`` and ``scripts/custom_train.py``
    so both derive identically. ``config`` may be a dict (train.py's model_cfg)
    or an attribute namespace (custom_train.py's flattened config). Reads
    ``event_mode``/``trigger_types`` off it, and when it's an event config with
    none set, derives them from ``records`` and writes them back. No-op (returns
    ``None``) for non-event configs or ones that already set ``trigger_types``;
    otherwise returns the applied list.
    """
    is_dict = isinstance(config, dict)
    event_mode = config.get("event_mode", False) if is_dict else getattr(config, "event_mode", False)
    current = config.get("trigger_types") if is_dict else getattr(config, "trigger_types", None)

    derived = derive_trigger_types_if_needed(event_mode, current, records)
    if derived is None:
        return None

    if is_dict:
        config["trigger_types"] = derived
    else:
        config.trigger_types = derived
    print(f"Derived {len(derived)} trigger types from training data: {derived}")
    return derived
