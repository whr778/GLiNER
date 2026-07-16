"""Generate a Markdown model card (``README.md``) for a trained GLiNER model.

GLiNER's ``save_pretrained`` writes only weights + config + tokenizer, so a saved
model carries no description of what it does or what it was trained on. The
training scripts call :func:`write_model_card` alongside each best-model save to
drop a human-readable card into the checkpoint directory, covering the model's
purpose, its label schema, training-data metrics, hyperparameters, and best
validation F1.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_MAX_TYPES_SHOWN = 40


def summarize_training_data(records: Optional[List[Dict]]) -> Optional[Dict[str, Any]]:
    """Summarize training records into card-ready metrics, or None if empty."""
    if not records:
        return None
    ent_counts: Dict[str, int] = {}
    rel_counts: Dict[str, int] = {}
    token_counts: List[int] = []
    for rec in records:
        token_counts.append(len(rec.get("tokenized_text") or []))
        for ent in rec.get("ner") or []:
            ent_counts[ent[2]] = ent_counts.get(ent[2], 0) + 1
        for rel in rec.get("relations") or []:
            rel_counts[rel[2]] = rel_counts.get(rel[2], 0) + 1
    return {
        "num_documents": len(records),
        "num_entity_mentions": sum(ent_counts.values()),
        "num_relations": sum(rel_counts.values()),
        "entity_types": sorted(ent_counts),
        "relation_types": sorted(rel_counts),
        "tokens_min": min(token_counts, default=0),
        "tokens_max": max(token_counts, default=0),
        "tokens_mean": round(statistics.mean(token_counts), 1) if token_counts else 0,
    }


def _task(config) -> str:
    if getattr(config, "event_mode", False):
        return "event-extraction"
    if getattr(config, "relations_layer", None) not in (None, "none"):
        return "relation-extraction"
    return "named-entity-recognition"


def _purpose(task: str) -> str:
    if task == "event-extraction":
        return (
            "This is a **GLiNER event-extraction** model. It detects event *triggers* "
            "(spans that signal an event happened) and links each trigger to its "
            "*argument fillers* with typed semantic roles."
        )
    if task == "relation-extraction":
        return (
            "This is a **GLiNER entity-and-relation-extraction** model. It detects "
            "entity spans and the typed relations between them."
        )
    return (
        "This is a **GLiNER named-entity-recognition** model. It detects and labels "
        "entity spans in text."
    )


def _type_line(label: str, types: List[str]) -> str:
    shown = types[:_MAX_TYPES_SHOWN]
    extra = len(types) - len(shown)
    listed = ", ".join(f"`{t}`" for t in shown)
    if extra > 0:
        listed += f" … (+{extra} more)"
    return f"- **{label} ({len(types)}):** {listed}\n"


def _label_schema(config, data_stats: Optional[Dict[str, Any]]) -> str:
    ent_types = list(data_stats["entity_types"]) if data_stats else []
    rel_types = list(data_stats["relation_types"]) if data_stats else []
    task = _task(config)
    out = "## Label schema\n\n"
    if task == "event-extraction":
        triggers = sorted(getattr(config, "trigger_types", None) or [])
        args = [t for t in ent_types if t not in set(triggers)]
        out += _type_line("Event trigger types", triggers)
        out += _type_line("Argument (entity) types", args)
        out += _type_line("Argument role types", rel_types)
    elif task == "relation-extraction":
        out += _type_line("Entity types", ent_types)
        out += _type_line("Relation types", rel_types)
    else:
        out += _type_line("Entity types", ent_types)
    return out


def _training_data_section(data_stats: Optional[Dict[str, Any]]) -> str:
    if not data_stats:
        return ""
    s = data_stats
    return (
        "## Training data\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| Documents | {s['num_documents']:,} |\n"
        f"| Entity/trigger mentions | {s['num_entity_mentions']:,} |\n"
        f"| Relations / roles | {s['num_relations']:,} |\n"
        f"| Distinct entity types | {len(s['entity_types'])} |\n"
        f"| Distinct relation types | {len(s['relation_types'])} |\n"
        f"| Tokens/doc (min / mean / max) | {s['tokens_min']} / {s['tokens_mean']} / {s['tokens_max']} |\n"
    )


def _config_section(config) -> str:
    rows = []
    for field in ("max_len", "max_width", "max_types", "span_mode", "relations_layer", "triples_layer"):
        val = getattr(config, field, None)
        if val is not None:
            rows.append(f"| {field} | {val} |\n")
    if not rows:
        return ""
    return "## Training configuration\n\n| Setting | Value |\n|---|---|\n" + "".join(rows)


def _usage_section(config) -> str:
    if _task(config) == "named-entity-recognition":
        call = 'entities = model.predict_entities(text, labels=["person", "organization"])'
    else:
        call = (
            'entities, relations = model.predict_relations(\n'
            '    text, labels=["person", "organization"], relations=["works_at"]\n'
            ')'
        )
    return (
        "## Usage\n\n```python\nfrom gliner import GLiNER\n\n"
        'model = GLiNER.from_pretrained("path/to/this/model")\n'
        f'text = "..."\n{call}\n```\n'
    )


def render_model_card(config, data_stats: Optional[Dict[str, Any]] = None, best_f1: Optional[float] = None) -> str:
    """Render the full Markdown model card for a trained GLiNER model."""
    task = _task(config)
    base_model = getattr(config, "model_name", None) or "unknown"
    frontmatter = (
        "---\n"
        "library_name: gliner\n"
        "pipeline_tag: token-classification\n"
        "tags:\n- gliner\n"
        f"- {task}\n"
        f"base_model: {base_model}\n"
        "metrics:\n- f1\n"
        "---"
    )
    title = {
        "event-extraction": "# GLiNER Event Extraction Model",
        "relation-extraction": "# GLiNER Relation Extraction Model",
        "named-entity-recognition": "# GLiNER Named Entity Recognition Model",
    }[task]

    overview = "## Overview\n\n"
    overview += f"- **Base encoder:** `{base_model}`\n"
    overview += f"- **Framework:** GLiNER (`event_mode={bool(getattr(config, 'event_mode', False))}`)\n"
    if best_f1 is not None:
        overview += f"- **Best validation F1:** {best_f1:.4f}"

    sections = [
        frontmatter,
        f"{title}\n\n{_purpose(task)}",
        overview,
        _label_schema(config, data_stats),
        _training_data_section(data_stats),
        _config_section(config),
        _usage_section(config),
        "---\n*Model card generated automatically at training time.*",
    ]
    return "\n\n".join(s.rstrip() for s in sections if s and s.strip()) + "\n"


def write_model_card(
    save_dir: Union[str, Path],
    config,
    data_stats: Optional[Dict[str, Any]] = None,
    best_f1: Optional[float] = None,
) -> Path:
    """Write the model card to ``<save_dir>/README.md`` and return its path."""
    path = Path(save_dir) / "README.md"
    path.write_text(render_model_card(config, data_stats, best_f1), encoding="utf-8")
    return path
