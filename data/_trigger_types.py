"""Backward-compatible shim: ``derive_trigger_types`` now lives in the
installable ``gliner`` package so the training scripts can import it at
runtime. See ``gliner/data_processing/trigger_types.py`` for the full rationale.
"""

from __future__ import annotations

from gliner.data_processing.trigger_types import derive_trigger_types

__all__ = ["derive_trigger_types"]
