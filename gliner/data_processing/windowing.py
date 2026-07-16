"""Sliding-window support for documents longer than ``max_len``.

Without this, ``preprocess_example`` truncates at ``max_len`` and everything
past the cutoff is invisible to the model, at both training and inference
time. This module splits long inputs into overlapping windows and, for
eval/inference, merges per-window predictions back into document-level
coordinates.

A relation/event-role is only recoverable if both its head and tail land in
the same window -- windowing raises entity/trigger recall, not relation
recall, for documents whose annotations span more than one window's worth of
tokens.
"""

from dataclasses import replace as dataclass_replace
from typing import Any, Dict, List, Optional, Tuple

from ..decoding.decoder import Span


def default_stride(max_len: int) -> int:
    """25% overlap between consecutive windows, floored at 1 token."""
    return max(max_len // 4, 1)


def split_windows(num_tokens: int, max_len: int, stride: int) -> List[Tuple[int, int]]:
    """Word-token index ranges covering ``[0, num_tokens)``.

    Each range has length <= ``max_len``; consecutive ranges overlap by
    ``stride`` tokens. Returns a single ``(0, num_tokens)`` range unchanged
    when the input already fits.
    """
    if num_tokens <= max_len:
        return [(0, num_tokens)]

    step = max(max_len - stride, 1)
    windows = []
    start = 0
    while start < num_tokens:
        end = min(start + max_len, num_tokens)
        windows.append((start, end))
        if end == num_tokens:
            break
        start += step
    return windows


def window_training_record(record: Dict[str, Any], max_len: int, stride: Optional[int] = None) -> List[Dict[str, Any]]:
    """Split a training record into overlapping windows.

    An entity is included, reindexed, in *every* window that fully contains
    it -- not just one -- so an entity truncated out of one window but fully
    present in an adjacent one is never silently unlabeled where it actually
    appears (which would train the classifier to treat a real entity as a
    negative). A relation/event-role is included in a window only if both
    endpoints are present in that window's reindexed entity list.

    Returns ``[record]`` unchanged if it already fits in ``max_len``.
    """
    tokens = record["tokenized_text"]
    num_tokens = len(tokens)
    if num_tokens <= max_len:
        return [record]

    if stride is None:
        stride = default_stride(max_len)

    ner = record.get("ner") or []
    relations = record.get("relations")

    windows = []
    for w_start, w_end in split_windows(num_tokens, max_len, stride):
        old_to_new: Dict[int, int] = {}
        window_ner = []
        for old_idx, ent in enumerate(ner):
            e_start, e_end, label = ent[0], ent[1], ent[2]
            if e_start >= w_start and e_end < w_end:
                old_to_new[old_idx] = len(window_ner)
                window_ner.append((e_start - w_start, e_end - w_start, label))

        window_record = dict(record)
        window_record["tokenized_text"] = tokens[w_start:w_end]
        window_record["ner"] = window_ner

        if relations is not None:
            window_relations = []
            for rel in relations:
                head_idx, tail_idx, rel_label = rel[0], rel[1], rel[2]
                if head_idx in old_to_new and tail_idx in old_to_new:
                    window_relations.append((old_to_new[head_idx], old_to_new[tail_idx], rel_label))
            window_record["relations"] = window_relations

        windows.append(window_record)

    return windows


def window_tokens(tokens: List[str], max_len: int, stride: Optional[int] = None) -> List[Tuple[List[str], int]]:
    """Split ``tokens`` into overlapping (chunk, document-level offset) pairs.

    Input-only chunking for eval/inference model input -- no label
    filtering, since there's nothing to filter.
    """
    if stride is None:
        stride = default_stride(max_len)
    return [(tokens[start:end], start) for start, end in split_windows(len(tokens), max_len, stride)]


def prepare_windowed_items(
    items: List[Dict[str, Any]], max_len: int, stride: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    """Expand ``{"tokenized_text": [...]}``-shaped items into model-input windows.

    Shared bookkeeping for eval (``test_data`` records) and raw-text
    inference (``input_x`` items) -- both only need ``tokenized_text`` for
    the forward pass. Returns the flat expanded list plus parallel
    ``owner_idx`` (which original item each window belongs to) and
    ``offset`` lists, so window-level decode output can be grouped back by
    owner and merged.
    """
    if stride is None:
        stride = default_stride(max_len)

    expanded: List[Dict[str, Any]] = []
    owner_idx: List[int] = []
    offsets: List[int] = []
    for i, item in enumerate(items):
        for chunk, offset in window_tokens(item["tokenized_text"], max_len, stride):
            expanded.append({"tokenized_text": chunk, "ner": None})
            owner_idx.append(i)
            offsets.append(offset)
    return expanded, owner_idx, offsets


def window_input_spans(
    word_spans_per_item: Optional[List[List[Tuple[int, int]]]],
    window_items: List[Dict[str, Any]],
    owner_idx: List[int],
    offsets: List[int],
) -> Optional[List[List[Tuple[int, int]]]]:
    """Filter and reindex each owner's pre-defined input spans into its windows.

    Mirrors ``window_training_record``'s "fully contained" entity filter,
    for plain (start, end) spans with no label -- used when the caller
    passes ``input_spans`` to classify pre-defined spans rather than let the
    model detect them. Returns ``None`` (unchanged) when there are no input
    spans to window.
    """
    if word_spans_per_item is None:
        return None

    windowed: List[List[Tuple[int, int]]] = []
    for item, owner, offset in zip(window_items, owner_idx, offsets):
        w_start = offset
        w_end = offset + len(item["tokenized_text"])
        spans = word_spans_per_item[owner]
        windowed.append([(s - w_start, e - w_start) for s, e in spans if s >= w_start and e < w_end])
    return windowed


def group_and_merge_spans(
    window_preds: List[List[Span]], owner_idx: List[int], offsets: List[int], num_owners: int
) -> List[List[Span]]:
    """Group per-window ``Span`` predictions by owner and merge into document-level lists."""
    grouped: List[List[Tuple[List[Span], int]]] = [[] for _ in range(num_owners)]
    for preds, owner, offset in zip(window_preds, owner_idx, offsets):
        grouped[owner].append((preds, offset))
    return [merge_windowed_spans(windows)[0] for windows in grouped]


def group_and_merge_relex_outputs(
    window_entities: List[List[Span]],
    window_relations: List[List[tuple]],
    owner_idx: List[int],
    offsets: List[int],
    num_owners: int,
) -> Tuple[List[List[Span]], List[List[tuple]]]:
    """Group per-window entity+relation predictions by owner and merge both together.

    Relation index remapping depends on the entity merge's index maps, so
    entities and relations must be merged in lockstep, per owner.
    """
    grouped_entities: List[List[Tuple[List[Span], int]]] = [[] for _ in range(num_owners)]
    grouped_relations: List[List[List[tuple]]] = [[] for _ in range(num_owners)]
    for ents, rels, owner, offset in zip(window_entities, window_relations, owner_idx, offsets):
        grouped_entities[owner].append((ents, offset))
        grouped_relations[owner].append(rels)

    merged_entities = []
    merged_relations = []
    for ents_windows, rels_windows in zip(grouped_entities, grouped_relations):
        ents, index_maps = merge_windowed_spans(ents_windows)
        rels = merge_windowed_relations(rels_windows, index_maps)
        merged_entities.append(ents)
        merged_relations.append(rels)

    return merged_entities, merged_relations


def merge_windowed_spans(windows: List[Tuple[List[Span], int]]) -> Tuple[List[Span], List[List[int]]]:
    """Remap window-local ``Span`` predictions into document-level coordinates and dedupe.

    Duplicate (start, end, label) predictions from overlapping windows
    collapse to one entry (highest score kept). Returns the merged list and,
    per input window, a list mapping that window's local span index to its
    position in the merged list -- needed to remap relation head/tail
    indices in ``merge_windowed_relations``.
    """
    merged: List[Span] = []
    seen: Dict[Tuple[int, int, str], int] = {}
    index_maps: List[List[int]] = []

    for spans, offset in windows:
        local_map = []
        for span in spans:
            shifted = dataclass_replace(span, start=span.start + offset, end=span.end + offset)
            key = (shifted.start, shifted.end, shifted.entity_type)
            if key in seen:
                merged_idx = seen[key]
                if shifted.score > merged[merged_idx].score:
                    merged[merged_idx] = shifted
            else:
                merged_idx = len(merged)
                merged.append(shifted)
                seen[key] = merged_idx
            local_map.append(merged_idx)
        index_maps.append(local_map)

    return merged, index_maps


def merge_windowed_relations(windows: List[List[tuple]], index_maps: List[List[int]]) -> List[tuple]:
    """Remap window-local relation index-pairs into merged-entity-index space and dedupe.

    Each relation tuple is ``(head_idx, label, tail_idx, score)`` (matching
    the decoder's output shape), with indices local to that window's own
    predicted entity list. Duplicate (head, tail, label) triples from
    overlapping windows collapse to one entry (highest score kept).
    """
    merged: List[tuple] = []
    seen: Dict[Tuple[int, int, Any], int] = {}

    for rels, local_map in zip(windows, index_maps):
        for rel in rels:
            head_idx, label, tail_idx = rel[0], rel[1], rel[2]
            score = rel[3] if len(rel) > 3 else None
            if head_idx >= len(local_map) or tail_idx >= len(local_map):
                continue
            merged_head = local_map[head_idx]
            merged_tail = local_map[tail_idx]
            key = (merged_head, merged_tail, label)
            if key in seen:
                idx = seen[key]
                existing_score = merged[idx][3]
                if score is not None and (existing_score is None or score > existing_score):
                    merged[idx] = (merged_head, label, merged_tail, score)
            else:
                seen[key] = len(merged)
                merged.append((merged_head, label, merged_tail, score))

    return merged
