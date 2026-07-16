"""Shared char-offset tokenization helpers.

Some sources ship raw text with character-offset annotations (CASIE) or
only surface-form strings with no offsets at all (DocEE, events_biotech),
rather than a pre-tokenized word list. This module word-tokenizes with
GLiNER's own ``WhitespaceTokenSplitter`` so token boundaries match what the
rest of the pipeline expects, then maps a character span (or a literal
substring search) back to a token index span.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gliner.data_processing.tokenizer import WhitespaceTokenSplitter  # noqa: E402

_splitter = WhitespaceTokenSplitter()


def tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Word-tokenize ``text``; returns (tokens, [(start, end), ...]) with end exclusive."""
    tokens: List[str] = []
    offsets: List[Tuple[int, int]] = []
    for tok, start, end in _splitter(text):
        tokens.append(tok)
        offsets.append((start, end))
    return tokens, offsets


def char_span_to_token_span(
    offsets: List[Tuple[int, int]], char_start: int, char_end: int
) -> Optional[Tuple[int, int]]:
    """Map a char span [char_start, char_end) to an inclusive token index span.

    Returns None if no token overlaps the span.
    """
    matched = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
    if not matched:
        return None
    return matched[0], matched[-1]


def find_surface_span(tokens: List[str], surface: str) -> Optional[Tuple[int, int]]:
    """Locate ``surface`` as a contiguous run of tokens inside ``tokens``.

    ``surface`` is itself whitespace-tokenized so multi-word surfaces match
    correctly. Returns the first match's inclusive (start, end) token index
    span, or None if it doesn't appear as a contiguous subsequence.
    """
    surf_tokens = [t for t, _, _ in _splitter(surface)]
    if not surf_tokens:
        return None
    n = len(surf_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == surf_tokens:
            return i, i + n - 1
    return None
