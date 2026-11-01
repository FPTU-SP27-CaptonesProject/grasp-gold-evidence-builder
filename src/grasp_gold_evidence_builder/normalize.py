"""Normalization shared by adapters and the passage-title mapper."""
from __future__ import annotations

import html
import unicodedata


def normalized_title(value: str) -> str:
    """Normalize titles only for matching; never change emitted source text."""
    text = unicodedata.normalize("NFKC", html.unescape(str(value))).casefold().replace("_", " ")
    # Punctuation distinguishes article identities, e.g. C, C++ and C#.
    return " ".join(text.split())


def nonempty_strings(value: object) -> list[str]:
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list):
        return []
    return [item.strip() for item in values if isinstance(item, str) and item.strip()]
