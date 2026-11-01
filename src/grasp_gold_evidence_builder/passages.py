"""Read-only access to the exact retrieval-corpus passage databases."""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .normalize import normalized_title
from .sources import sha256


@dataclass(frozen=True)
class PassageLookup:
    doc_ids: frozenset[str]
    title_to_ids: dict[str, frozenset[str]]
    database: Path
    database_sha256: str


def _database_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro"


def load_lookup(
    database: str | Path,
    *,
    required_titles: set[str] | None = None,
    required_doc_ids: set[str] | None = None,
) -> PassageLookup:
    """Scan a passage DB once while retaining only evidence requested by source rows.

    The HotpotQA corpus has millions of passages and no title index. Retaining every
    title-to-ID mapping would consume multiple gigabytes, while a one-pass sparse
    lookup preserves the same fail-closed matching semantics for the requested split.
    Passing ``None`` preserves the full-lookup behavior for small test databases.
    """
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(f"passages.sqlite not found: {path}")
    connection = sqlite3.connect(_database_uri(path), uri=True)
    try:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(passages)")}
        required = {"doc_id", "title"}
        if not required <= columns:
            raise ValueError(f"{path}: passages table lacks required columns {sorted(required - columns)}")
        wanted_titles = None if required_titles is None else {normalized_title(value) for value in required_titles}
        wanted_ids = None if required_doc_ids is None else {str(value) for value in required_doc_ids}
        rows = connection.execute("SELECT doc_id FROM passages" if wanted_titles == set() else "SELECT doc_id, title FROM passages")
        ids = set()
        titles: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            doc_id = row[0]
            title = row[1] if len(row) > 1 else None
            identifier = str(doc_id)
            if wanted_ids is None or identifier in wanted_ids:
                ids.add(identifier)
            if title is not None and str(title).strip():
                key = normalized_title(str(title))
                if wanted_titles is None or key in wanted_titles:
                    titles[key].add(identifier)
            if wanted_titles == set() and wanted_ids is not None and ids >= wanted_ids:
                break
    finally:
        connection.close()
    return PassageLookup(frozenset(ids), {key: frozenset(value) for key, value in titles.items()}, path.resolve(), sha256(path))
