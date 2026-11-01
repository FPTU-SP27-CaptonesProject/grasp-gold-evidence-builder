"""Explicit passage evidence from processed sources, never retrieved context."""
from __future__ import annotations


def processed_ids(record: dict) -> set[str]:
    contexts = record.get("gold_ctxs", record.get("gold_contexts", [])) or []
    if not isinstance(contexts, list):
        raise TypeError("gold_ctxs/gold_contexts must be a list of passage objects")
    identifiers = set()
    for context in contexts:
        if not isinstance(context, dict):
            raise TypeError("each gold context must be a passage object")
        value = next((context[key] for key in ("doc_id", "passage_id", "id", "document_id") if context.get(key) is not None), None)
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
            raise ValueError("every gold context must contain a nonempty passage ID; partial labels are not accepted")
        identifiers.add(str(value).strip())
    return identifiers
