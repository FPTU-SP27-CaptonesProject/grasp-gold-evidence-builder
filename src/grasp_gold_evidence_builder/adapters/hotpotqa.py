"""Official HotpotQA supporting-fact adapter."""
from __future__ import annotations

from typing import Any

from ..normalize import nonempty_strings
from .common import processed_ids


def adapt(record: dict[str, Any]) -> dict[str, object]:
    identifier = record.get("_id", record.get("id"))
    question = record.get("question") or ""
    question = question.strip() if isinstance(question, str) else ""
    answers = nonempty_strings(record.get("gold_answers", record.get("answer", record.get("answers", []))))
    identifiers = processed_ids(record)
    titles = {fact[0].strip() for fact in (record.get("supporting_facts") or []) if isinstance(fact, (list, tuple)) and fact and isinstance(fact[0], str) and fact[0].strip()}
    return {"source_id": None if identifier is None else str(identifier), "question": question, "gold_answers": answers, "source_doc_ids": identifiers, "supporting_titles": set() if identifiers else titles}
