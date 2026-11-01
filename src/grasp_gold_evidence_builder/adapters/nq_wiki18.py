"""Processed NQ adapter: only direct verified passage IDs are accepted."""
from __future__ import annotations

from typing import Any

from ..normalize import nonempty_strings
from .common import processed_ids


def adapt(record: dict[str, Any]) -> dict[str, object]:
    identifier = record.get("id", record.get("_id"))
    question = record.get("question") or ""
    question = question.strip() if isinstance(question, str) else ""
    answers = nonempty_strings(record.get("gold_answers", record.get("answer", record.get("answers", []))))
    identifiers = processed_ids(record)
    return {"source_id": None if identifier is None else str(identifier), "question": question, "gold_answers": answers, "source_doc_ids": identifiers, "supporting_titles": set()}
