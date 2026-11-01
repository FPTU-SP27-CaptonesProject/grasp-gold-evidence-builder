"""Strict validator for published canonical gold-evidence files."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .mapping import default_corpora
from .passages import PassageLookup, load_lookup
from .questions import normalized_text
from .sources import read_records, sha256

REQUIRED = {"id", "dataset", "split", "question", "gold_answers", "gold_doc_ids"}


def validate(inputs: list[str | Path], retrieval_root: str | Path) -> tuple[dict[str, Any], dict[str, PassageLookup]]:
    lookup_paths = default_corpora(retrieval_root)
    lookups: dict[str, PassageLookup] = {}
    counts: Counter[str] = Counter()
    by_dataset: dict[str, Counter[str]] = {}
    seen_ids: set[str] = set()
    errors: list[dict[str, str]] = []
    seen_questions: dict[tuple[str, str], str] = {}
    rows = [(source, line, row) for source in inputs for line, row in enumerate(read_records(source), start=1)]
    counts["rows"] = 0
    counts["invalid_doc_ids"] = 0
    counts["mapped_rows"] = 0
    counts["unmapped_rows"] = 0
    for source in inputs:
        if not any(Path(found_source) == Path(source) for found_source, _, _ in rows):
            errors.append({"file": str(source), "line": "0", "error": "empty split file"})
    requested_ids: dict[str, set[str]] = {}
    for _, _, row in rows:
        dataset = str(row.get("dataset", ""))
        if dataset in lookup_paths and isinstance(row.get("gold_doc_ids"), list):
            requested_ids.setdefault(dataset, set()).update(str(value) for value in row["gold_doc_ids"])
    for dataset, identifiers in requested_ids.items():
        lookups[dataset] = load_lookup(lookup_paths[dataset], required_titles=set(), required_doc_ids=identifiers)
    for source, line, row in rows:
        counts["rows"] += 1
        missing = REQUIRED - set(row)
        dataset = str(row.get("dataset", ""))
        if missing:
            errors.append({"file": str(source), "line": str(line), "error": f"missing fields: {sorted(missing)}"})
            continue
        if dataset not in lookup_paths:
            errors.append({"file": str(source), "line": str(line), "error": f"unknown dataset: {dataset}"})
            continue
        if dataset not in lookups:
            lookups[dataset] = load_lookup(lookup_paths[dataset], required_titles=set(), required_doc_ids=set())
        bucket = by_dataset.setdefault(dataset, Counter())
        bucket["rows"] += 1
        identifier = row["id"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen_ids:
            errors.append({"file": str(source), "line": str(line), "error": "empty or duplicate id"})
        else:
            seen_ids.add(identifier)
            if not identifier.startswith(dataset + "::"):
                errors.append({"file": str(source), "line": str(line), "error": "id prefix must match dataset"})
        split = row["split"]
        if split not in ("train", "dev", "test") or (Path(source).stem in ("train", "dev", "test") and split != Path(source).stem):
            errors.append({"file": str(source), "line": str(line), "error": "invalid or mismatched split"})
        question, answers = row["question"], row["gold_answers"]
        if not isinstance(question, str) or not question.strip() or not isinstance(answers, list) or not answers or any(not isinstance(value, str) or not value.strip() for value in answers):
            errors.append({"file": str(source), "line": str(line), "error": "empty question or gold_answers"})
        if isinstance(question, str):
            key = (dataset, normalized_text(question))
            if key in seen_questions and seen_questions[key] != split:
                errors.append({"file": str(source), "line": str(line), "error": "question overlaps another split"})
            seen_questions[key] = split
        if not isinstance(row["gold_doc_ids"], list) or any(not isinstance(value, str) or not value.strip() for value in row["gold_doc_ids"]):
            errors.append({"file": str(source), "line": str(line), "error": "gold_doc_ids must be nonempty strings"})
        ids = [str(value) for value in row["gold_doc_ids"]] if isinstance(row["gold_doc_ids"], list) else []
        if len(ids) != len(set(ids)):
            errors.append({"file": str(source), "line": str(line), "error": "duplicate gold_doc_ids"})
        status = row.get("evidence_status", "mapped")
        if status not in ("mapped", "unmapped"):
            errors.append({"file": str(source), "line": str(line), "error": "evidence_status must be mapped or unmapped"})
        else:
            counts[status + "_rows"] += 1
            bucket[status] += 1
        if status == "mapped" and not ids:
            errors.append({"file": str(source), "line": str(line), "error": "mapped row has empty gold_doc_ids"})
        if status == "unmapped" and ids:
            errors.append({"file": str(source), "line": str(line), "error": "unmapped row must have empty gold_doc_ids"})
        invalid = set(ids) - lookups[dataset].doc_ids
        if invalid:
            counts["invalid_doc_ids"] += len(invalid)
            bucket["invalid_doc_ids"] += len(invalid)
            errors.append({"file": str(source), "line": str(line), "error": f"invalid_doc_ids: {sorted(invalid)[:5]}"})
        else:
            bucket["valid_doc_ids"] += len(ids)
            bucket.setdefault("invalid_doc_ids", 0)
    counts["invalid_rows"] = len({(error["file"], error["line"]) for error in errors})
    report = {"valid": not errors, **dict(counts), "error_count": len(errors), "files": {Path(source).name: {"sha256": sha256(source)} for source in inputs}, "datasets": {key: dict(value) for key, value in sorted(by_dataset.items())}, "errors": errors[:100]}
    return report, lookups
