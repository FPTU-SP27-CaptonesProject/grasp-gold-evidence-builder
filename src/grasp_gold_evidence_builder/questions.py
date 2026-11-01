"""Keep the questions actually used by GRASP and join them to original labels."""
from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .normalize import nonempty_strings
from .sources import iter_records

ALIASES = {
    "2wiki": "2wiki_ircot", "2wikimultihopqa": "2wiki_ircot", "2wiki_ircot": "2wiki_ircot",
    "hotpotqa": "hotpotqa_abstracts_2017", "hotpotqa_abstracts_2017": "hotpotqa_abstracts_2017",
    "nq": "wiki18", "wiki18": "wiki18",
}


def normalized_text(value: str) -> str:
    # Preserve punctuation: near-matching questions must not be silently merged.
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_questions(paths: list[str | Path], *, keep_all: bool = False) -> list[dict]:
    result, seen = [], set()
    for path in paths:
        for position, row in enumerate(iter_records(path), start=1):
            if not keep_all and row.get("status") not in (None, "ok"):
                continue
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise TypeError(f"{path}:{position}: metadata must be an object")
            corpus = row.get("corpus") or row.get("dataset") or metadata.get("dataset")
            dataset = ALIASES.get(str(corpus).casefold())
            if dataset is None:
                raise ValueError(f"{path}:{position}: unknown corpus {corpus!r}")
            identifier = row.get("id", row.get("qid"))
            question = row.get("question")
            answers = nonempty_strings(row.get("gold_answers", row.get("answers", row.get("answer", []))))
            if identifier is None or not str(identifier).strip() or not isinstance(question, str) or not question.strip() or not answers:
                raise ValueError(f"{path}:{position}: id/qid, question, and answers are required")
            key = (dataset, str(identifier))
            if key in seen:
                raise ValueError(f"duplicate input question ID: {dataset}::{identifier}")
            seen.add(key)
            result.append({"id": str(identifier), "dataset": dataset, "question": question.strip(), "gold_answers": answers})
    if not result:
        raise ValueError("no usable input questions")
    return result


def match_questions(targets: list[dict], annotations: list[dict]) -> tuple[list[dict], dict]:
    """Require question AND answer agreement, even for a matching source ID."""
    by_question = defaultdict(list)
    for annotation in annotations:
        if annotation["source_id"] and annotation["question"] and annotation["gold_answers"]:
            by_question[normalized_text(annotation["question"])].append(annotation)
    selected, matches, rejected = [], [], []
    used_source_ids = set()
    for target in targets:
        candidates = by_question.get(normalized_text(target["question"]), [])
        wanted = {normalized_text(answer) for answer in target["gold_answers"]}
        compatible = [row for row in candidates if wanted & {normalized_text(answer) for answer in row["gold_answers"]}]
        input_id = target["id"].rsplit("::", 1)[-1]
        by_id = [row for row in compatible if row["source_id"] == input_id]
        pool = by_id if len(by_id) == 1 else compatible
        if not candidates:
            status = "question_not_found"
        elif not compatible:
            status = "answer_mismatch"
        elif len(pool) != 1:
            status = "ambiguous_question"
        elif pool[0]["source_id"] in used_source_ids:
            status = "duplicate_source_question"
        else:
            annotation = pool[0]
            used_source_ids.add(annotation["source_id"])
            selected.append(annotation)
            matches.append({"input_id": target["id"], "source_id": annotation["source_id"], "method": "id_question_answer" if by_id else "question_answer"})
            continue
        rejected.append({"input_id": target["id"], "question": target["question"], "status": status})
    return selected, {"input_questions": len(targets), "matched_questions": len(matches), "matches": matches, "unmatched_questions": rejected}


def retain_questions(
    targets: list[dict], mapped_rows: list[dict], joins: dict, report: dict, split: str
) -> list[dict]:
    """Keep every input QA in order; missing evidence is explicit, never a fabricated label."""
    mapped = {row["id"]: row for row in mapped_rows}
    matches = {
        (dataset, match["input_id"]): match["source_id"]
        for dataset, join in joins.items() for match in join["matches"]
    }
    rejected = {
        (dataset, row["input_id"]): row["status"]
        for dataset, join in joins.items() for row in join["unmatched_questions"]
    }
    failures = {
        (dataset, row["source_id"]): row["status"]
        for dataset, rows in report.get("unmapped_source_rows", {}).items() for row in rows
    }
    missing = {entry["dataset"] for entry in report.get("missing_sources", [])}
    output, seen = [], set()
    counts, reasons = defaultdict(Counter), defaultdict(Counter)
    for target in targets:
        dataset, input_id = target["dataset"], target["id"]
        source_id = matches.get((dataset, input_id))
        resolved = mapped.get(f"{dataset}::{source_id}") if source_id is not None else None
        row = {
            "id": resolved["id"] if resolved else f"{dataset}::unmapped::{input_id}",
            "input_id": input_id, "dataset": dataset, "split": split,
            "question": target["question"], "gold_answers": list(target["gold_answers"]),
            "gold_doc_ids": list(resolved["gold_doc_ids"]) if resolved else [],
            "evidence_status": "mapped" if resolved else "unmapped",
        }
        if source_id is not None:
            row["source_id"] = source_id
        if not resolved:
            reason = rejected.get((dataset, input_id)) or failures.get((dataset, source_id))
            row["evidence_reason"] = reason or (
                "missing_source_annotations" if dataset in missing else "missing_gold_evidence"
            )
            reasons[dataset][row["evidence_reason"]] += 1
        if row["id"] in seen:
            raise ValueError(f"duplicate retained QA ID: {row['id']}")
        seen.add(row["id"])
        counts[dataset][row["evidence_status"]] += 1
        output.append(row)
    for dataset, counter in counts.items():
        entry = report.setdefault("datasets", {}).setdefault(dataset, {})
        entry["source_rows_considered"] = entry.get("input", 0)
        entry["examples_without_mapped_evidence"] = entry.pop("examples_not_written", [])
        entry.update({"input": sum(counter.values()), "output_rows": sum(counter.values()), "mapped": counter["mapped"],
                      "unmapped": counter["unmapped"], "unmapped_reasons": dict(reasons[dataset])})
    report.update({"rows_written": len(output), "mapped_rows": len(mapped_rows),
                   "unmapped_rows": len(output) - len(mapped_rows), "retains_all_input_questions": True})
    return output
