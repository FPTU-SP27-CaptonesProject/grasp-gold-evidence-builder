"""Fail-closed mapping from original benchmark evidence to retrieval doc IDs."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .adapters import ADAPTERS
from .normalize import normalized_title
from .passages import PassageLookup
from .sources import read_records, sha256


@dataclass(frozen=True)
class CorpusSpec:
    dataset: str
    source: Path
    lookup: PassageLookup


def default_corpora(retrieval_root: str | Path) -> dict[str, Path]:
    root = Path(retrieval_root)
    return {
        "2wiki_ircot": root / "2wiki_ircot" / "passages.sqlite",
        "hotpotqa_abstracts_2017": root / "hotpotqa_abstracts_2017" / "passages.sqlite",
        "wiki18": root / "wiki18" / "passages.sqlite",
    }


def resolve_evidence(adapted: dict[str, object], lookup: PassageLookup) -> tuple[list[str] | None, str]:
    direct = {str(value) for value in adapted["source_doc_ids"]}
    invalid_direct = direct - lookup.doc_ids
    if invalid_direct:
        return None, "source_doc_id_not_in_passages"
    resolved = set(direct)
    for title in adapted["supporting_titles"]:
        candidates = lookup.title_to_ids.get(normalized_title(str(title)), frozenset())
        if not candidates:
            return None, "supporting_title_not_in_passages"
        if len(candidates) != 1:
            return None, "ambiguous_supporting_title"
        resolved.update(candidates)
    if not resolved:
        return None, "missing_gold_evidence"
    return sorted(resolved), "mapped"


def build_split(specifications: list[CorpusSpec], split: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    output: list[dict[str, object]] = []
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)
    source_info: dict[str, dict[str, object]] = {}
    seen_ids: set[str] = set()
    for spec in specifications:
        adapter = ADAPTERS[spec.dataset]
        records = read_records(spec.source)
        source_info[spec.dataset] = {"path": str(spec.source.resolve()), "sha256": sha256(spec.source), "rows": len(records)}
        for position, source_row in enumerate(records):
            counters[spec.dataset]["input"] += 1
            item = adapter(source_row)
            source_id = item["source_id"]
            question = item["question"]
            answers = item["gold_answers"]
            if not source_id or not question or not answers:
                status = "invalid_source_row"
            else:
                doc_ids, status = resolve_evidence(item, spec.lookup)
                if doc_ids is not None:
                    identifier = f"{spec.dataset}::{source_id}"
                    if identifier in seen_ids:
                        raise ValueError(f"duplicate canonical id: {identifier}")
                    seen_ids.add(identifier)
                    output.append({"id": identifier, "dataset": spec.dataset, "split": split, "question": question, "gold_answers": answers, "gold_doc_ids": doc_ids, "evidence_status": "mapped"})
                    counters[spec.dataset]["mapped"] += 1
                    counters[spec.dataset]["gold_doc_count"] += len(doc_ids)
                    continue
            counters[spec.dataset][status] += 1
            if len(samples[spec.dataset]) < 20:
                samples[spec.dataset].append({"source_row": str(position), "source_id": str(source_id or ""), "status": status, "question": str(question or "")})
    report = {
        "split": split,
        "rows_written": len(output),
        "sources": source_info,
        "datasets": {
            dataset: {**dict(counter), "mean_gold_doc_count": counter["gold_doc_count"] / counter["mapped"] if counter["mapped"] else 0.0, "examples_not_written": samples[dataset]}
            for dataset, counter in sorted(counters.items())
        },
    }
    return output, report


def source_requirements(dataset: str, source: Path) -> tuple[list[dict[str, object]], set[str], set[str]]:
    """Adapt a raw file once and return its source rows plus sparse lookup requirements."""
    adapter = ADAPTERS[dataset]
    adapted = [adapter(row) for row in read_records(source)]
    titles = {str(title) for row in adapted for title in row["supporting_titles"]}
    doc_ids = {str(doc_id) for row in adapted for doc_id in row["source_doc_ids"]}
    return adapted, titles, doc_ids


def build_adapted_split(
    specifications: list[tuple[str, Path, PassageLookup, list[dict[str, object]]]], split: str
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build from pre-adapted sources so large source files are read only once."""
    output: list[dict[str, object]] = []
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)
    source_info: dict[str, dict[str, object]] = {}
    failures: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for dataset, source, lookup, records in specifications:
        source_info[dataset] = {"path": str(source.resolve()), "sha256": sha256(source), "rows": len(records)}
        for position, item in enumerate(records):
            counters[dataset]["input"] += 1
            source_id, question, answers = item["source_id"], item["question"], item["gold_answers"]
            if not source_id or not question or not answers:
                status = "invalid_source_row"
            else:
                doc_ids, status = resolve_evidence(item, lookup)
                if doc_ids is not None:
                    identifier = f"{dataset}::{source_id}"
                    if identifier in seen_ids:
                        raise ValueError(f"duplicate canonical id: {identifier}")
                    seen_ids.add(identifier)
                    output.append({"id": identifier, "dataset": dataset, "split": split, "question": question, "gold_answers": answers, "gold_doc_ids": doc_ids, "evidence_status": "mapped"})
                    counters[dataset]["mapped"] += 1
                    counters[dataset]["gold_doc_count"] += len(doc_ids)
                    continue
            counters[dataset][status] += 1
            failures[dataset].append({"source_id": str(source_id or ""), "status": status})
            if len(samples[dataset]) < 20:
                samples[dataset].append({"source_row": str(position), "source_id": str(source_id or ""), "status": status, "question": str(question or "")})
    report = {"split": split, "rows_written": len(output), "sources": source_info, "unmapped_source_rows": dict(failures), "datasets": {dataset: {**dict(counter), "mean_gold_doc_count": counter["gold_doc_count"] / counter["mapped"] if counter["mapped"] else 0.0, "examples_not_written": samples[dataset]} for dataset, counter in sorted(counters.items())}}
    return output, report
