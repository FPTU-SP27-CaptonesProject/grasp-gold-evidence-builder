"""Canonical JSONL and manifest writers."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .passages import PassageLookup
from .sources import read_records, sha256


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def write_jsonl(path: str | Path, rows: list[dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(destination)


def build_manifest(
    files: list[Path],
    lookups: dict[str, PassageLookup],
    retrieval_revision: str,
    source_inventory: dict[str, Any] | None = None,
) -> dict[str, object]:
    directories = {"2wiki_ircot": "2wiki/", "hotpotqa_abstracts_2017": "hotpotqa/", "wiki18": "nq/"}
    inventory_files = (source_inventory or {}).get("files", [])
    corpora = {}
    for dataset, lookup in sorted(lookups.items()):
        sources = [{"path": entry["path"], "sha256": entry["sha256"], "source": entry.get("source", {})} for entry in inventory_files if entry["path"].startswith(directories[dataset])]
        corpora[dataset] = {
            "passages_sqlite": str(lookup.database),
            "passages_sqlite_sha256": lookup.database_sha256,
            "source_dataset": sources,
            "mapping_method": "gold_ctx_id | verified_normalized_token_containment_v1" if dataset == "wiki18" else "gold_ctx_id | supporting_fact_title",
            "title_normalization": "HTML unescape; Unicode NFKC; casefold; underscores to spaces; collapse whitespace; preserve punctuation",
        }
    file_records = {}
    for path in files:
        rows = read_records(path)
        file_records[path.name] = {
            "rows": len(rows), "sha256": sha256(path),
            "mapped_rows": sum(row.get("evidence_status", "mapped") == "mapped" for row in rows),
            "unmapped_rows": sum(row.get("evidence_status") == "unmapped" for row in rows),
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "retrieval_repository": "doancuavahk/grasp-retrieval",
        "retrieval_revision": retrieval_revision,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "corpora": corpora,
        "files": file_records,
        "source_inventory": source_inventory,
        "evidence_status_policy": "mapped requires verified gold_doc_ids; unmapped has no gold_doc_ids and recall is unavailable",
    }
    if "train_mapped.jsonl" in file_records:
        result["recall_train_file"] = "train_mapped.jsonl"
    return result
