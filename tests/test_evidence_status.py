import json
import sys

import pytest
from test_builder import database

from grasp_gold_evidence_builder.cli import build_main, validate_main
from grasp_gold_evidence_builder.questions import retain_questions
from grasp_gold_evidence_builder.sources import read_records, sha256
from grasp_gold_evidence_builder.validate import validate


def test_retain_unmapped_cli_preserves_order_answers_and_failure_reasons(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "hotpotqa"
    raw.mkdir(parents=True)
    annotations = [
        {"_id": "yes", "question": "Mapped?", "answer": "A", "supporting_facts": [["Page", 0]]},
        {"_id": "ambiguous", "question": "Ambiguous?", "answer": "B", "supporting_facts": [["Duplicate", 0]]},
    ]
    (raw / "train.json").write_text(json.dumps(annotations), encoding="utf-8")
    corpus = tmp_path / "retrieval" / "hotpotqa_abstracts_2017"
    corpus.mkdir(parents=True)
    database(corpus / "passages.sqlite", [("1", "Page", "T"), ("2", "Duplicate", "T"), ("3", "Duplicate", "T")])
    targets = [
        {"id": "b", "dataset": "hotpotqa", "question": "Ambiguous?", "gold_answers": ["B"]},
        {"id": "a", "dataset": "hotpotqa", "question": "Mapped?", "gold_answers": ["A", "Alias"]},
        {"id": "c", "dataset": "hotpotqa", "question": "Unknown?", "gold_answers": ["C"]},
        {"id": "d", "dataset": "nq", "question": "Missing source?", "gold_answers": ["D"]},
    ]
    qa, output, report = tmp_path / "qa.jsonl", tmp_path / "train.jsonl", tmp_path / "report.json"
    qa.write_text("".join(json.dumps(row) + "\n" for row in targets), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["build", "--split", "train", "--raw-root", str(raw.parent),
        "--retrieval-root", str(corpus.parent), "--qa", str(qa), "--retain-unmapped",
        "--output", str(output), "--report", str(report)])
    build_main()
    rows = read_records(output)
    assert [row["input_id"] for row in rows] == [row["id"] for row in targets]
    assert [row["gold_answers"] for row in rows] == [row["gold_answers"] for row in targets]
    assert [row["question"] for row in rows] == [row["question"] for row in targets]
    assert [row["evidence_status"] for row in rows] == ["unmapped", "mapped", "unmapped", "unmapped"]
    assert rows[0]["evidence_reason"] == "ambiguous_supporting_title"
    assert rows[0]["source_id"] == "ambiguous"
    assert rows[1]["gold_doc_ids"] == ["1"] and rows[1]["source_id"] == "yes"
    assert rows[2]["evidence_reason"] == "question_not_found"
    assert rows[3]["evidence_reason"] == "missing_source_annotations"
    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["rows_written"] == 4 and summary["unmapped_rows"] == 3


def test_all_unmapped_is_still_a_valid_retained_output():
    targets = [{"id": "q", "dataset": "wiki18", "question": "Q", "gold_answers": ["A"]}]
    report = {"missing_sources": [{"dataset": "wiki18"}]}
    rows = retain_questions(targets, [], {}, report, "train")
    assert len(rows) == 1 and rows[0]["gold_doc_ids"] == []
    assert rows[0]["evidence_status"] == "unmapped"
    assert report["datasets"]["wiki18"]["mapped"] == 0


@pytest.mark.parametrize(("status", "ids", "valid"), [
    ("mapped", ["d"], True), ("unmapped", [], True),
    ("mapped", [], False), ("unmapped", ["d"], False), ("unknown", [], False),
])
def test_validator_enforces_evidence_status(tmp_path, status, ids, valid):
    corpus = tmp_path / "retrieval" / "wiki18"
    corpus.mkdir(parents=True)
    database(corpus / "passages.sqlite", [("d", "Page", "T")])
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"id": "wiki18::q", "dataset": "wiki18", "split": "train",
        "question": "Q", "gold_answers": ["A"], "gold_doc_ids": ids, "evidence_status": status}) + "\n", encoding="utf-8")
    report, _ = validate([source], corpus.parent)
    assert report["valid"] is valid


def test_validation_emits_exact_mapped_subset_and_manifest(tmp_path, monkeypatch):
    corpus = tmp_path / "retrieval" / "wiki18"
    corpus.mkdir(parents=True)
    database(corpus / "passages.sqlite", [("d", "Page", "T")])
    rows = [{"id": f"wiki18::{status}", "dataset": "wiki18", "split": "train", "question": status,
        "gold_answers": ["A"], "gold_doc_ids": ids, "evidence_status": status}
        for status, ids in [("unmapped", []), ("mapped", ["d"])]]
    source, subset = tmp_path / "train.jsonl", tmp_path / "train_mapped.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    inventory = tmp_path / "inventory.json"
    inventory.write_text('{"files": []}', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(sys, "argv", ["validate", "--input", str(source), "--retrieval-root", str(corpus.parent),
        "--manifest", str(manifest), "--report", str(tmp_path / "report.json"), "--retrieval-revision", "pinned",
        "--source-inventory", str(inventory), "--mapped-train-output", str(subset)])
    validate_main()
    assert read_records(subset) == [rows[1]]
    result = json.loads(manifest.read_text(encoding="utf-8"))
    assert result["files"]["train.jsonl"]["rows"] == 2
    assert result["files"]["train.jsonl"]["unmapped_rows"] == 1
    assert result["files"]["train_mapped.jsonl"]["sha256"] == sha256(subset)
    assert result["recall_train_file"] == "train_mapped.jsonl"
