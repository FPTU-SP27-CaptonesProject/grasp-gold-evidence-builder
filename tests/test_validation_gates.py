import json

from test_builder import database

from grasp_gold_evidence_builder.validate import validate


def test_empty_split_is_rejected(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text("", encoding="utf-8")
    report, _ = validate([source], tmp_path)
    assert not report["valid"]
    assert report["rows"] == 0


def test_null_answer_and_invalid_split_are_rejected(tmp_path):
    corpus = tmp_path / "retrieval" / "hotpotqa_abstracts_2017"
    corpus.mkdir(parents=True)
    database(corpus / "passages.sqlite", [("d", "Title", "Text")])
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"id": "hotpotqa_abstracts_2017::x", "dataset": "hotpotqa_abstracts_2017", "split": "wrong", "question": "Q", "gold_answers": [None], "gold_doc_ids": ["d"]}) + "\n", encoding="utf-8")
    report, _ = validate([source], corpus.parent)
    assert not report["valid"]
    assert {error["error"] for error in report["errors"]} == {"invalid or mismatched split", "empty question or gold_answers"}


def test_same_question_cannot_leak_across_splits(tmp_path):
    corpus = tmp_path / "retrieval" / "hotpotqa_abstracts_2017"
    corpus.mkdir(parents=True)
    database(corpus / "passages.sqlite", [("d", "Title", "Text")])
    files = []
    for split in ("train", "dev"):
        source = tmp_path / f"{split}.jsonl"
        source.write_text(json.dumps({"id": f"hotpotqa_abstracts_2017::{split}", "dataset": "hotpotqa_abstracts_2017", "split": split, "question": "Q", "gold_answers": ["A"], "gold_doc_ids": ["d"]}) + "\n", encoding="utf-8")
        files.append(source)
    report, _ = validate(files, corpus.parent)
    assert not report["valid"]
    assert report["invalid_doc_ids"] == 0
    assert report["errors"][0]["error"] == "question overlaps another split"
