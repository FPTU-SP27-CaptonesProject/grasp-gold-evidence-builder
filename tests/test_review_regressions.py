import json

import pytest
from test_builder import database

from grasp_gold_evidence_builder.adapters.common import processed_ids
from grasp_gold_evidence_builder.align_nq import align_one
from grasp_gold_evidence_builder.mapping import resolve_evidence
from grasp_gold_evidence_builder.normalize import normalized_title
from grasp_gold_evidence_builder.passages import load_lookup
from grasp_gold_evidence_builder.sources import read_records
from grasp_gold_evidence_builder.validate import validate


def test_punctuation_distinguishes_articles(tmp_path):
    path = tmp_path / "passages.sqlite"
    database(path, [("c", "C", "Text"), ("cpp", "C++", "Text")])
    lookup = load_lookup(path)
    ids, status = resolve_evidence({"source_doc_ids": set(), "supporting_titles": {"C++"}}, lookup)
    assert ids == ["cpp"] and status == "mapped"
    assert normalized_title("C#") != normalized_title("C++") != normalized_title("C")
    assert normalized_title("A_&amp;_B") == normalized_title("A & B")


def test_passage_database_path_can_contain_uri_characters(tmp_path):
    path = tmp_path / "corpus #1.sqlite"
    database(path, [("d", "Title", "Text")])
    assert load_lookup(path).doc_ids == frozenset({"d"})


def test_incomplete_processed_contexts_are_not_partially_labeled():
    with pytest.raises(ValueError, match="partial labels"):
        processed_ids({"gold_ctxs": [{"id": "first"}, {"title": "Missing passage ID"}]})


def test_nq_requires_nonempty_provenance_and_matching_candidate_title():
    text = " ".join(f"word{i}" for i in range(25))
    row = {"title": "", "doc_url": "", "context": text}
    assert align_one(row, [{"doc_id": "wrong", "title": "", "text": text}])[1] == "missing_document_provenance"
    row.update(title="Page", doc_url="https://en.wikipedia.org/wiki/Page")
    assert not align_one(row, [{"doc_id": "wrong", "title": "Other", "text": text}])[0]


def test_bom_reading_and_invalid_rows_count(tmp_path):
    corpus = tmp_path / "wiki18"
    corpus.mkdir()
    database(corpus / "passages.sqlite", [("d", "Page", "Text")])
    path = tmp_path / "train.jsonl"
    row = {"id": "wiki18::q", "dataset": "wiki18", "split": "wrong", "question": "Q",
           "gold_answers": [None], "gold_doc_ids": ["d", "d"], "evidence_status": "mapped"}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8-sig")
    assert read_records(path) == [row]
    report, _ = validate([path], tmp_path)
    assert not report["valid"] and report["invalid_rows"] == 1
    assert report["error_count"] == 3
