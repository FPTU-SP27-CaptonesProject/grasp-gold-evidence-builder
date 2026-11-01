from __future__ import annotations

import json
import sqlite3

from grasp_gold_evidence_builder.mapping import CorpusSpec, build_split
from grasp_gold_evidence_builder.passages import load_lookup
from grasp_gold_evidence_builder.validate import validate


def database(path, rows):
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE passages (doc_id TEXT, title TEXT, text TEXT)")
    connection.executemany("INSERT INTO passages VALUES (?, ?, ?)", rows)
    connection.commit()
    connection.close()


def test_hotpot_titles_map_only_when_unique(tmp_path):
    raw = tmp_path / "hotpot.json"
    raw.write_text(json.dumps([{"_id": "h1", "question": "Q?", "answer": "A", "supporting_facts": [["One", 0], ["Two", 1]]}]), encoding="utf-8")
    db = tmp_path / "passages.sqlite"
    database(db, [("1", "One", ""), ("2", "Two", "")])
    rows, report = build_split([CorpusSpec("hotpotqa_abstracts_2017", raw, load_lookup(db))], "train")
    assert rows == [{"id": "hotpotqa_abstracts_2017::h1", "dataset": "hotpotqa_abstracts_2017", "split": "train", "question": "Q?", "gold_answers": ["A"], "gold_doc_ids": ["1", "2"], "evidence_status": "mapped"}]
    assert report["datasets"]["hotpotqa_abstracts_2017"]["mapped"] == 1


def test_ambiguous_title_is_excluded(tmp_path):
    raw = tmp_path / "2wiki.json"
    raw.write_text(json.dumps([{"_id": "w1", "question": "Q?", "answer": "A", "supporting_facts": [["Duplicate", 0]]}]), encoding="utf-8")
    db = tmp_path / "passages.sqlite"
    database(db, [("1", "Duplicate", ""), ("2", "Duplicate", "")])
    rows, report = build_split([CorpusSpec("2wiki_ircot", raw, load_lookup(db))], "train")
    assert rows == []
    assert report["datasets"]["2wiki_ircot"]["ambiguous_supporting_title"] == 1


def test_validator_rejects_invalid_document_id(tmp_path):
    retrieval = tmp_path / "retrieval"
    for corpus in ("2wiki_ircot", "hotpotqa_abstracts_2017", "wiki18"):
        directory = retrieval / corpus
        directory.mkdir(parents=True)
        database(directory / "passages.sqlite", [("valid", "Title", "")])
    rows = tmp_path / "train.jsonl"
    rows.write_text(json.dumps({"id": "hotpotqa_abstracts_2017::x", "dataset": "hotpotqa_abstracts_2017", "split": "train", "question": "Q", "gold_answers": ["A"], "gold_doc_ids": ["invalid"]}) + "\n", encoding="utf-8")
    report, _ = validate([rows], retrieval)
    assert not report["valid"]
    assert report["invalid_rows"] == 1
