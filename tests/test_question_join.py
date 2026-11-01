import json
import sys

from test_builder import database

from grasp_gold_evidence_builder.adapters.hotpotqa import adapt
from grasp_gold_evidence_builder.cli import build_main
from grasp_gold_evidence_builder.questions import load_questions, match_questions


def annotation(identifier, question="Who sang?", answer="Singer"):
    return adapt({"_id": identifier, "question": question, "answer": answer, "supporting_facts": [["Page", 0]]})


def test_synthetic_id_matches_question_and_answer():
    targets = [{"id": "train_36328", "question": "Who sang?", "gold_answers": ["Singer"]}]
    selected, report = match_questions(targets, [annotation("original-id")])
    assert selected[0]["source_id"] == "original-id"
    assert report["matches"][0]["method"] == "question_answer"


def test_id_collision_never_bypasses_question_or_answer_check():
    target = {"id": "same-id", "question": "Who sang?", "gold_answers": ["Singer"]}
    selected, report = match_questions([target], [annotation("same-id", question="Different?")])
    assert not selected
    assert report["unmatched_questions"][0]["status"] == "question_not_found"
    selected, report = match_questions([target], [annotation("same-id", answer="Other")])
    assert not selected
    assert report["unmatched_questions"][0]["status"] == "answer_mismatch"


def test_ambiguous_questions_are_not_guessed():
    target = {"id": "synthetic", "question": "Who sang?", "gold_answers": ["Singer"]}
    selected, report = match_questions([target], [annotation("one"), annotation("two")])
    assert not selected
    assert report["unmatched_questions"][0]["status"] == "ambiguous_question"


def test_processed_gold_contexts_supported_without_supporting_facts():
    row = adapt({"id": "x", "question": "Q", "answers": ["answer"], "gold_ctxs": [{"id": "direct-id", "title": "Page"}]})
    assert row["source_doc_ids"] == {"direct-id"}
    assert row["supporting_titles"] == set()


def test_thin_question_extraction_discards_teacher_context(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps({"qid": "train_1", "dataset": "hotpotqa", "question": "Q", "gold_answers": ["A"], "context": [{"doc_id": "not-a-gold-label"}]}) + "\n", encoding="utf-8")
    row = load_questions([path])[0]
    assert row["dataset"] == "hotpotqa_abstracts_2017"
    assert "context" not in row and "gold_doc_ids" not in row


def test_cli_maps_only_requested_questions_and_reports_missing_nq(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "hotpotqa"
    raw.mkdir(parents=True)
    (raw / "train.json").write_text(json.dumps([{"_id": key, "question": question, "answer": "A", "supporting_facts": [["Page", 0]]} for key, question in [("one", "Requested?"), ("two", "Unrequested?")]]), encoding="utf-8")
    retrieval = tmp_path / "retrieval" / "hotpotqa_abstracts_2017"
    retrieval.mkdir(parents=True)
    database(retrieval / "passages.sqlite", [("doc", "Page", "Text")])
    qa = tmp_path / "qa.jsonl"
    qa.write_text("\n".join(json.dumps(row) for row in [{"id": "synthetic", "dataset": "hotpotqa", "question": "Requested?", "gold_answers": ["A"]}, {"id": "nq-id", "dataset": "nq", "question": "NQ?", "gold_answers": ["A"]}]) + "\n", encoding="utf-8")
    output, report = tmp_path / "train.jsonl", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["grasp-gold-build", "--split", "train", "--raw-root", str(raw.parent), "--retrieval-root", str(retrieval.parent), "--qa", str(qa), "--output", str(output), "--report", str(report)])
    build_main()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["id"] == "hotpotqa_abstracts_2017::one"
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["missing_sources"][0]["dataset"] == "wiki18"


def test_mixed_corpus_input_order_is_preserved(tmp_path, monkeypatch):
    corpora = [("hotpotqa", "hotpotqa_abstracts_2017"), ("2wiki", "2wiki_ircot")]
    targets = []
    for short, canonical in corpora:
        raw = tmp_path / "raw" / short
        raw.mkdir(parents=True)
        (raw / "train.json").write_text(json.dumps([{"_id": short, "question": short, "answer": "A", "supporting_facts": [["Page", 0]]}]), encoding="utf-8")
        corpus = tmp_path / "retrieval" / canonical
        corpus.mkdir(parents=True)
        database(corpus / "passages.sqlite", [("doc", "Page", "Text")])
        targets.append({"id": short, "dataset": canonical, "question": short, "gold_answers": ["A"]})
    qa = tmp_path / "qa.jsonl"
    qa.write_text("\n".join(json.dumps(row) for row in targets) + "\n", encoding="utf-8")
    output = tmp_path / "train.jsonl"
    monkeypatch.setattr(sys, "argv", ["grasp-gold-build", "--split", "train", "--raw-root", str(tmp_path / "raw"), "--retrieval-root", str(tmp_path / "retrieval"), "--qa", str(qa), "--output", str(output), "--report", str(tmp_path / "report.json")])
    build_main()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["dataset"] for row in rows] == [row["dataset"] for row in targets]
