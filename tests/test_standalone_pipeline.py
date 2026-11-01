import json
import zipfile

import pytest
from test_builder import database

from grasp_gold_evidence_builder.handoff import main as package_main
from grasp_gold_evidence_builder.pipeline import main as pipeline_main
from grasp_gold_evidence_builder.pipeline import retrieval_revision
from grasp_gold_evidence_builder.sources import read_records, sha256


def inputs(tmp_path):
    raw, retrieval = tmp_path / "raw", tmp_path / "retrieval"
    targets = []
    for name, dataset in (("2wiki", "2wiki_ircot"), ("hotpotqa", "hotpotqa_abstracts_2017"), ("nq", "wiki18")):
        passages = []
        for split in ("train", "dev"):
            title, question = f"Article {name} {split}", f"Question {name} {split}?"
            identifier = f"{name}-{split}"
            text = " ".join(f"word{i}" for i in range(25)) + f" {name} {split}"
            if name == "nq":
                row = {"example_id": identifier, "question": question, "short_answers": ["A"],
                       "context": text, "title": title, "doc_url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")}
            else:
                row = {"_id": identifier, "question": question, "answer": "A", "supporting_facts": [[title, 0]]}
            destination = raw / name / f"{split}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps([row]), encoding="utf-8")
            passages.append((identifier, title, text))
            if split == "train":
                targets.append({"id": f"input-{name}", "dataset": dataset, "question": question, "gold_answers": ["A"]})
        directory = retrieval / dataset
        directory.mkdir(parents=True)
        database(directory / "passages.sqlite", passages)
    # A failed teacher trace is still usable QA when retaining all input questions.
    targets.insert(1, {"id": "unmapped", "dataset": "wiki18", "question": "Unmatched?", "gold_answers": ["A"], "status": "error"})
    qa = tmp_path / "original.jsonl"
    qa.write_text("".join(json.dumps(row) + "\n" for row in targets), encoding="utf-8")
    return raw, retrieval, qa, targets


def test_build_and_package_without_other_repositories(tmp_path):
    raw, retrieval, qa, targets = inputs(tmp_path)
    original_hash = sha256(qa)
    output = tmp_path / "work"
    pipeline_main(["--qa", str(qa), "--raw-root", str(raw), "--retrieval-root", str(retrieval),
                   "--retrieval-revision", "a" * 40, "--output-dir", str(output), "--expected-train-rows", "4"])
    rows = read_records(output / "train.jsonl")
    assert [row["input_id"] for row in rows] == [row["id"] for row in targets]
    assert sha256(qa) == original_hash
    assert len(rows) == 4 and rows[1]["evidence_status"] == "unmapped"
    assert read_records(output / "train_mapped.jsonl") == [row for row in rows if row["evidence_status"] == "mapped"]
    archive = tmp_path / "handoff.zip"
    package_main(["--evidence-dir", str(output), "--output", str(archive)])
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert len(package.namelist()) == 10
        assert all(name.startswith("data/grasp-gold-evidence/") for name in package.namelist())
        assert package.read("data/grasp-gold-evidence/train.jsonl") == (output / "train.jsonl").read_bytes()
    # Repackaging unchanged artifacts must not invalidate an already bound smoke gate.
    manifest_hash = sha256(output / "manifest.json")
    package_main(["--evidence-dir", str(output), "--output", str(archive)])
    assert sha256(output / "manifest.json") == manifest_hash
    (output / "train.jsonl").write_text("tampered QA\n", encoding="utf-8")
    archive_hash = sha256(archive)
    with pytest.raises(ValueError, match="evidence changed after validation"):
        package_main(["--evidence-dir", str(output), "--output", str(archive)])
    assert sha256(archive) == archive_hash


def test_pipeline_wrong_row_count_fails_before_writing(tmp_path):
    raw, retrieval, qa, _ = inputs(tmp_path)
    output = tmp_path / "work"
    with pytest.raises(ValueError, match="expected 9000 QA rows, found 4"):
        pipeline_main(["--qa", str(qa), "--raw-root", str(raw), "--retrieval-root", str(retrieval),
                       "--output-dir", str(output), "--expected-train-rows", "9000"])
    assert not output.exists()


def test_revision_metadata_conflict_is_rejected(tmp_path):
    for dataset in ("2wiki_ircot", "hotpotqa_abstracts_2017", "wiki18"):
        path = tmp_path / ".cache/huggingface/download" / dataset / "passages.sqlite.metadata"
        path.parent.mkdir(parents=True)
        path.write_text("a" * 40 + "\n", encoding="utf-8")
    assert retrieval_revision(tmp_path, None) == "a" * 40
    with pytest.raises(ValueError, match="conflicts"):
        retrieval_revision(tmp_path, "b" * 40)
