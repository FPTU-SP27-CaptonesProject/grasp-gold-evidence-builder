from grasp_gold_evidence_builder.align_nq import align_one


def record(context):
    return {"context": context, "title": "Article", "doc_url": "https://en.wikipedia.org/w/index.php?title=Article&oldid=123"}


def test_exact_gold_span_preserves_verified_offsets():
    words = [f"word{index}" for index in range(40)]
    gold = " ".join(words[10:30])
    mapped, status = align_one(record(gold), [{"doc_id": "local", "title": "Article", "text": " ".join(words)}])
    assert status == "mapped"
    assert mapped[0]["alignment"]["passage_token_span"] == [10, 30]


def test_partial_overlap_and_duplicate_copy_are_rejected():
    words = [f"word{index}" for index in range(40)]
    gold = " ".join(words)
    passage = {"doc_id": "one", "title": "Article", "text": gold}
    assert align_one(record(gold), [passage, {**passage, "doc_id": "two"}])[1] == "ambiguous_duplicate_gold_span"
    changed = " ".join(words[:25] + ["different"] + words[26:])
    assert align_one(record(gold), [{**passage, "text": changed}])[1] == "no_exact_gold_span_alignment"


def test_wrong_document_provenance_is_rejected():
    row = record(" ".join(["word"] * 20))
    row["doc_url"] = "https://en.wikipedia.org/wiki/Other"
    assert align_one(row, [])[1] == "document_provenance_mismatch"
