"""Map original NQ gold spans to a pinned Wiki18 corpus by exact token spans.

No answers, retrieval rankings, or generated triples are used to discover evidence.
Title/URL selects the original article; complete contiguous gold/passage token
containment verifies the passage. Offsets refer to the documented normalized tokens,
not the unavailable original HTML offsets or a guessed Wikipedia revision match.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .manifest import write_json
from .normalize import nonempty_strings, normalized_title
from .questions import load_questions, match_questions
from .sources import read_records, sha256


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", html.unescape(text)).casefold())


def span(haystack: list[str], needle: list[str]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    for index in range(len(haystack) - len(needle) + 1):
        if haystack[index:index + len(needle)] == needle:
            return index
    return None


def adapt_gold(record: dict) -> dict:
    identifier = record.get("example_id", record.get("id", record.get("_id")))
    return {
        "source_id": str(identifier) if identifier is not None else None,
        "question": record.get("question") or "",
        "gold_answers": nonempty_strings(record.get("short_answers", record.get("answers", []))),
        "source_doc_ids": set(), "supporting_titles": set(),
        "title": record.get("title") or "", "context": record.get("context") or "",
        "doc_url": record.get("doc_url") or "",
    }


def url_title(url: str) -> str:
    parsed = urlparse(html.unescape(url))
    if parsed.hostname not in ("en.wikipedia.org", "www.en.wikipedia.org"):
        return ""
    query = parse_qs(parsed.query)
    return query.get("title", [unquote(parsed.path.split("/wiki/", 1)[-1]) if "/wiki/" in parsed.path else ""])[0]


def align_one(row: dict, candidates: list[dict]) -> tuple[list[dict], str]:
    if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("title", "doc_url", "context")):
        return [], "missing_document_provenance"
    original_title = normalized_title(url_title(row["doc_url"]))
    if not original_title or original_title != normalized_title(row["title"]):
        return [], "document_provenance_mismatch"
    gold = tokens(row["context"])
    if len(gold) < 20:
        return [], "gold_span_too_short"
    matched = []
    for passage in candidates:
        if normalized_title(passage.get("title") or "") != original_title:
            continue
        passage_tokens = tokens(passage["text"])
        if len(passage_tokens) < 20:
            continue
        position = span(passage_tokens, gold)
        if position is not None:
            proof = {"method": "gold_span_in_passage", "gold_token_span": [0, len(gold)], "passage_token_span": [position, position + len(gold)]}
        else:
            position = span(gold, passage_tokens)
            if position is None:
                continue
            proof = {"method": "passage_in_gold_span", "gold_token_span": [position, position + len(passage_tokens)], "passage_token_span": [0, len(passage_tokens)]}
        matched.append({"id": passage["doc_id"], "title": passage["title"], "alignment": proof, "text_sha256": hashlib.sha256(passage["text"].encode("utf-8")).hexdigest()})
    if not matched:
        return [], "no_exact_gold_span_alignment" if candidates else "title_not_in_corpus"
    # Repeated copies of the same aligned gold interval cannot choose a unique ID.
    intervals = [tuple(item["alignment"]["gold_token_span"]) for item in matched]
    if len(intervals) != len(set(intervals)):
        return [], "ambiguous_duplicate_gold_span"
    return matched, "mapped"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify original NQ gold spans against Wiki18 passage text")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--retrieval-root", required=True)
    parser.add_argument("--qa", action="append", help="Restrict train annotations to existing mixed QA questions")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    sources, joins = {}, {}
    targets = [row for row in load_questions(args.qa, keep_all=True) if row["dataset"] == "wiki18"] if args.qa else None
    for split in ("train", "dev"):
        path = Path(args.raw_root) / "nq" / f"{split}.json"
        adapted = [adapt_gold(row) for row in read_records(path)]
        if split == "train" and targets is not None:
            adapted, joins[split] = match_questions(targets, adapted)
        sources[split] = (path, adapted)
    wanted = {normalized_title(row["title"]) for _, rows in sources.values() for row in rows}
    database = Path(args.retrieval_root) / "wiki18" / "passages.sqlite"
    candidates = defaultdict(list)
    print(f"Scanning Wiki18 once for {len(wanted)} evidence titles", flush=True)
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        for doc_id, title, text in connection.execute("SELECT doc_id,title,text FROM passages"):
            key = normalized_title(title or "")
            if key in wanted:
                candidates[key].append({"doc_id": str(doc_id), "title": title, "text": text})
    corpus_hash = sha256(database)
    report = {
        "method": "verified_normalized_token_containment_v1",
        "normalization": "HTML unescape; Unicode NFKC; casefold; regex word tokens; contiguous exact match; minimum 20 tokens",
        "retrieval_corpus_sha256": corpus_hash, "original_html_offsets_available": False,
        "revision_claim": "Exact local passage text is verified; original Wikipedia oldid is preserved, not inferred for the corpus",
        "question_join": joins, "splits": {},
    }
    for split, (source, records) in sources.items():
        output, proofs, rejected = [], [], []
        counts = Counter()
        for row in records:
            contexts, status = align_one(row, candidates.get(normalized_title(row["title"]), []))
            counts[status] += 1
            if contexts:
                output.append({"id": row["source_id"], "question": row["question"], "gold_answers": row["gold_answers"], "gold_ctxs": contexts})
                proofs.append({"source_id": row["source_id"], "doc_url": row["doc_url"], "gold_context_sha256": hashlib.sha256(row["context"].encode("utf-8")).hexdigest(), "passages": contexts})
            else:
                rejected.append({"id": row["source_id"], "question": row["question"], "title": row["title"], "status": status})
        destination = Path(args.output_dir) / f"{split}.json"
        write_json(destination, output)
        report["splits"][split] = {"source": str(source), "source_sha256": sha256(source), "input_rows": len(records), "counts": dict(counts), "unmapped_examples": rejected[:30], "unmapped_rows": rejected, "alignment_proofs": proofs, "output_sha256": sha256(destination)}
        print(f"NQ {split}: {dict(counts)}", flush=True)
    write_json(Path(args.output_dir) / "alignment_report.json", report)


if __name__ == "__main__":
    main()
