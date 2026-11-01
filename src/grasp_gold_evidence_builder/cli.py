"""Command-line entry points for inventory, build, and validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import build_manifest, write_json, write_jsonl
from .mapping import build_adapted_split, default_corpora, source_requirements
from .passages import load_lookup
from .questions import load_questions, match_questions, retain_questions
from .sources import inventory, sha256
from .validate import validate


def _source_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be DATASET=PATH")
    dataset, raw_path = value.split("=", 1)
    if not dataset or not raw_path:
        raise argparse.ArgumentTypeError("--source must be DATASET=PATH")
    return dataset, Path(raw_path)


def inventory_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inventory raw benchmark annotation files")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = inventory(args.raw_root)
    write_json(args.output, result)
    print(json.dumps({"files": len(result["files"]), "output": str(Path(args.output).resolve())}))


def build_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a fail-closed canonical gold-evidence JSONL split")
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--retrieval-root", required=True)
    parser.add_argument("--source", type=_source_argument, action="append", help="Optional DATASET=PATH override; repeat per corpus")
    parser.add_argument("--qa", action="append", help="Only enrich these existing QA/teacher JSONL questions; repeat per input file")
    parser.add_argument("--retain-unmapped", action="store_true", help="Keep every --qa row with explicit mapped/unmapped evidence_status")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    overrides = dict(args.source or [])
    corpus_paths = default_corpora(args.retrieval_root)
    unknown_sources = set(overrides) - set(corpus_paths)
    if unknown_sources or len(overrides) != len(args.source or []):
        raise ValueError(f"unknown or duplicate --source keys: {sorted(unknown_sources)}")
    targets = load_questions(args.qa, keep_all=args.retain_unmapped) if args.qa else None
    if args.retain_unmapped and targets is None:
        parser.error("--retain-unmapped requires --qa")
    directories = {"2wiki_ircot": "2wiki", "hotpotqa_abstracts_2017": "hotpotqa", "wiki18": "nq"}
    specs, joins, missing_sources = [], {}, []
    for dataset, database in corpus_paths.items():
        requested = [row for row in targets if row["dataset"] == dataset] if targets is not None else None
        if requested == []:
            continue
        source = overrides.get(dataset, Path(args.raw_root) / directories[dataset] / f"{args.split}.json")
        if source.is_file():
            adapted, titles, doc_ids = source_requirements(dataset, source)
            if requested is not None:
                adapted, joins[dataset] = match_questions(requested, adapted)
                titles = {str(title) for row in adapted for title in row["supporting_titles"]}
                doc_ids = {str(doc_id) for row in adapted for doc_id in row["source_doc_ids"]}
            if not adapted:
                continue
            specs.append((dataset, source, load_lookup(database, required_titles=titles, required_doc_ids=doc_ids), adapted))
        elif requested is not None:
            missing_sources.append({"dataset": dataset, "questions": len(requested), "expected_source": str(source), "status": "missing_source_annotations"})
    if not specs and not args.retain_unmapped:
        write_json(args.report, {"rows_written": 0, "question_join": joins, "missing_sources": missing_sources})
        raise ValueError("no eligible source rows; inspect question_join/missing_sources in the report")
    rows, report = build_adapted_split(specs, args.split)
    report["missing_sources"] = missing_sources
    if args.retain_unmapped:
        rows = retain_questions(targets, rows, joins, report, args.split)
    elif targets is not None:
        input_order = {(row["dataset"], row["id"]): index for index, row in enumerate(targets)}
        canonical_order = {
            f"{dataset}::{match['source_id']}": input_order[(dataset, match["input_id"])]
            for dataset, join in joins.items() for match in join["matches"]
        }
        rows.sort(key=lambda row: canonical_order[row["id"]])
    report["question_join"] = joins
    report["missing_sources"] = missing_sources
    report["input_mode"] = "existing_questions" if targets is not None else "all_source_questions"
    report["qa_inputs"] = [{"path": str(Path(path).resolve()), "sha256": sha256(path)} for path in args.qa or []]
    if not rows:
        write_json(args.report, report)
        raise SystemExit("no rows mapped; inspect report before building a release")
    write_jsonl(args.output, rows)
    write_json(args.report, report)
    print(json.dumps({"rows_written": len(rows), "report": str(Path(args.report).resolve())}))


def questions_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract only existing questions/answers from QA files or teacher traces")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rows = load_questions(args.input)
    write_jsonl(args.output, rows)
    print(json.dumps({"questions": len(rows), "output": str(Path(args.output).resolve())}))


def validate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate canonical gold-evidence JSONL files against passages.sqlite")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--retrieval-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--retrieval-revision", required=True)
    parser.add_argument("--source-inventory", required=True, help="Output of grasp-gold-inventory for the exact raw files")
    parser.add_argument("--mapped-train-output", help="Also emit the exact mapped-only train subset and include it in the manifest")
    args = parser.parse_args(argv)
    input_paths = [Path(value) for value in args.input]
    if len({path.name for path in input_paths}) != len(input_paths):
        parser.error("input filenames must be unique in the manifest")
    report, lookups = validate(args.input, args.retrieval_root)
    write_json(args.report, report)
    if not report["valid"]:
        raise SystemExit(f"validation failed; report: {Path(args.report).resolve()}")
    if args.mapped_train_output:
        from .sources import read_records
        train_paths = [path for path in input_paths if path.stem == "train"]
        if len(train_paths) != 1:
            parser.error("--mapped-train-output requires exactly one train.jsonl input")
        subset_path = Path(args.mapped_train_output)
        if subset_path.name in {path.name for path in input_paths}:
            parser.error("mapped subset filename must differ from validation inputs")
        subset = [row for row in read_records(train_paths[0]) if row.get("evidence_status", "mapped") == "mapped"]
        if not subset:
            raise ValueError("no mapped training rows; cannot create a recall-aware train subset")
        write_jsonl(subset_path, subset)
        input_paths.append(subset_path)
    inventory_path = Path(args.source_inventory)
    source_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    write_json(args.manifest, build_manifest(input_paths, lookups, args.retrieval_revision, source_inventory))
    print(json.dumps({"valid": True, "rows": report["rows"], "manifest": str(Path(args.manifest).resolve())}))
