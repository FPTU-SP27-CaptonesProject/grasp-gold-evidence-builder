"""Build the full QA artifact and mapped subset on Windows or Linux, without trainer code."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .align_nq import main as align_main
from .cli import build_main, inventory_main, validate_main
from .mapping import default_corpora
from .questions import load_questions
from .sources import read_records


def retrieval_revision(root: Path, explicit: str | None) -> str:
    """Use an explicit pinned commit or infer one from all three HF download records."""
    records = [root / ".cache/huggingface/download" / dataset / "passages.sqlite.metadata"
               for dataset in default_corpora(root)]
    revisions = set()
    for record in records:
        if record.is_file():
            lines = record.read_text(encoding="utf-8").splitlines()
            if not lines or not re.fullmatch(r"[0-9a-f]{40}", lines[0]):
                raise ValueError(f"invalid retrieval revision metadata: {record}")
            revisions.add(lines[0])
    if explicit is not None and not re.fullmatch(r"[0-9a-f]{40}", explicit):
        raise ValueError("--retrieval-revision must be a pinned 40-character commit SHA")
    if len(revisions) > 1 or (explicit and revisions and revisions != {explicit}):
        raise ValueError("retrieval metadata differs across corpora or conflicts with --retrieval-revision")
    if explicit:
        return explicit
    if len(revisions) == 1 and all(path.is_file() for path in records):
        return next(iter(revisions))
    raise ValueError("set --retrieval-revision to the exact HF commit, or retain metadata for all three corpora")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", required=True, help="Original mixed QA JSONL; all usable rows are retained")
    parser.add_argument("--retrieval-root", required=True)
    parser.add_argument("--retrieval-revision", help="Pinned retrieval commit; otherwise infer from HF metadata")
    parser.add_argument("--raw-root", default="raw")
    parser.add_argument("--output-dir", default="work/exact-grpo")
    parser.add_argument("--expected-train-rows", type=int, help="Fail before building if the QA row count is unexpected")
    args = parser.parse_args(argv)
    qa, raw, root, output = Path(args.qa), Path(args.raw_root), Path(args.retrieval_root), Path(args.output_dir)
    targets = load_questions([qa], keep_all=True)
    if args.expected_train_rows is not None and len(targets) != args.expected_train_rows:
        raise ValueError(f"expected {args.expected_train_rows} QA rows, found {len(targets)}")
    revision = retrieval_revision(root, args.retrieval_revision)
    required = [raw / dataset / f"{split}.json" for dataset in ("2wiki", "hotpotqa", "nq") for split in ("train", "dev")]
    required.extend(default_corpora(root).values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input artifacts: {missing}; download raw annotations with grasp-gold-fetch and provide the pinned retrieval corpus")
    inventory_main(["--raw-root", str(raw), "--output", str(output / "source_inventory.json")])
    aligned = output / "nq-aligned"
    align_main(["--raw-root", str(raw), "--retrieval-root", str(root), "--qa", str(qa), "--output-dir", str(aligned)])
    common = ["--raw-root", str(raw), "--retrieval-root", str(root)]
    build_main(["--split", "train", *common, "--qa", str(qa), "--retain-unmapped",
                "--source", f"wiki18={aligned / 'train.json'}", "--output", str(output / "train.jsonl"),
                "--report", str(output / "reports/train_mapping_report.json")])
    build_main(["--split", "dev", *common, "--source", f"wiki18={aligned / 'dev.json'}",
                "--output", str(output / "dev.jsonl"), "--report", str(output / "reports/dev_mapping_report.json")])
    validate_main(["--input", str(output / "train.jsonl"), "--input", str(output / "dev.jsonl"),
                   "--retrieval-root", str(root), "--retrieval-revision", revision,
                   "--source-inventory", str(output / "source_inventory.json"),
                   "--manifest", str(output / "manifest.json"), "--report", str(output / "reports/validation_report.json"),
                   "--mapped-train-output", str(output / "train_mapped.jsonl")])
    rows = read_records(output / "train.jsonl")
    if len(rows) != len(targets):
        raise RuntimeError("output train row count differs from input QA")
    print(json.dumps({"output_dir": str(output.resolve()), "train_rows": len(rows),
                      "mapped_rows": sum(row["evidence_status"] == "mapped" for row in rows),
                      "next": "grasp-gold-package; then real SFT rollout smoke before primary GRPO"}))


if __name__ == "__main__":
    main()
