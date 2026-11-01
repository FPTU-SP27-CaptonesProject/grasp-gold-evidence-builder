"""Package verified QA independently; optionally include an explicitly supplied trainer."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from .manifest import write_json
from .sources import read_records, sha256

DATA_FILES = ("train.jsonl", "train_mapped.jsonl", "dev.jsonl", "manifest.json", "README.md", "source_inventory.json")
REPORT_FILES = ("train_mapping_report.json", "dev_mapping_report.json", "validation_report.json", "nq_alignment_report.json")


def prepare_evidence(evidence: Path, nq_report: Path | None = None) -> list[Path]:
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads((evidence / "reports/validation_report.json").read_text(encoding="utf-8"))
    if validation.get("valid") is not True or validation.get("invalid_doc_ids") != 0:
        raise ValueError("canonical evidence validation must pass before packaging")
    if not {"train.jsonl", "dev.jsonl", "train_mapped.jsonl"} <= set(manifest.get("files", {})):
        raise ValueError("manifest must include full train, mapped train and dev")
    for name, info in manifest["files"].items():
        if name not in DATA_FILES:
            raise ValueError(f"unexpected canonical filename in manifest: {name}")
        if sha256(evidence / name) != info.get("sha256"):
            raise ValueError(f"evidence changed after validation: {name}")
        if len(read_records(evidence / name)) != info.get("rows"):
            raise ValueError(f"manifest row count mismatch: {name}")
    for name in ("train.jsonl", "dev.jsonl"):
        if validation.get("files", {}).get(name, {}).get("sha256") != manifest["files"][name]["sha256"]:
            raise ValueError("validation report does not bind the current QA hashes; run grasp-gold-validate again")
    train = read_records(evidence / "train.jsonl")
    mapped = [row for row in train if row.get("evidence_status") == "mapped"]
    if read_records(evidence / "train_mapped.jsonl") != mapped:
        raise ValueError("train_mapped.jsonl is not the exact mapped subset")
    for name in ("train.jsonl", "train_mapped.jsonl", "dev.jsonl"):
        rows = read_records(evidence / name)
        if any(row.get("evidence_status") not in ("mapped", "unmapped") or
               bool(row.get("gold_doc_ids")) != (row["evidence_status"] == "mapped") for row in rows):
            raise ValueError(f"invalid evidence status in {name}")
    if nq_report is None:
        candidates = [evidence / "nq-aligned/alignment_report.json", evidence / "reports/nq_alignment_report.json"]
        nq_report = next((path for path in candidates if path.is_file()), None)
    if nq_report is None:
        raise FileNotFoundError("NQ alignment report missing; provide --nq-report or run grasp-gold-pipeline")
    alignment = json.loads(nq_report.read_text(encoding="utf-8"))
    if alignment.get("retrieval_corpus_sha256") != manifest["corpora"]["wiki18"]["passages_sqlite_sha256"]:
        raise ValueError("NQ alignment report belongs to a different Wiki18 corpus")
    for split in ("train", "dev"):
        report = json.loads((evidence / f"reports/{split}_mapping_report.json").read_text(encoding="utf-8"))
        nq_source = report.get("sources", {}).get("wiki18")
        if nq_source and nq_source.get("sha256") != alignment.get("splits", {}).get(split, {}).get("output_sha256"):
            raise ValueError(f"NQ {split} alignment report does not match processed source")
    destination = evidence / "reports/nq_alignment_report.json"
    if nq_report.resolve() != destination.resolve():
        shutil.copy2(nq_report, destination)
    manifest["reports"] = {"reports/" + name: {"sha256": sha256(evidence / "reports" / name)} for name in REPORT_FILES}
    write_json(manifest_path, manifest)
    counts = manifest["files"]["train.jsonl"]
    card = f'''---
language: en
task_categories:
  - question-answering
license: other
---
# GRASP gold evidence (local release candidate)

Retrieval revision: `{manifest["retrieval_revision"]}`.
Full train: {counts["rows"]}; mapped: {counts["mapped_rows"]}; unmapped: {counts["unmapped_rows"]}.
Dev: {manifest["files"]["dev.jsonl"]["rows"]}. Counts describe this artifact, not full benchmark coverage.

train.jsonl retains the input QA in order with input_id. evidence_status is mapped or unmapped.
Mapped rows have verified gold_doc_ids. Unmapped rows have [] and evidence_reason; recall is
unavailable, not zero. train_mapped.jsonl is the exact mapped-only subset for recall-aware GRPO.
Do not infer gold labels from retrieved rankings, answer-string search or generated triples.
Gold IDs are used only after rollout for reward/evaluation, never in model prompts.

HotpotQA/2Wiki use original supporting-fact titles with punctuation-preserving title matching.
NQ uses original DPR gold contexts and Wikipedia URL/title provenance. Exact normalized-token
containment (minimum 20 tokens) verifies passage content. Proofs preserve token offsets and hashes.
Original HTML offsets and the corpus Wikipedia oldid are unavailable and are never fabricated.
Only public labeled dev annotations are supplied; no hidden-test labels are fabricated.

Source URLs, revisions, SHA256 and attribution are in manifest.json/source_inventory.json.
HotpotQA: https://github.com/hotpotqa/hotpot (CC BY-SA 4.0).
2Wiki: https://github.com/Alab-NII/2wikimultihop/blob/main/LICENSE (source terms).
NQ: https://github.com/google-research-datasets/natural-questions (annotation terms).
DPR: https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py (source/licensing).
Wikipedia-derived material retains the original attribution/share-alike requirements.
This package supplies QA labels and reports; it does not duplicate the retrieval corpus.

Release candidate: canonical data validation passed; real SFT rollout smoke is pending.
Run smoke (30 questions/corpus, nonzero context recall and reward variance) before primary GRPO.
Package before smoke; changing the manifest afterwards invalidates its hash binding.
Local preflight and data validation are separate checks from the real model rollout gate.
'''
    (evidence / "README.md").write_text(card, encoding="utf-8")
    return [evidence / name for name in DATA_FILES] + [evidence / "reports" / name for name in REPORT_FILES]


def trainer_files(trainer: Path) -> list[Path]:
    files = [trainer / name for name in ("pyproject.toml", "requirements.txt", "README.md", "RUN_VAST_EVIDENCE_V2.md")]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"trainer artifact missing: {path}")
    files.extend(path for name in (".gitignore", ".gitattributes") if (path := trainer / name).is_file())
    for folder, suffixes in {"src": {".py"}, "tests": {".py"}, "scripts": {".py", ".sh", ".ps1"},
                             "docs": {".md", ".txt"}, "config": {".json", ".example"}}.items():
        files.extend(path for path in (trainer / folder).rglob("*") if path.is_file()
                     and not path.is_symlink() and path.suffix in suffixes and "__pycache__" not in path.parts
                     and (folder != "config" or "example" in path.name))
    return sorted(files)


def write_archive(output: Path, evidence: Path, files: list[Path], trainer: Path | None = None) -> None:
    code = trainer_files(trainer) if trainer else []
    if output.resolve() in {path.resolve() for path in files + code}:
        raise ValueError("archive output must not overwrite an input artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for path in files:
            package.write(path, "data/grasp-gold-evidence/" + path.relative_to(evidence).as_posix())
        for path in code:
            data = path.read_bytes()
            if path.suffix in (".py", ".sh") or path.name.endswith(".env.example"):
                data = data.replace(b"\r\n", b"\n")
            package.writestr("grasp-grpo-trainer/" + path.relative_to(trainer).as_posix(), data)
    temporary.replace(output)


def export_evidence(evidence: Path, files: list[Path], destination: Path) -> None:
    """Copy only canonical artifacts, byte for byte, into a Git-tracked handoff directory."""
    if destination.resolve() == evidence.resolve():
        raise ValueError("Git output directory must differ from the working evidence directory")
    for source in files:
        target = destination / source.relative_to(evidence)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        if sha256(source) != sha256(target):
            raise ValueError(f"export SHA256 mismatch: {target}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", default="work/exact-grpo")
    parser.add_argument("--nq-report", help="Defaults to this evidence directory's nq-aligned or staged reports")
    parser.add_argument("--trainer-dir", help="Optional trainer clone to include in a combined archive")
    parser.add_argument("--output", help="Defaults to a data-only ZIP; combined ZIP when --trainer-dir is supplied")
    parser.add_argument("--git-output-dir", help="Also export canonical QA/reports for Git, e.g. outputs/grasp-gold-evidence")
    args = parser.parse_args(argv)
    trainer = Path(args.trainer_dir).resolve() if args.trainer_dir else None
    if trainer:
        trainer_files(trainer)  # Fail before modifying evidence if the optional clone is missing.
    output = Path(args.output or ("work/handoff/grasp-evidence-grpo-v2.zip" if trainer else "work/handoff/grasp-gold-evidence-v1.zip"))
    if output.suffix.lower() != ".zip":
        parser.error("--output must be a .zip path")
    evidence = Path(args.evidence_dir).resolve()
    files = prepare_evidence(evidence, Path(args.nq_report) if args.nq_report else None)
    write_archive(output, evidence, files, trainer)
    if args.git_output_dir:
        export_evidence(evidence, files, Path(args.git_output_dir))
    print(json.dumps({"archive": str(output.resolve()), "bytes": output.stat().st_size,
                      "git_output_dir": str(Path(args.git_output_dir).resolve()) if args.git_output_dir else None,
                      "includes_trainer": bool(trainer), "status": "prepared; real SFT rollout smoke still required"}))


if __name__ == "__main__":
    main()
