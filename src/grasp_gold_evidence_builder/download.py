"""Acquire labeled HotpotQA/2Wiki/NQ annotations with hashes and source revisions."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

USER_AGENT = "grasp-gold-evidence-builder/0.2"
HOTPOT_REPO = "hotpotqa/hotpot_qa"
HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
TWOWIKI_URL = "https://www.dropbox.com/s/npidmtadreo6df2/data.zip?dl=1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def download(url: str, destination: Path, expected_sha256: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and expected_sha256 and sha256(destination) == expected_sha256:
        print(f"Cached: {destination.name}", flush=True)
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"Downloading: {destination.name}", flush=True)
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if expected_sha256 and sha256(temporary) != expected_sha256:
        raise ValueError(f"SHA256 mismatch: {destination}")
    temporary.replace(destination)


def register(root: Path, destination: Path, metadata: dict) -> None:
    path = root / "source_metadata.json"
    value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    value[destination.relative_to(root).as_posix()] = {
        **metadata, "sha256": sha256(destination), "downloaded_at_utc": datetime.now(UTC).isoformat(),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def supporting_facts(value):
    if isinstance(value, dict):
        return list(zip(value.get("title", []), value.get("sent_id", []), strict=True))
    return value


def context(value):
    if isinstance(value, dict):
        return list(zip(value.get("title", []), value.get("sentences", []), strict=True))
    return value


def fetch_hotpot(root: Path, cache: Path, splits: list[str], source: str, revision: str = HOTPOT_REVISION) -> None:
    if source == "official":
        for split in splits:
            filename = "hotpot_train_v1.1.json" if split == "train" else "hotpot_dev_distractor_v1.json"
            url = f"http://curtis.ml.cmu.edu/datasets/hotpot/{filename}"
            destination = root / "hotpotqa" / f"{split}.json"
            download(url, destination)
            register(root, destination, {"url": url, "revision": filename, "license": "CC BY-SA 4.0", "source_split": split})
        return
    try:
        from pyarrow import parquet
    except ImportError as exc:
        raise RuntimeError('Install download dependencies: python -m pip install -e ".[download]"') from exc
    tree = request_json(f"https://huggingface.co/api/datasets/{HOTPOT_REPO}/tree/{revision}/distractor")
    for split in splits:
        upstream_split = "train" if split == "train" else "validation"
        files = sorted((item for item in tree if item["path"].startswith(f"distractor/{upstream_split}-") and item["path"].endswith(".parquet")), key=lambda item: item["path"])
        if not files:
            raise ValueError(f"No {upstream_split} parquet files at {revision}")
        destination = root / "hotpotqa" / f"{split}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.partial")
        rows, artifacts = 0, []
        with temporary.open("w", encoding="utf-8") as output:
            output.write("[\n")
            for item in files:
                url = f"https://huggingface.co/datasets/{HOTPOT_REPO}/resolve/{revision}/{item['path']}"
                local = cache / "hotpotqa" / Path(item["path"]).name
                download(url, local, item.get("lfs", {}).get("oid"))
                artifacts.append({"url": url, "sha256": sha256(local)})
                for batch in parquet.ParquetFile(local).iter_batches(batch_size=64):
                    for row in batch.to_pylist():
                        required = {"id", "question", "answer", "supporting_facts", "context"}
                        if not required <= set(row):
                            raise ValueError(f"Unexpected HotpotQA schema: {row.keys()}")
                        record = {"_id": row["id"], "question": row["question"], "answer": row["answer"], "supporting_facts": supporting_facts(row["supporting_facts"]), "context": context(row["context"])}
                        output.write((",\n" if rows else "") + json.dumps(record, ensure_ascii=False))
                        rows += 1
            output.write("\n]\n")
        temporary.replace(destination)
        register(root, destination, {"url": f"https://huggingface.co/datasets/{HOTPOT_REPO}", "revision": revision, "license": "CC BY-SA 4.0", "source_split": upstream_split, "rows": rows, "downloaded_artifacts": artifacts, "transformation": "Parquet to official JSON schema; preserve original supporting facts"})
        print(f"Registered {split}: {rows} questions", flush=True)


def fetch_twowiki(root: Path, cache: Path, splits: list[str]) -> None:
    archive = cache / "2wiki-data.zip"
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        download(TWOWIKI_URL, archive)
    archive_hash = sha256(archive)
    with zipfile.ZipFile(archive) as package:
        for split in splits:
            matches = [name for name in package.namelist() if Path(name.replace("\\", "/")).name == f"{split}.json"]
            if len(matches) != 1:
                raise ValueError(f"Expected one {split}.json in 2Wiki archive; found {matches}")
            destination = root / "2wiki" / f"{split}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".json.partial")
            with package.open(matches[0]) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(destination)
            register(root, destination, {"url": TWOWIKI_URL, "revision": f"archive-sha256:{archive_hash}", "license": "See official 2WikiMultiHopQA dataset terms", "source_split": split, "archive_member": matches[0], "archive_sha256": archive_hash})
            print(f"Registered 2Wiki {split}", flush=True)


def fetch_nq(root: Path, cache: Path, splits: list[str]) -> None:
    """Original NQ gold contexts/provenance repackaged by the official DPR authors."""
    for split in splits:
        url = f"https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-{split}_gold_info.json.gz"
        compressed = cache / f"nq-{split}_gold_info.json.gz"
        if not compressed.is_file():
            download(url, compressed)
        destination = root / "nq" / f"{split}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.partial")
        with gzip.open(compressed, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(destination)
        register(root, destination, {
            "url": url, "revision": f"gzip-sha256:{sha256(compressed)}",
            "license": "NQ annotation license: Apache-2.0; Wikipedia evidence text: CC BY-SA",
            "source_split": f"DPR nq-{split} gold info (not Google hidden test)",
            "compressed_sha256": sha256(compressed),
            "schema": "data[]: example_id/question/short_answers/context/title/doc_url",
        })
        print(f"Registered NQ {split}; passage alignment is still required", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download labeled HotpotQA, 2Wiki or NQ annotations")
    parser.add_argument("--dataset", required=True, choices=("hotpotqa", "2wiki", "nq"))
    parser.add_argument("--split", required=True, choices=("train", "dev", "all"))
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--cache-dir", default=".downloads")
    parser.add_argument("--hotpot-source", choices=("huggingface", "official"), default="huggingface", help="HF preserves labeled annotations; the original CMU download host may fail DNS")
    parser.add_argument("--hotpot-revision", default=HOTPOT_REVISION, help="Pinned HF commit for reproducible HotpotQA downloads")
    args = parser.parse_args(argv)
    root, cache = Path(args.raw_root), Path(args.cache_dir)
    splits = ["train", "dev"] if args.split == "all" else [args.split]
    if args.dataset == "hotpotqa":
        fetch_hotpot(root, cache, splits, args.hotpot_source, args.hotpot_revision)
    elif args.dataset == "2wiki":
        fetch_twowiki(root, cache, splits)
    else:
        fetch_nq(root, cache, splits)


if __name__ == "__main__":
    main()
