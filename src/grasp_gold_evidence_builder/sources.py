"""Raw-file loading, SHA256 inventory, and source metadata handling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def iter_records(path: str | Path):
    """Stream JSONL so teacher traces are never all retained in memory."""
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        with source.open(encoding="utf-8-sig") as handle:
            for number, line in enumerate(handle, start=1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{source}:{number}: expected an object")
                    yield row
    else:
        yield from read_records(source)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    value: Any
    if source.suffix.lower() == ".jsonl":
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
    if isinstance(value, dict):
        for key in ("data", "examples", "records"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{source}: expected a JSON/JSONL list of objects")
    return value


def load_metadata(raw_root: str | Path) -> dict[str, Any]:
    path = Path(raw_root) / "source_metadata.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def inventory(raw_root: str | Path) -> dict[str, Any]:
    root = Path(raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"raw annotations directory not found: {root}")
    metadata = load_metadata(root)
    files = []
    for path in sorted([*root.rglob("*.json"), *root.rglob("*.jsonl")]):
        if path.name == "source_metadata.json":
            continue
        relative = path.relative_to(root).as_posix()
        try:
            rows = read_records(path)
            keys = sorted({key for row in rows[:100] for key in row})
            row_count: int | None = len(rows)
            error = None
            del rows
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            keys, row_count, error = [], None, str(exc)
        files.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size, "rows": row_count, "observed_top_level_keys": keys, "source": metadata.get(relative, metadata.get(path.parent.name, {})), "read_error": error})
    return {"raw_root": str(root.resolve()), "files": files}
