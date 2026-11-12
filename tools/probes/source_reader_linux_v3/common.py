"""Standard-library-only identities and durable receipts for the synthetic probe."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
MEMORY_BYTES = 6_000_000_000
CPU_COUNT = 4
LOAD_SECONDS = 180
PROMPT_SECONDS = 180
WORKER_SECONDS = 600
MODEL_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def event(path: Path, kind: str, **fields: object) -> dict:
    row = {"event": kind, "monotonic_ns": time.monotonic_ns(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return row


def claim(path: Path) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write("First attempt only; do not reuse this directory.\n")
        handle.flush()
        os.fsync(handle.fileno())


def identities() -> dict:
    return {p.name: sha256(p) for p in sorted(ROOT.iterdir())
            if p.is_file() and (p.suffix in {".py", ".json", ".txt", ".yml"}
                                or p.name == "Dockerfile")}


def model_manifest() -> dict:
    data = json.loads((ROOT / "model-manifest.json").read_text())
    if (data["repo"] != "mlx-community/Qwen3-4B-Instruct-2507-4bit"
            or data["revision"] != MODEL_REVISION or len(data["files"]) != 11):
        raise ValueError("Unexpected public model identity")
    names = [row["name"] for row in data["files"]]
    if len(set(names)) != 11 or any(Path(n).name != n for n in names):
        raise ValueError("Unsafe or repeated model filename")
    return data


def verify_model(directory: Path) -> dict:
    data = model_manifest()
    if sorted(p.name for p in directory.iterdir()) != sorted(r["name"] for r in data["files"]):
        raise ValueError("Local model directory differs from the exact 11-file allowlist")
    for row in data["files"]:
        path = directory / row["name"]
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]
                or sha256(path) != row["sha256"]):
            raise ValueError("Model SHA/size mismatch: " + row["name"])
    return {"revision": data["revision"], "manifest_sha256": sha256(ROOT / "model-manifest.json"),
            "files": len(data["files"]), "bytes": sum(r["bytes"] for r in data["files"])}


def cgroup_snapshot(root: Path = Path("/sys/fs/cgroup")) -> dict:
    names = ("memory.max", "memory.current", "memory.peak", "memory.swap.max",
             "memory.swap.current", "memory.events", "cpu.max", "cpuset.cpus.effective")
    result = {}
    for name in names:
        try:
            result[name] = (root / name).read_text().strip()
        except OSError as exc:
            result[name] = {"unavailable": type(exc).__name__}
    return result


def memory_limit() -> dict:
    page_size = os.sysconf("SC_PAGE_SIZE")
    if type(page_size) is not int or page_size <= 0:
        raise RuntimeError("RESOURCE_ENFORCEMENT_UNAVAILABLE: invalid OS page size")
    resolved = (MEMORY_BYTES // page_size) * page_size
    if resolved <= 0 or resolved > MEMORY_BYTES:
        raise RuntimeError("RESOURCE_ENFORCEMENT_UNAVAILABLE: invalid page-aligned limit")
    return {"requested_bytes": MEMORY_BYTES, "resolved_bytes": resolved, "page_size": page_size}


def verify_cgroup(snapshot: dict) -> None:
    resolved = memory_limit()["resolved_bytes"]
    if snapshot.get("memory.max") != str(resolved) or snapshot.get("memory.swap.max") != "0":
        raise RuntimeError("RESOURCE_ENFORCEMENT_UNAVAILABLE: memory/swap cgroup mismatch")
    cpu = snapshot.get("cpu.max")
    if not isinstance(cpu, str) or len(cpu.split()) != 2:
        raise RuntimeError("RESOURCE_ENFORCEMENT_UNAVAILABLE: cpu.max missing")
    quota, period = cpu.split()
    if quota == "max" or int(quota) != CPU_COUNT * int(period):
        raise RuntimeError("RESOURCE_ENFORCEMENT_UNAVAILABLE: CPU quota mismatch")
    for key in ("memory.current", "memory.peak", "memory.swap.current", "memory.events"):
        if not isinstance(snapshot.get(key), str):
            raise RuntimeError("RESOURCE_ENFORCEMENT_UNAVAILABLE: " + key)


def memory_event_failures(before: dict, after: dict) -> dict:
    def counts(snapshot):
        return {key: int(value) for key, value in
                (line.split() for line in snapshot["memory.events"].splitlines())}
    initial, final = counts(before), counts(after)
    return {key: final.get(key, 0) - initial.get(key, 0)
            for key in ("max", "oom", "oom_kill", "oom_group_kill")
            if final.get(key, 0) != initial.get(key, 0)}


def process_rss(pid: int) -> dict:
    result = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"VmRSS", "VmHWM"}:
            result[key + "_bytes"] = int(value.split()[0]) * 1024
    return result
