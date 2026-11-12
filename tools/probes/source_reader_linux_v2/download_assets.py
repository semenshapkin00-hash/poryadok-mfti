"""Anonymous, single-attempt downloads of the exact public model; no NN imports."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import signal
import time
import traceback
import urllib.request

from common import atomic_json, claim, model_manifest, verify_model


def download_one(url: str, target: Path, expected: dict) -> None:
    temporary = target.with_suffix(target.suffix + ".partial")
    digest = hashlib.sha256()
    total = 0
    # A fresh opener does not use ~/.netrc, HF caches, bearer tokens or proxy auth.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"User-Agent": "public-synthetic-linux-probe/1"})
    with opener.open(request, timeout=60) as response, temporary.open("xb") as handle:
        while block := response.read(1024 * 1024):
            total += len(block)
            if total > expected["bytes"]:
                raise ValueError("Download exceeds manifest size: " + target.name)
            digest.update(block)
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
    if total != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError("Download SHA/size mismatch: " + target.name)
    os.replace(temporary, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    claim(args.output / "download-first-attempt.claim")
    receipt = {"status": "PREPARING", "public_only": True, "started_monotonic_ns": time.monotonic_ns(),
               "files_completed": []}
    path = args.output / "preparation.json"
    atomic_json(path, receipt)
    def interrupted(number, _frame):
        raise InterruptedError(f"Preparation interrupted by signal {number}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        args.destination.mkdir(parents=True, exist_ok=False)
        data = model_manifest()
        for row in data["files"]:
            url = f"https://huggingface.co/{data['repo']}/resolve/{data['revision']}/{row['name']}"
            download_one(url, args.destination / row["name"], row)
            receipt["files_completed"].append(row["name"])
            atomic_json(path, receipt)
        receipt.update(status="ASSETS_VERIFIED", model=verify_model(args.destination))
        return 0
    except BaseException:
        receipt.update(status="ASSETS_UNAVAILABLE", error=traceback.format_exc())
        return 1
    finally:
        receipt["ended_monotonic_ns"] = time.monotonic_ns()
        atomic_json(path, receipt)


if __name__ == "__main__":
    raise SystemExit(main())
