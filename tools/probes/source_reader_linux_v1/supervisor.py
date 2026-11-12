"""Host-side Docker supervisor. No MLX/tokenizer imports, no model retries."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback

from common import (CPU_COUNT, LOAD_SECONDS, MEMORY_BYTES, PROMPT_SECONDS,
                    WORKER_SECONDS, atomic_json, cgroup_snapshot, claim, event,
                    identities, process_rss)


def docker(*args: str, timeout: int = 15) -> str:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("Docker command failed: " + " ".join(args[:2]) + "\n" + result.stderr[:8000])
    return result.stdout.strip()


def events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Only a currently incomplete final append can be ignored.
            if index != len(lines) - 1:
                raise
    return rows


def deadline_reason(now: float, started: float, rows: list[dict], *,
                    load_seconds: float = LOAD_SECONDS, prompt_seconds: float = PROMPT_SECONDS,
                    worker_seconds: float = WORKER_SECONDS) -> str | None:
    if now - started >= worker_seconds:
        return "TIMEOUT_WORKER"
    starts = [r for r in rows if r.get("event") == "phase_start"]
    if not starts:
        return "TIMEOUT_LOAD" if now - started >= load_seconds else None
    for phase in starts:
        ended = next((r for r in rows if r.get("event") == "phase_end"
                      and r.get("stage") == phase.get("stage")
                      and r.get("prompt") == phase.get("prompt")
                      and r["monotonic_ns"] >= phase["monotonic_ns"]), None)
        limit = prompt_seconds if phase["stage"] == "prompt" else load_seconds
        phase_end = ended["monotonic_ns"] / 1e9 if ended else now
        if phase_end - phase["monotonic_ns"] / 1e9 >= limit:
            return "TIMEOUT_PROMPT" if phase["stage"] == "prompt" else "TIMEOUT_LOAD"
    return None


def host_cgroup(pid: int) -> Path:
    for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            return Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    raise RuntimeError("RESOURCE_ENFORCEMENT_UNAVAILABLE: cgroup v2 not found")


def stop_container(container: str) -> None:
    try:
        docker("stop", "--time", "5", container, timeout=8)
    except Exception:
        docker("kill", container, timeout=5)


def final_status(state: dict, runtime: dict, abort_reason, attach_exit: int) -> str:
    if state.get("OOMKilled"):
        return "FAIL_RESOURCE_OOM"
    if abort_reason:
        return abort_reason
    if state.get("Running"):
        return "UNAVAILABLE"
    status = runtime.get("status", "UNAVAILABLE")
    if status == "PASS_LINUX_CPU_TINY_SMOKE":
        return status if state.get("ExitCode") == 0 and attach_exit == 0 else "UNAVAILABLE"
    return status if status in {"UNAVAILABLE", "SHORT_OUTPUT", "FAIL_RESOURCE", "TIMEOUT_PROMPT"} else "UNAVAILABLE"


def monitor_process(process, output: Path, started: float, stop, sample, *,
                    load_seconds: float = LOAD_SECONDS, prompt_seconds: float = PROMPT_SECONDS,
                    worker_seconds: float = WORKER_SECONDS, interrupted=lambda: False) -> dict:
    """Also exercised with a real fake child; stop/sample are host-only callbacks."""
    maximum_rss = 0
    latest_sample = {}
    reason = None
    while process.poll() is None:
        now = time.monotonic()
        latest_sample = sample()
        maximum_rss = max(maximum_rss, latest_sample.get("rss", {}).get("VmRSS_bytes", 0))
        event(output / "resource-samples.jsonl", "sample", **latest_sample)
        reason = "INTERRUPTED" if interrupted() else deadline_reason(
            now, started, events(output / "runtime-events.jsonl"), load_seconds=load_seconds,
            prompt_seconds=prompt_seconds, worker_seconds=worker_seconds)
        if reason:
            event(output / "supervisor-events.jsonl", "abort", reason=reason)
            stop()
            break
        time.sleep(0.05)  # Target <=100 ms; actual timestamp gaps remain in the evidence.
    reason = reason or deadline_reason(time.monotonic(), started, events(output / "runtime-events.jsonl"),
                                      load_seconds=load_seconds, prompt_seconds=prompt_seconds,
                                      worker_seconds=worker_seconds)
    return {"abort_reason": reason, "sampled_maximum_rss_bytes": maximum_rss,
            "last_sample": latest_sample, "attached_process_exit": process.wait(timeout=8)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    claim(output / "supervisor-first-attempt.claim")
    receipt = {"status": "STARTING", "synthetic_only": True, "vps_tested": False,
               "probe_hashes": identities(), "started_monotonic_ns": time.monotonic_ns()}
    atomic_json(output / "supervisor-result.json", receipt)
    container = None
    process = None
    interrupted = False

    def receive_signal(_number, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, receive_signal)
    signal.signal(signal.SIGINT, receive_signal)
    try:
        if not Path("/sys/fs/cgroup/cgroup.controllers").exists():
            raise RuntimeError("RESOURCE_ENFORCEMENT_UNAVAILABLE: host cgroup v2 required")
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) < CPU_COUNT:
            raise RuntimeError("RESOURCE_ENFORCEMENT_UNAVAILABLE: fewer than four host CPUs")
        image = json.loads(docker("image", "inspect", args.image))[0]
        if image.get("Architecture") != "amd64" or image.get("Os") != "linux":
            raise RuntimeError("Required linux/amd64 image unavailable")
        receipt.update(image_id=image["Id"], image_repo_digests=image.get("RepoDigests", []),
                       runner_image=os.environ.get("ImageVersion"), git_sha=os.environ.get("GITHUB_SHA"))
        container = docker("create", "--network", "none", "--cpus", "4", "--cpuset-cpus",
                           ",".join(map(str, allowed[:4])), "--memory", str(MEMORY_BYTES),
                           "--memory-swap", str(MEMORY_BYTES), "--read-only", "--cap-drop", "ALL",
                           "--security-opt", "no-new-privileges", "--log-driver", "local",
                           "--log-opt", "max-size=2m", "--log-opt", "max-file=1",
                           "--tmpfs", "/tmp:rw,nosuid,noexec,size=67108864",
                           "--mount", f"type=bind,src={args.model.resolve()},dst=/model,readonly",
                           "--mount", f"type=bind,src={output},dst=/evidence", args.image)
        inspect = json.loads(docker("inspect", container))[0]
        config = inspect["HostConfig"]
        if (config["Memory"] != MEMORY_BYTES or config["MemorySwap"] != MEMORY_BYTES
                or config["NanoCpus"] != CPU_COUNT * 1_000_000_000 or config["NetworkMode"] != "none"):
            raise RuntimeError("RESOURCE_ENFORCEMENT_UNAVAILABLE: Docker configuration mismatch")
        receipt.update(container_id=container, host_config={key: config[key] for key in
                       ("Memory", "MemorySwap", "NanoCpus", "CpusetCpus", "NetworkMode")})
        atomic_json(output / "supervisor-result.json", receipt)
        started = time.monotonic()
        with (output / "container.stdout.log").open("xb") as log:
            process = subprocess.Popen(["docker", "start", "--attach", container], stdout=log,
                                       stderr=subprocess.STDOUT,
                                       preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_FSIZE,
                                                                            (2_097_152, 2_097_152)))
            pid = 0
            while not pid and process.poll() is None:
                pid = int(json.loads(docker("inspect", container))[0]["State"].get("Pid", 0))
                if time.monotonic() - started > 15:
                    raise RuntimeError("Container process failed to start within 15 seconds")
                if not pid:
                    time.sleep(0.05)
            cgroup = host_cgroup(pid) if pid else None

            def sample():
                row = {"cgroup": cgroup_snapshot(cgroup)} if cgroup else {}
                try:
                    row["rss"] = process_rss(pid)
                except OSError:
                    row["rss"] = {"unavailable": "process exited"}
                return row

            receipt.update(monitor_process(process, output, started, lambda: stop_container(container),
                                           sample, interrupted=lambda: interrupted))
            log.flush()
            os.fsync(log.fileno())
        state = json.loads(docker("inspect", container))[0]["State"]
        if state.get("Running"):
            receipt["abort_reason"] = receipt.get("abort_reason") or "ATTACH_PROCESS_FAILED"
            stop_container(container)
            state = json.loads(docker("inspect", container))[0]["State"]
        receipt["container_state"] = state
        runtime_path = output / "runtime-result.json"
        runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        receipt["runtime_status"] = runtime.get("status", "NO_RUNTIME_RECEIPT")
        receipt["status"] = final_status(state, runtime, receipt["abort_reason"], receipt["attached_process_exit"])
        return 0 if receipt["status"] == "PASS_LINUX_CPU_TINY_SMOKE" else 1
    except BaseException:
        receipt.update(status="UNAVAILABLE", error=traceback.format_exc())
        if container:
            try:
                stop_container(container)
                receipt["container_state"] = json.loads(docker("inspect", container))[0]["State"]
                if receipt["container_state"].get("OOMKilled"):
                    receipt["status"] = "FAIL_RESOURCE_OOM"
            except Exception:
                receipt["cleanup_error"] = traceback.format_exc()
        return 1
    finally:
        receipt["ended_monotonic_ns"] = time.monotonic_ns()
        atomic_json(output / "supervisor-result.json", receipt)
        if container:
            try:
                docker("rm", "--force", container)
            except Exception:
                event(output / "supervisor-events.jsonl", "cleanup_failed", error=traceback.format_exc())


if __name__ == "__main__":
    raise SystemExit(main())
