"""Two synthetic CPU-only 512 -> 32 calls. MLX is imported only inside run()."""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

from common import (ROOT, PROMPT_SECONDS, atomic_json, cgroup_snapshot, claim, event, identities,
                    memory_event_failures, memory_limit,
                    process_rss, verify_cgroup, verify_model)


def assemble_prompt(tokenizer, system: str, seed: str, filler_line: str) -> dict:
    """Only the last filler line may be shortened; never truncate seed/template."""
    def render(filler: str) -> tuple[str, list[int]]:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": seed + "\n\n[Синтетический наполнитель]\n" + filler}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return text, list(tokenizer.encode(text, add_special_tokens=False))

    _, base_ids = render("")
    if len(base_ids) >= 512:
        raise ValueError("Synthetic seed/template leaves no filler budget")
    filler = filler_line * 256
    # Fixed exhaustive character-prefix search, no inference or output-dependent edits.
    for length in range(len(filler) + 1):
        text, ids = render(filler[:length])
        if len(ids) == 512:
            return {"rendered_text": text, "input_ids": ids, "input_tokens": 512,
                    "filler_characters": length,
                    "rendered_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "input_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
    raise ValueError("Exact 512-token assembly unavailable; no model call allowed")


def package_versions() -> dict:
    versions = {}
    for line in (ROOT / "requirements-lock.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        pin = line.split()[0]
        name, expected = pin.split("==")
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Pinned package mismatch: {name}: {actual} != {expected}")
        versions[name] = actual
    for forbidden in ("mlx-metal", "mlx-cuda-12", "mlx-cuda-13", "torch"):
        try:
            importlib.metadata.version(forbidden)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise RuntimeError("Forbidden backend/extra installed: " + forbidden)
    return versions


def check_shared_libraries() -> dict:
    roots = [Path(sys.prefix) / "lib/python3.12/site-packages/mlx"]
    libraries = sorted({p for root in roots for p in root.rglob("*.so")})
    if not libraries:
        raise RuntimeError("No installed MLX shared libraries found")
    results = {}
    for library in libraries:
        completed = subprocess.run(["ldd", str(library)], capture_output=True, text=True, timeout=10)
        results[str(library.relative_to(Path(sys.prefix)))] = completed.stdout + completed.stderr
        if completed.returncode or "not found" in results[str(library.relative_to(Path(sys.prefix)))]:
            raise RuntimeError("MLX shared-library linkage unavailable: " + str(library))
    return results


def generate_one(mx, stream_generate, model, tokenizer, prompt: dict, index: int,
                 sampler, output: Path) -> dict:
    stream = mx.new_stream(mx.cpu)
    start = event(output / "runtime-events.jsonl", "phase_start", stage="prompt", prompt=index)
    tokens, timestamps, segments = [], [], []
    finish_reason = None
    library_prompt_seconds = None
    generator = stream_generate(model, tokenizer, prompt=prompt["input_ids"], max_tokens=32,
                                sampler=sampler, max_kv_size=544, prefill_step_size=512,
                                prompt_cache=None, kv_bits=None, stream=stream)
    with contextlib.closing(generator):
        for response in generator:
            if int(response.prompt_tokens) != 512:
                raise RuntimeError("Actual model prompt token count differs from 512")
            if library_prompt_seconds is None:
                library_prompt_seconds = 512 / float(response.prompt_tps)
            mx.synchronize(stream)
            if bool(mx.any(mx.isnan(response.logprobs)).item()):
                raise RuntimeError("NaN in actual output log probabilities")
            mx.synchronize(mx.cpu)
            position = int(response.generation_tokens)
            token = int(response.token)
            if position == len(tokens) + 1:
                stamp = time.monotonic_ns()
                tokens.append(token)
                timestamps.append(stamp)
                event(output / "runtime-events.jsonl", "token", prompt=index,
                      position=position, token=token, evaluated_monotonic_ns=stamp)
            elif position != len(tokens) or not tokens or token != tokens[-1]:
                raise RuntimeError("Unexpected stream token-position sequence")
            segments.append(response.text)
            finish_reason = response.finish_reason
    mx.synchronize(stream)
    end = time.monotonic_ns()
    if end - start["monotonic_ns"] >= PROMPT_SECONDS * 1e9:
        raise RuntimeError("TIMEOUT_PROMPT: completed outside the fixed wall bound")
    count = len(tokens)
    completed = count == 32 and finish_reason == "length" and not any(t in tokenizer.eos_token_ids for t in tokens)
    result = {"prompt": index, "status": "COMPLETED_32" if completed else "SHORT_OUTPUT",
              "input_tokens": prompt["input_tokens"], "generated_tokens": count,
              "output_token_ids": tokens, "token_monotonic_ns": timestamps,
              "output_text": "".join(segments), "finish_reason": finish_reason,
              "request_start_monotonic_ns": start["monotonic_ns"], "ended_monotonic_ns": end,
              "wall_seconds": (end - start["monotonic_ns"]) / 1e9,
              "library_prefill_plus_first_token_seconds": library_prompt_seconds,
              "ttft_seconds": (timestamps[0] - start["monotonic_ns"]) / 1e9 if timestamps else None,
              "rate32_including_prefill_tps": 32e9 / (timestamps[-1] - start["monotonic_ns"]) if completed else None,
              "decode31_tps": 31e9 / (timestamps[-1] - timestamps[0]) if completed else None,
              "rss": process_rss(os.getpid()), "ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "cgroup": cgroup_snapshot()}
    atomic_json(output / f"prompt-{index}-result.json", result)
    event(output / "runtime-events.jsonl", "phase_end", stage="prompt", prompt=index, status=result["status"])
    del generator
    gc.collect()
    return result


def run(model_dir: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    claim(output / "runtime-first-attempt.claim")
    receipt = {"status": "STARTING", "synthetic_only": True, "quality_claimed": False,
               "started_monotonic_ns": time.monotonic_ns(), "probe_hashes": identities(), "prompts": []}
    atomic_json(output / "runtime-result.json", receipt)
    stage = "preflight"
    event(output / "runtime-events.jsonl", "phase_start", stage="load")
    try:
        if sys.platform != "linux" or platform.machine() != "x86_64" or sys.version_info[:3] != (3, 12, 14):
            raise RuntimeError("Required Linux x86_64 CPython 3.12.14 is unavailable")
        snapshot = cgroup_snapshot()
        receipt.update(cgroup_initial=snapshot, memory_limit_requested_bytes=6_000_000_000)
        atomic_json(output / "runtime-result.json", receipt)
        receipt["memory_limit"] = memory_limit()
        atomic_json(output / "runtime-result.json", receipt)
        verify_cgroup(snapshot)
        if sorted(p.name for p in Path("/sys/class/net").iterdir()) != ["lo"]:
            raise RuntimeError("Required network=none namespace is unavailable")
        for name, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}.items():
            if os.environ.get(name) != value:
                raise RuntimeError("Offline environment mismatch: " + name)
        receipt.update(cgroup_initial=snapshot, versions=package_versions(),
                       model=verify_model(model_dir), linkage=check_shared_libraries(),
                       python=sys.version, machine=platform.machine(), kernel=platform.release(),
                       glibc=platform.libc_ver(), cpuinfo=Path("/proc/cpuinfo").read_text()[:32768])
        atomic_json(output / "runtime-result.json", receipt)
        stage = "cpu_import"
        import mlx.core as mx
        if not mx.is_available(mx.cpu):
            raise RuntimeError("Required CPU backend is unavailable")
        mx.set_default_device(mx.cpu)
        if mx.default_device() != mx.cpu:
            raise RuntimeError("CPU default device was not selected")
        event(output / "runtime-events.jsonl", "cpu_selected", device=str(mx.default_device()))
        stage = "prompt_assembly"
        from transformers import AutoTokenizer
        preparatory_tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False)
        spec = json.loads((ROOT / "probe-prompts.json").read_text())
        if len(spec["prompts"]) != 2:
            raise ValueError("Exactly two frozen synthetic prompts are required")
        prompts = [assemble_prompt(preparatory_tokenizer, spec["system"], seed, spec["filler_line"])
                   for seed in spec["prompts"]]
        atomic_json(output / "prepared-prompts.json", prompts)
        del preparatory_tokenizer
        stage = "model_load"
        # CPU MUST be selected before this import: generate.py creates its stream at import time.
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        mx.random.seed(0)
        model, tokenizer = load(str(model_dir), lazy=False, trust_remote_code=False,
                                tokenizer_config={"local_files_only": True, "trust_remote_code": False})
        mx.synchronize(mx.cpu)
        loaded = event(output / "runtime-events.jsonl", "phase_end", stage="load")
        receipt.update(status="LOADED", load_seconds=(loaded["monotonic_ns"] - receipt["started_monotonic_ns"]) / 1e9,
                       device=str(mx.default_device()), rss_after_load=process_rss(os.getpid()))
        atomic_json(output / "runtime-result.json", receipt)
        for index, prompt in enumerate(prompts, 1):
            stage = f"prompt_{index}"
            result = generate_one(mx, stream_generate, model, tokenizer, prompt, index,
                                  make_sampler(temp=0.0), output)
            receipt["prompts"].append(result)
            atomic_json(output / "runtime-result.json", receipt)
        final = cgroup_snapshot()
        verify_cgroup(final)
        receipt["memory_event_failures"] = memory_event_failures(snapshot, final)
        if (final["memory.swap.current"] != "0" or int(final["memory.peak"]) >= receipt["memory_limit"]["resolved_bytes"]
                or receipt["memory_event_failures"]):
            raise RuntimeError("FAIL_RESOURCE: memory ceiling or swap usage")
        receipt.update(status="PASS_LINUX_CPU_TINY_SMOKE" if all(p["status"] == "COMPLETED_32" for p in receipt["prompts"]) else "SHORT_OUTPUT",
                       cgroup_final=final, ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        return 0 if receipt["status"] == "PASS_LINUX_CPU_TINY_SMOKE" else 1
    except BaseException as exc:
        status = next((prefix for prefix in ("FAIL_RESOURCE", "TIMEOUT_PROMPT")
                       if str(exc).startswith(prefix)), "UNAVAILABLE")
        receipt.update(status=status, failed_stage=stage, error=traceback.format_exc())
        return 1
    finally:
        receipt["ended_monotonic_ns"] = time.monotonic_ns()
        atomic_json(output / "runtime-result.json", receipt)
        event(output / "runtime-events.jsonl", "worker_end", status=receipt["status"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/model"))
    parser.add_argument("--output", type=Path, default=Path("/evidence"))
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.model, arguments.output))
