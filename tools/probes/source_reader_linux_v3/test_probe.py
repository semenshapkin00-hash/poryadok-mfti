"""Static and fake-process tests only: never import NN/tokenizer/Docker clients."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import common
import runtimeprobe
import supervisor


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeMX:
    cpu = "fake-cpu"

    @staticmethod
    def new_stream(_device):
        return "fake-stream"

    @staticmethod
    def synchronize(_stream):
        pass

    @staticmethod
    def isnan(value):
        return value == "nan"

    @staticmethod
    def any(value):
        return Scalar(value)


class CharacterTokenizer:
    """A deliberately tiny fake, not a model tokenizer."""
    eos_token_ids = {999}

    def apply_chat_template(self, messages, **_kwargs):
        return messages[0]["content"] + "\n" + messages[1]["content"] + "\nASSISTANT:"

    def encode(self, text, **_kwargs):
        return list(text.encode("utf-8"))


class ProbeTests(unittest.TestCase):
    def test_no_top_level_neural_imports_and_cpu_selected_first(self):
        source = (common.ROOT / "runtimeprobe.py").read_text()
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(a.name.startswith(("mlx", "transformers", "torch")) for a in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith(("mlx", "transformers", "torch")))
        selected = source.index("mx.set_default_device(mx.cpu)")
        self.assertLess(selected, source.index("from mlx_lm import load, stream_generate"))
        for path in common.ROOT.glob("*.py"):
            ast.parse(path.read_text(), filename=path.name)

    def test_exact_hash_pins_and_public_model_allowlist(self):
        text = (common.ROOT / "requirements-lock.txt").read_text()
        pins = re.findall(r"^([^#\s]+==[^\s]+) --hash=sha256:([0-9a-f]{64})$", text, re.M)
        self.assertEqual(len(pins), 34)
        self.assertEqual(len({p[0].split("==")[0] for p in pins}), 34)
        self.assertNotIn("mlx-metal", text)
        self.assertNotIn("cuda", text)
        data = common.model_manifest()
        self.assertEqual(sum(r["bytes"] for r in data["files"]), 2_278_969_697)
        weight = next(r for r in data["files"] if r["name"] == "model.safetensors")
        self.assertEqual(weight["sha256"], "2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f")
        prompts = json.loads((common.ROOT / "probe-prompts.json").read_text())
        self.assertEqual(len(prompts["prompts"]), 2)

    def test_workflow_is_manual_and_artifact_path_is_evidence_only(self):
        workflow = (common.ROOT / "workflow.yml").read_text()
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotRegex(workflow, r"(?m)^\s+(push|pull_request|schedule):")
        self.assertIn("path: probe-evidence/", workflow)
        self.assertNotIn("path: ${{ runner.temp }}", workflow)
        self.assertIn("--network\", \"none", (common.ROOT / "supervisor.py").read_text())
        dockerfile = (common.ROOT / "Dockerfile").read_text()
        self.assertIn("sha256:1aaa65a85fda306ffb8b910824d4e93bdce61e212c7e87168123ea3073b41a1a", dockerfile)
        self.assertEqual(dockerfile.count("--require-hashes"), 2)

    def test_cgroup_rejects_unlimited_memory_swap_and_cpu(self):
        valid = {"memory.max": "5999996928", "memory.swap.max": "0", "cpu.max": "400000 100000",
                 "memory.current": "1", "memory.peak": "2", "memory.swap.current": "0", "memory.events": "oom 0"}
        with patch.object(common.os, "sysconf", return_value=4096):
            common.verify_cgroup(valid)
            for key, value in (("memory.max", "max"), ("memory.swap.max", "1"), ("cpu.max", "max 100000")):
                with self.assertRaises(RuntimeError):
                    common.verify_cgroup({**valid, key: value})

    def test_only_exact_os_page_floor_is_accepted(self):
        valid = {"memory.max": "5999996928", "memory.swap.max": "0", "cpu.max": "400000 100000",
                 "memory.current": "1", "memory.peak": "2", "memory.swap.current": "0", "memory.events": "oom 0"}
        for page_size in (4096, 16384, 65536):
            with patch.object(common.os, "sysconf", return_value=page_size) as mocked:
                resolved = (6_000_000_000 // page_size) * page_size
                self.assertEqual(common.memory_limit(), {"requested_bytes": 6_000_000_000,
                                                       "resolved_bytes": resolved, "page_size": page_size})
                mocked.assert_called_with("SC_PAGE_SIZE")
                common.verify_cgroup({**valid, "memory.max": str(resolved)})
                for wrong in ("max", str(resolved - page_size), str(resolved + page_size), "6000000000", "6000004096"):
                    with self.assertRaises(RuntimeError):
                        common.verify_cgroup({**valid, "memory.max": wrong})
        for invalid in (0, -1, True, 4096.0, 6_000_000_001):
            with patch.object(common.os, "sysconf", return_value=invalid):
                with self.assertRaises(RuntimeError):
                    common.memory_limit()

    def test_preflight_rejection_preserves_snapshot_before_any_neural_import(self):
        snapshot = {"memory.max": "max", "memory.swap.max": "0", "cpu.max": "400000 100000"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(runtimeprobe.sys, "platform", "linux"), patch.object(runtimeprobe.sys, "version_info", (3, 12, 14)), \
                 patch.object(runtimeprobe.platform, "machine", return_value="x86_64"), \
                 patch.object(runtimeprobe, "cgroup_snapshot", return_value=snapshot), \
                 patch.object(common.os, "sysconf", return_value=4096):
                self.assertEqual(runtimeprobe.run(Path("/model-not-opened"), output), 1)
            result = json.loads((output / "runtime-result.json").read_text())
            self.assertEqual(result["failed_stage"], "preflight")
            self.assertEqual(result["cgroup_initial"], snapshot)
            self.assertEqual(result["memory_limit"], {"requested_bytes": 6_000_000_000,
                                                     "resolved_bytes": 5_999_996_928, "page_size": 4096})

    def test_exact_512_assembly_preserves_seed(self):
        result = runtimeprobe.assemble_prompt(CharacterTokenizer(), "system", "seed unchanged", "Synthetic filler line.\n")
        self.assertEqual(len(result["input_ids"]), 512)
        self.assertIn("seed unchanged", result["rendered_text"])
        self.assertTrue(result["rendered_text"].endswith("\nASSISTANT:"))

    def test_completed_phase_cannot_hide_late_deadline(self):
        rows = [{"event": "phase_start", "stage": "load", "monotonic_ns": 1_000_000_000},
                {"event": "phase_end", "stage": "load", "monotonic_ns": 182_000_000_000}]
        self.assertEqual(supervisor.deadline_reason(182.1, 1, rows), "TIMEOUT_LOAD")

    def test_oom_without_killed_main_process_is_failure(self):
        before = {"memory.events": "max 0\noom 0\noom_kill 0"}
        after = {"memory.events": "max 1\noom 1\noom_kill 0"}
        self.assertEqual(common.memory_event_failures(before, after), {"max": 1, "oom": 1})

    def test_nonzero_exit_cannot_reuse_saved_pass(self):
        runtime = {"status": "PASS_LINUX_CPU_TINY_SMOKE"}
        self.assertEqual(supervisor.final_status({"ExitCode": 0, "Running": False}, runtime, None, 0), runtime["status"])
        self.assertEqual(supervisor.final_status({"ExitCode": 1, "Running": False}, runtime, None, 0), "UNAVAILABLE")
        self.assertEqual(supervisor.final_status({"ExitCode": 0, "Running": False}, runtime, None, 1), "UNAVAILABLE")
        self.assertEqual(supervisor.final_status({"ExitCode": 137, "OOMKilled": True}, runtime, None, 137), "FAIL_RESOURCE_OOM")

    def fake_generation(self, count=32, stop=False):
        for position in range(1, count + 1):
            yield SimpleNamespace(generation_tokens=position, token=999 if stop and position == count else position,
                                  logprobs="finite", text=f"{position} ", prompt_tokens=512, prompt_tps=51200,
                                  finish_reason=("stop" if stop else "length") if position == count else None)

    def test_completed_32_has_actual_ids_and_correct_rates(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(runtimeprobe, "process_rss", return_value={}), patch.object(runtimeprobe, "cgroup_snapshot", return_value={}):
                result = runtimeprobe.generate_one(FakeMX(), lambda *_a, **_k: self.fake_generation(), None,
                                                  CharacterTokenizer(), {"input_ids": [1] * 512, "input_tokens": 512},
                                                  1, None, output)
            self.assertEqual(result["status"], "COMPLETED_32")
            self.assertEqual(result["output_token_ids"], list(range(1, 33)))
            self.assertEqual(len(result["token_monotonic_ns"]), 32)
            stamps = result["token_monotonic_ns"]
            self.assertEqual(result["rate32_including_prefill_tps"], 32e9 / (stamps[-1] - result["request_start_monotonic_ns"]))
            self.assertEqual(result["decode31_tps"], 31e9 / (stamps[-1] - stamps[0]))
            self.assertEqual(len(supervisor.events(output / "runtime-events.jsonl")), 34)

    def test_early_eos_never_claims_32_token_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(runtimeprobe, "process_rss", return_value={}), patch.object(runtimeprobe, "cgroup_snapshot", return_value={}):
                result = runtimeprobe.generate_one(FakeMX(), lambda *_a, **_k: self.fake_generation(4, True), None,
                                                  CharacterTokenizer(), {"input_ids": [1] * 512, "input_tokens": 512},
                                                  1, None, Path(temporary))
            self.assertEqual(result["status"], "SHORT_OUTPUT")
            self.assertEqual(result["generated_tokens"], 4)
            self.assertIsNone(result["rate32_including_prefill_tps"])

    def test_nan_aborts_with_first_token_durable(self):
        def fake(*_args, **_kwargs):
            yield SimpleNamespace(generation_tokens=1, token=1, logprobs="finite", text="one", finish_reason=None, prompt_tokens=512, prompt_tps=51200)
            yield SimpleNamespace(generation_tokens=2, token=2, logprobs="nan", text="two", finish_reason=None, prompt_tokens=512, prompt_tps=51200)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "NaN"):
                runtimeprobe.generate_one(FakeMX(), fake, None, CharacterTokenizer(),
                                          {"input_ids": [1] * 512, "input_tokens": 512}, 1, None, output)
            rows = supervisor.events(output / "runtime-events.jsonl")
            self.assertEqual([r["token"] for r in rows if r["event"] == "token"], [1])

    def test_external_timeout_stops_real_fake_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            started = time.monotonic()
            common.event(output / "runtime-events.jsonl", "phase_start", stage="load")
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                     start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                result = supervisor.monitor_process(child, output, started,
                    lambda: os.killpg(child.pid, signal.SIGKILL), lambda: {"rss": {"VmRSS_bytes": 1234}},
                    load_seconds=0.15, prompt_seconds=0.15, worker_seconds=1)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
            self.assertEqual(result["abort_reason"], "TIMEOUT_LOAD")
            self.assertEqual(result["sampled_maximum_rss_bytes"], 1234)
            self.assertNotEqual(result["attached_process_exit"], 0)
            self.assertTrue((output / "supervisor-events.jsonl").exists())

    def test_claim_prevents_silent_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "first.claim"
            common.claim(path)
            with self.assertRaises(FileExistsError):
                common.claim(path)

    def test_single_file_logging_disables_compression(self):
        source = (common.ROOT / "supervisor.py").read_text()
        self.assertIn('"--log-opt", "max-file=1"', source)
        self.assertIn('"--log-opt", "compress=false"', source)
        self.assertIn('"--cap-drop", "ALL"', source)
        self.assertIn('"--security-opt", "no-new-privileges"', source)
        self.assertNotIn('"--privileged"', source)

    def test_v1_hash_manifest_preserves_model_budget_and_runtime_bytes(self):
        source = json.loads((common.ROOT / "v1-source-manifest.json").read_text())
        self.assertEqual(source["first_ci_run"], "43153277100")
        self.assertEqual(source["first_public_commit"], "9ac6329faec3ceb9e63680fb7a0f2eeeb500b1e3")
        for name in ("download_assets.py", "Dockerfile",
                     "model-manifest.json", "probe-prompts.json", "requirements-lock.txt"):
            self.assertEqual(common.sha256(common.ROOT / name), source["files"][name]["sha256"], name)

    def test_v2_fixed_inputs_and_budgets_are_preserved(self):
        source = json.loads((common.ROOT / "previous-source-manifest.json").read_text())["sources"]["v2"]
        for name in ("download_assets.py", "Dockerfile", "model-manifest.json", "probe-prompts.json", "requirements-lock.txt"):
            self.assertEqual(common.sha256(common.ROOT / name), source[name]["sha256"], name)
        self.assertEqual((common.MEMORY_BYTES, common.CPU_COUNT, common.LOAD_SECONDS,
                          common.PROMPT_SECONDS, common.WORKER_SECONDS), (6_000_000_000, 4, 180, 180, 600))

    def test_disappearing_final_cgroup_keeps_first_valid_sample(self):
        class FinishedFakeProcess:
            polls = 0
            def poll(self):
                self.polls += 1
                return None if self.polls <= 2 else 0
            def wait(self, **_kwargs):
                return 0
        first = {"cgroup": {"memory.max": "5999996928", "memory.current": "1"}, "rss": {"VmRSS_bytes": 1234}}
        gone = {"cgroup": {"memory.max": {"unavailable": "FileNotFoundError"}}, "rss": {"unavailable": "process exited"}}
        samples = iter((first, gone))
        with tempfile.TemporaryDirectory() as temporary, patch.object(supervisor.time, "sleep"):
            result = supervisor.monitor_process(FinishedFakeProcess(), Path(temporary), time.monotonic(),
                                                lambda: self.fail("No abort is needed"), lambda: next(samples))
        self.assertEqual(result["first_valid_sample"], first)
        self.assertEqual(result["last_valid_sample"], first)
        self.assertEqual(result["last_sample"], gone)
        self.assertIsNone(result["abort_reason"])

    def test_existing_host_user_can_write_evidence_without_mode_expansion(self):
        with patch.object(supervisor.os, "getuid", return_value=1001), patch.object(supervisor.os, "getgid", return_value=1002):
            options = supervisor.identity_options()
        self.assertEqual(options[:4], ["--user", "1001:1002", "--env", "HOME=/tmp"])
        self.assertIn("size=67108864", options[-1])
        self.assertTrue(options[-1].endswith("uid=1001,gid=1002,mode=0700"))
        self.assertNotIn("777", options[-1])
        with patch.object(supervisor.os, "getuid", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "non-root runner"):
                supervisor.identity_options()


if __name__ == "__main__":
    unittest.main(verbosity=2)
