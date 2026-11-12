# Isolated Linux CPU resource probe

CODE ONLY: no real tokenizer, model, Docker build, Actions run or download was used to validate this directory locally. `test_probe.py` uses fake tokenizers/responses and one ordinary sleeping child process. The authoritative plan is `docs/SOURCE_READER_LINUX_PROBE_PLAN_20261112.md`.

ROOT must copy only this directory into a **new public clone/branch from immutable exact9b**. Replace the already registered `.github/workflows/container.yml` on that branch with this directory's `workflow.yml`; preserve the latter copy for code identity. Do not install another workflow only on the branch and assume GitHub registered it. Dispatch that registered workflow with the explicit reviewed branch ref. No workflow runs on push/PR.

Review the complete branch diff before dispatch. No live application, evaluation, experiment, private file, credential or weight copy belongs in the branch. Model files are downloaded anonymously during preparation and remain in the runner temporary directory. They are not artifacts. The image contains only the pinned binary dependencies and probe code. Runtime has no network, one worker and two sequential 512-to-32 calls.

The workflow performs binary-only hash-checked preparation, then uses the host supervisor to create the restricted container and preserve first receipts. Reusing an evidence/model directory fails; no retry or raised limits is built in. Uploads contain only `probe-evidence/`, including synthetic prompt/output tokens, resource samples and errors. Treat `supervisor-result.json` as the final status; a green build or asset download is not an inference pass.

Run only the non-neural checks with `PYTHONDONTWRITEBYTECODE=1 python3 tools/probes/source_reader_linux_v1/test_probe.py`.

A tiny-smoke pass proves neither source-reader semantics nor 8192/2048 resource use, VPS latency, resident full-app memory or release readiness. Unsupported package/backend, timeout, OOM and short outputs remain first-run failures.
