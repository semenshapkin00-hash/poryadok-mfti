# Isolated Linux CPU resource probe v2

This is an explicitly authorized configuration repair after first CI `43153277100`, public commit `9ac6329faec3ceb9e63680fb7a0f2eeeb500b1e3`. That run verified all eleven public asset SHAs and built the image/passed pip check, then failed before the model process (`Pid=0`, zero StartedAt, no runtime receipt): `failed to initialize logging driver: compression cannot be enabled when max file count is 1`. The first run and v1 source remain preserved. This directory has not run a second CI or model call.

The only execution fixes are explicit `compress=false` for local log rotation with `max-file=1`, and the existing non-root host UID:GID for the evidence bind plus `HOME=/tmp`. The same 64-MiB tmpfs belongs to that UID:GID with mode0700; no evidence chmod, added capabilities or privilege expansion. Docker supports uid/gid/mode tmpfs options in its [official documentation](https://docs.docker.com/engine/storage/tmpfs/). Container inspection must confirm these options before start. V1 source hashes are in `v1-source-manifest.json`.

CODE ONLY: no real tokenizer, model, Docker build, Actions run or download was used to validate this directory locally. `test_probe.py` uses fake tokenizers/responses and one ordinary sleeping child process. The authoritative plan is `docs/SOURCE_READER_LINUX_PROBE_PLAN_20261112.md`.

ROOT must copy only this directory into a **new public clone/branch from immutable exact9b**. Replace the already registered `.github/workflows/container.yml` on that branch with this directory's `workflow.yml`; preserve the latter copy for code identity. Do not install another workflow only on the branch and assume GitHub registered it. Dispatch that registered workflow with the explicit reviewed branch ref. No workflow runs on push/PR.

Review the complete branch diff before dispatch. No live application, evaluation, experiment, private file, credential or weight copy belongs in the branch. Model files are downloaded anonymously during preparation and remain in the runner temporary directory. They are not artifacts. The image contains only the pinned binary dependencies and probe code. Runtime has no network, one worker and two sequential 512-to-32 calls.

The workflow performs binary-only hash-checked preparation, then uses the host supervisor to create the restricted container and preserve first receipts. Reusing an evidence/model directory fails; no retry or raised limits is built in. Uploads contain only `probe-evidence/`, including synthetic prompt/output tokens, resource samples and errors. Treat `supervisor-result.json` as the final status; a green build or asset download is not an inference pass.

Run only the non-neural checks with `PYTHONDONTWRITEBYTECODE=1 python3 tools/probes/source_reader_linux_v2/test_probe.py`.

Runtime code, budgets (6e9 bytes/no swap/4 CPUs/180s stages/600s worker), Python/image digest,34 hash-pinned wheels, model revision/eleven files and both512→32 prompts are byte-identical to v1. The bound remains at most two fixed synthetic generations. A new reviewed public branch/commit and separate authorization precede any second run; this source fix does not trigger a rerun.

A tiny-smoke pass proves neither source-reader semantics nor 8192/2048 resource use, VPS latency, resident full-app memory or release readiness. Unsupported package/backend, timeout, OOM and short outputs remain first-run failures.
