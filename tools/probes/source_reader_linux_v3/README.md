# Isolated Linux CPU resource probe v3

CODE ONLY. V3 repairs the exact page-aligned cgroup-limit check after public v2 run43157502433 (commit6e3e184) failed in preflight before any model/tokenizer load. The observed memory.max was5999996928, swap.max0 and cpu.max400000100000. V1 run43153277100 failed earlier in logging-driver startup before PID creation. Both prior source snapshots and first-failure receipts remain preserved; previous-source-manifest.json records v1/v2 source hashes.

The requested Docker memory limit remains exactly6,000,000,000 bytes. V3 accepts only `(requested // os.sysconf("SC_PAGE_SIZE")) * page_size`, positive and no larger than the request. With4096-byte pages this is5,999,996,928 bytes. Unlimited, unrounded and adjacent-page values fail. This is an exact OS-page rule, not a tolerance or raised limit. Runtime durably writes the requested/resolved/page_size identity and actual preflight cgroup snapshot before validation. The final peak must remain below that resolved ceiling.

The host sampler retains first_valid_sample and last_valid_sample separately from the last attempted sample. A disappearing final cgroup is recorded as unavailable and does not erase earlier valid evidence. It does not authorize a success without the worker's own completed receipts.

V2 user/log fixes remain: compress=false at max-file1; existing host UID:GID; HOME=/tmp; unchanged64-MiB tmpfs owned by that UID:GID with0700. No chmod777, capabilities or privilege expansion. The five files Dockerfile/download_assets.py/requirements-lock.txt/model-manifest.json/probe-prompts.json remain byte-identical to v2. Generation, model revision, all eleven public assets,512→32 tokens, two fixed prompts, CPU4, no swap,180s phase deadlines and600s worker bound are unchanged. Runtime edits only save and validate the normalized memory limit.

Only static checks, fake tokenizer/response objects, and one ordinary sleeping child process were used locally. No real NN/tokenizer, Docker, CI, API or publication occurred for this v3 change; no local pilot/private/evaluation data was read. Run the non-neural suite with `PYTHONDONTWRITEBYTECODE=1 python3 tools/probes/source_reader_linux_v3/test_probe.py`.

ROOT alone reviews/publishes a new branch and dispatches a separately authorized third attempt. Copy this directory's workflow.yml into the already registered `.github/workflows/container.yml` on that reviewed public branch; dispatch the explicit branch ref. The workflow is manual-only and artifacts remain restricted to synthetic probe-evidence/. No auto-rerun is introduced.

A completed tiny smoke proves neither source-reader semantics nor full8192/2048 resource use, VPS latency, full-app memory or release readiness. The maximum remains two fixed synthetic generations.
