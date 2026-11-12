FROM --platform=linux/amd64 python:3.12.14-slim-bookworm@sha256:1aaa65a85fda306ffb8b910824d4e93bdce61e212c7e87168123ea3073b41a1a
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    HF_HOME=/tmp/fresh-hf-cache TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
WORKDIR /probe
COPY requirements-lock.txt /probe/requirements-lock.txt
RUN python -m pip download --only-binary=:all: --require-hashes --no-cache-dir --retries 0 --timeout 60 \
      --dest /wheelhouse -r requirements-lock.txt \
 && python -m pip install --no-index --find-links=/wheelhouse --only-binary=:all: \
      --require-hashes --no-cache-dir -r requirements-lock.txt \
 && python -m pip check \
 && rm -rf /wheelhouse
COPY common.py runtimeprobe.py supervisor.py download_assets.py model-manifest.json probe-prompts.json workflow.yml /probe/
ENTRYPOINT ["python", "/probe/runtimeprobe.py", "--model", "/model", "--output", "/evidence"]
