ARG NODE_IMAGE=node:24.22.1-bookworm-slim@sha256:be8204d4bd327363357f532187b752ff62759c31bedf98e9e18fc08c9cbd605c
ARG PYTHON_IMAGE=python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

FROM ${NODE_IMAGE} AS ocr
WORKDIR /ocr
RUN npm install --global pnpm@11.25.0 --ignore-scripts --registry=https://registry.npmjs.org
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --prod --frozen-lockfile --ignore-scripts --registry=https://registry.npmjs.org

FROM ${PYTHON_IMAGE} AS runtime
ARG DEBIAN_SNAPSHOT=20261111T000000Z
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 \
    ORT_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    NAVIGATOR_NODE_BINARY=/usr/local/bin/node NAVIGATOR_NODE_MODULES=/app/node_modules \
    NAVIGATOR_DATA_DIR=/app/data NAVIGATOR_REQUIRE_MODELS=1 PORT=8769
WORKDIR /app
# Fixed Debian snapshot, not floating apt security repositories.
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s bookworm main\ndeb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/%s bookworm-security main\n' "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list && \
    apt-get -o Acquire::Retries=3 update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgomp1 libstdc++6 libatomic1 tzdata ca-certificates fonts-dejavu-core && \
    dpkg-query -W > /container-system-packages.txt && rm -rf /var/lib/apt/lists/*
COPY requirements-lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements-lock.txt && python -m pip check && \
    python -m pip freeze --all > /container-python-packages.txt && \
    groupadd --gid 1000 app && useradd --uid 1000 --gid 1000 --create-home app && \
    mkdir -p /app/data /app/models /app/.cache && chown -R app:app /app /home/app
COPY --from=ocr /usr/local/bin/node /usr/local/bin/node
COPY --from=ocr /ocr/node_modules /app/node_modules
# Never COPY . .: host uploads, reference/, data/ and weights are not image inputs.
COPY --chown=app:app app.py ./
COPY --chown=app:app frontend/ ./frontend/
COPY --chown=app:app app_services/ ./app_services/
COPY --chown=app:app navigator_engine/ ./navigator_engine/
COPY --chown=app:app demo/__init__.py demo/fixtures.py demo/manifest.json ./demo/
COPY --chown=app:app demo/documents/ ./demo/documents/
COPY --chown=app:app tools/tesseract_ocr.cjs tools/container_models.py tools/container_check.py tools/container_entrypoint.py ./tools/
USER 1000:1000
RUN node --version && node -e "if(require('/app/node_modules/tesseract.js/package.json').version!=='7.0.0')process.exit(1)"
EXPOSE 8769
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD ["python", "tools/container_check.py", "--health"]
ENTRYPOINT ["python", "tools/container_entrypoint.py"]

# Default Docker-based public Space. Only public SHA-pinned model files are downloaded.
# Local Compose explicitly targets runtime and initializes its model volume separately.
FROM runtime AS space
ENV PORT=7860 NAVIGATOR_PUBLIC_DEMO=1 NAVIGATOR_SEED_DEMO=1 NAVIGATOR_DATA_DIR=/home/app/data
RUN python tools/container_models.py --download --target /app/models && mkdir -p /home/app/data
EXPOSE 7860
