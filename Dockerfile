# Single Dockerfile for Northflank CI/CD.
#
# Starts the OpenEnv environment server (with cached mock FHIR — no real
# FHIR Docker container needed), then runs GRPO training against it.
#
# Prerequisites before building:
#   Build the FHIR cache once against the real server (run locally):
#     cd medagentbench_env && uv run python -m server.fhir_cache --build \
#         --fhir-url http://localhost:8080/fhir/ \
#         --output data/fhir_cache.json
#   Then commit data/fhir_cache.json — it will be baked into the image.
#
# Northflank env vars (set in service settings):
#   HF_TOKEN         — HuggingFace token (for downloading Qwen weights)
#   WANDB_API_KEY    — optional, for experiment tracking
#   TRAIN_ARGS       — optional extra args forwarded to train.py

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        curl git build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

# ── uv ────────────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv   /usr/local/bin/uv && \
    mv /root/.local/bin/uvx  /usr/local/bin/uvx

WORKDIR /app

# ── Python package + deps ─────────────────────────────────────────────────
# Copy package manifest first for better layer caching
COPY medagentbench_env/pyproject.toml medagentbench_env/uv.lock* ./medagentbench_env/

RUN uv venv --python 3.11 /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -e "medagentbench_env[train]"

# ── Source code ───────────────────────────────────────────────────────────
COPY medagentbench_env/ ./medagentbench_env/

# ── MedAgentBench data (FHIR functions + eval module) ─────────────────────
# Only copy the data + evaluation source — skip large output dirs
COPY medagentbenchv2/medagentbench_v2/src/ ./medagentbenchv2/medagentbench_v2/src/

# ── Runtime environment ───────────────────────────────────────────────────
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV OUTPUT_DIR=/output
ENV ENV_URL=http://localhost:8000

RUN mkdir -p /output

# ── Startup script ────────────────────────────────────────────────────────
# 1. Launch OpenEnv server (auto-detects fhir_cache.json → uses mock FHIR)
# 2. Wait until the server is ready
# 3. Run GRPO training against it
RUN cat > /app/start.sh <<'SCRIPT'
#!/bin/bash
set -euo pipefail

echo "=== Starting OpenEnv environment server ==="
python -m uvicorn medagentbench_env.server.app:app \
    --host 0.0.0.0 --port 8000 --log-level warning &
SERVER_PID=$!

echo "Waiting for env server..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/schema > /dev/null 2>&1; then
        echo "Env server ready (${i}s)."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Env server did not start within 60s." >&2
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

echo "=== Starting GRPO training ==="
python medagentbench_env/train.py \
    --env-url  http://localhost:8000 \
    --data-dir /app/medagentbench_env/data \
    --output-dir "${OUTPUT_DIR:-/output}" \
    ${TRAIN_ARGS:-}

EXIT_CODE=$?
echo "=== Training finished (exit $EXIT_CODE) ==="
kill $SERVER_PID 2>/dev/null || true
exit $EXIT_CODE
SCRIPT

RUN chmod +x /app/start.sh

# Northflank reads EXPOSE to create the public port definition.
# The env server listens here; training talks to it over localhost.
EXPOSE 8000

CMD ["/app/start.sh"]
