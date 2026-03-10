# Dockerfile for Northflank CI/CD.
#
# Runs the OpenEnv environment server — exposes /reset, /step, /state,
# /schema, /ws, and the UI at /.  Training connects separately via ENV_URL.
#
# No GPU required: the env server only does FHIR lookups against the
# baked-in cache and runs the FastAPI server.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

# ── uv ────────────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv   /usr/local/bin/uv && \
    mv /root/.local/bin/uvx  /usr/local/bin/uvx

WORKDIR /app

# ── Source code ───────────────────────────────────────────────────────────
COPY medagentbench_env/ ./medagentbench_env/

# ── MedAgentBench eval module (refsol graders) ────────────────────────────
COPY medagentbenchv2/medagentbench_v2/src/ ./medagentbenchv2/medagentbench_v2/src/

# ── Python package + deps (env server only, no train extras) ──────────────
RUN uv venv --python 3.11 /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -e "medagentbench_env"

# ── Runtime environment ───────────────────────────────────────────────────
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

# ── Expose env server port ────────────────────────────────────────────────
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the OpenEnv environment server.
# Training connects to this service via the ENV_URL env var.
CMD ["uvicorn", "medagentbench_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
