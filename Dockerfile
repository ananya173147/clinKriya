# Dockerfile for Northflank CI/CD.
#
# The combined service runs the OpenEnv environment server — it exposes
# the /reset, /step, /state, /schema, and /ws endpoints that training
# connects to.  Training is triggered separately (Northflank job or local).
#
# Prerequisites before building:
#   Build the FHIR cache once against the real server (run locally):
#     cd medagentbench_env && uv run python -m server.fhir_cache --build \
#         --fhir-url http://localhost:8080/fhir/ \
#         --output data/fhir_cache.json
#   Then commit data/fhir_cache.json — it will be baked into the image.

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

# ── Source code ───────────────────────────────────────────────────────────
COPY medagentbench_env/ ./medagentbench_env/

# ── MedAgentBench data (FHIR functions + eval module) ─────────────────────
COPY medagentbenchv2/medagentbench_v2/src/ ./medagentbenchv2/medagentbench_v2/src/

# ── Python package + deps ─────────────────────────────────────────────────
RUN uv venv --python 3.11 /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -e "medagentbench_env[train]"

# ── Runtime environment ───────────────────────────────────────────────────
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV OUTPUT_DIR=/output

RUN mkdir -p /output

# ── Expose env server port ────────────────────────────────────────────────
EXPOSE 8000

# Run the OpenEnv environment server.
# Training connects to this service via the ENV_URL env var set in the
# training job (e.g. http://<northflank-service-url>:8000).
CMD ["python", "-m", "uvicorn", "medagentbench_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
