"""
Lightweight mock FHIR HTTP server backed by fhir_cache.json.

Serves cached GET responses and accepts POST/DELETE requests so
rl_training/ can run fully without a real FHIR Docker container.

Authentic in the ways that matter for training:
  - Real HTTP calls on port 8080 (same interface as Docker FHIR)
  - Real FHIR bundle responses from cache
  - POST accepted and echoed back with a generated ID (agent learns
    correct payload structure)
  - DELETE no-ops (episodes are stateless — fhir_reset is a no-op)
  - GET /fhir/metadata passes the verify_fhir_server() health check

Usage:
    python -m rl_training.mock_fhir_server          # default port 8080
    python -m rl_training.mock_fhir_server --port 8080 --cache medagentbench_env/data/fhir_cache.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("Missing dependencies. Run: pip install fastapi uvicorn")
    sys.exit(1)

# ── Load MockFHIR ────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_DEFAULT_CACHE = _HERE.parent / "medagentbench_env" / "data" / "fhir_cache.json"

def _load_mock(cache_path: str, fhir_base: str):
    spec = importlib.util.spec_from_file_location(
        "fhir_cache",
        _HERE.parent / "medagentbench_env" / "server" / "fhir_cache.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MockFHIR.from_cache(cache_path, fhir_base)


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="MockFHIR")
_mock = None  # initialised in main()


@app.get("/fhir/metadata")
@app.get("/fhir/metadata")
async def metadata():
    """Health check — verify_fhir_server() hits this endpoint."""
    return JSONResponse({
        "resourceType": "CapabilityStatement",
        "status": "active",
        "fhirVersion": "4.0.1",
    })


@app.get("/fhir/{resource}")
async def fhir_get(resource: str, request: Request):
    """Serve cached FHIR GET responses."""
    url = str(request.url).replace(str(request.base_url).rstrip("/"), "http://localhost:8080")
    resp = _mock.get(url)
    data = resp.get("data", {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []})
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            pass
    return JSONResponse(data)


@app.post("/fhir/{resource}")
async def fhir_post(resource: str, request: Request):
    """Accept POST (resource creation) — echo back with generated ID.

    The agent learns to construct correct FHIR payloads. The resource
    is not actually persisted (episodes are reset between runs anyway).
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Attach server-generated metadata so callers get a well-formed response
    body["id"] = str(uuid.uuid4())[:8]
    body["meta"] = {
        "lastUpdated": datetime.now(tz=timezone.utc).isoformat(),
        "versionId": "1",
    }
    return JSONResponse(body, status_code=201)


@app.delete("/fhir/{resource}/{resource_id}")
async def fhir_delete(resource: str, resource_id: str):
    """Accept DELETE — no-op since we don't persist POSTs."""
    return JSONResponse({"resourceType": resource, "id": resource_id}, status_code=200)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mock FHIR HTTP server for RL training")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--cache", type=str, default=str(_DEFAULT_CACHE),
        help="Path to fhir_cache.json",
    )
    args = parser.parse_args()

    global _mock
    fhir_base = f"http://localhost:{args.port}/fhir/"
    print(f"Loading FHIR cache from {args.cache}...")
    _mock = _load_mock(args.cache, fhir_base)
    print(f"MockFHIR server starting on http://{args.host}:{args.port}/fhir/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
