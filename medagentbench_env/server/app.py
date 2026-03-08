# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the MedAgentBench RL Environment.

Endpoints:
    - POST /reset: Reset the environment (start a new clinical task)
    - POST /step: Execute an action (GET/POST/FINISH)
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import json
from pathlib import Path

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

from fastapi import HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from medagentbench_env.models import MedAgentBenchAction, MedAgentBenchObservation
from .medagentbench_env_environment import MedAgentBenchEnvironment

_ROOT = Path(__file__).parent.parent
_UI_HTML = (_ROOT / "ui" / "index.html").read_text()


class _UIMiddleware(BaseHTTPMiddleware):
    """Serve custom UI for /web and / before OpenEnv's default handler."""
    async def dispatch(self, request: Request, call_next):
        p = request.url.path
        if p == "/" or p == "/ui" or p == "/web" or p.startswith("/web/"):
            return HTMLResponse(content=_UI_HTML)
        return await call_next(request)


app = create_app(
    MedAgentBenchEnvironment,
    MedAgentBenchAction,
    MedAgentBenchObservation,
    env_name="medagentbench_env",
    max_concurrent_envs=1,
)

app.add_middleware(_UIMiddleware)


@app.get("/api/tasks")
async def get_tasks():
    """Return the task list (instruction, context, MRN, type) for the UI."""
    tasks_path = _ROOT / "data" / "stratified_benchmark.json"
    if not tasks_path.exists():
        raise HTTPException(status_code=404, detail="stratified_benchmark.json not found")
    with open(tasks_path) as f:
        tasks = json.load(f)
    return JSONResponse(content=[
        {
            "index": i,
            "id": t["id"],
            "task_type": t["id"].split("_")[0],
            "instruction": t["instruction"],
            "context": t.get("context", ""),
            "eval_MRN": t.get("eval_MRN", ""),
        }
        for i, t in enumerate(tasks)
    ])


@app.get("/api/baseline-results")
async def get_baseline_results():
    """Return pre-computed baseline evaluation results."""
    results_path = _ROOT / "data" / "baseline_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="baseline_results.json not found")
    with open(results_path) as f:
        return JSONResponse(content=json.load(f))




def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
