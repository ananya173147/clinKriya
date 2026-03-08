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
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_ROOT = Path(__file__).parent.parent
_UI_HTML_PATH = _ROOT / "ui" / "index.html"
_UI_HTML = _UI_HTML_PATH.read_text() if _UI_HTML_PATH.exists() else "<h1>MedAgentBench Env</h1>"

_OPENENV_AVAILABLE = False
try:
    from openenv.core.env_server.http_server import create_app
    from medagentbench_env.models import MedAgentBenchAction, MedAgentBenchObservation
    from .medagentbench_env_environment import MedAgentBenchEnvironment
    _OPENENV_AVAILABLE = True
except Exception:
    pass


class _UIMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        p = request.url.path
        if p == "/" or p == "/ui" or p == "/web" or p.startswith("/web/"):
            return HTMLResponse(content=_UI_HTML)
        return await call_next(request)


if _OPENENV_AVAILABLE:
    app = create_app(
        MedAgentBenchEnvironment,
        MedAgentBenchAction,
        MedAgentBenchObservation,
        env_name="medagentbench_env",
        max_concurrent_envs=1,
    )
else:
    # Standalone fallback app when openenv-core is not installed
    app = FastAPI(title="MedAgentBench Env", version="0.1.0")

    _env_state: Dict[str, Any] = {"task_index": 0, "step": 0, "done": False}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/reset")
    async def reset(body: Optional[Dict[str, Any]] = None):
        _env_state.update({"task_index": _env_state["task_index"], "step": 0, "done": False})
        tasks_path = _ROOT / "data" / "stratified_benchmark.json"
        task = {}
        if tasks_path.exists():
            with open(tasks_path) as f:
                tasks = json.load(f)
            idx = (body or {}).get("task_index", _env_state["task_index"]) % len(tasks)
            task = tasks[idx]
            _env_state["task_index"] = idx
            _env_state["task"] = task
        return JSONResponse({"done": False, "reward": 0.0, "task_id": task.get("id", ""),
                             "instruction": task.get("instruction", ""), "step_number": 0})

    @app.post("/step")
    async def step(body: Dict[str, Any]):
        _env_state["step"] += 1
        action_type = body.get("action_type", "FINISH")
        done = action_type == "FINISH" or _env_state["step"] >= 8
        reward = 0.0
        if action_type == "GET":
            response_text = "GET not available in standalone mode."
        elif action_type == "POST":
            response_text = "POST request accepted."
        else:
            done = True
            response_text = "Task completed."
        return JSONResponse({"done": done, "reward": reward,
                             "response_text": response_text,
                             "step_number": _env_state["step"]})

    @app.get("/state")
    async def state():
        return JSONResponse(_env_state)

    @app.get("/schema")
    async def schema():
        return JSONResponse({"action": {"action_type": "GET|POST|FINISH", "url": "str", "body": "dict", "answer": "list"},
                             "observation": {"done": "bool", "reward": "float", "response_text": "str"}})

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
