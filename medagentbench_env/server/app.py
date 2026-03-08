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

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

from medagentbench_env.models import MedAgentBenchAction, MedAgentBenchObservation
from .medagentbench_env_environment import MedAgentBenchEnvironment


app = create_app(
    MedAgentBenchEnvironment,
    MedAgentBenchAction,
    MedAgentBenchObservation,
    env_name="medagentbench_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
