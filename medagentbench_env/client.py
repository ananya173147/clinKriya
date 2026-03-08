# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MedAgentBench Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import MedAgentBenchAction, MedAgentBenchObservation, MedAgentBenchState


class MedAgentBenchEnv(
    EnvClient[MedAgentBenchAction, MedAgentBenchObservation, MedAgentBenchState]
):
    """
    Client for the MedAgentBench RL Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with MedAgentBenchEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.instruction)
        ...
        ...     action = MedAgentBenchAction(
        ...         action_type="GET",
        ...         url="http://localhost:8080/fhir/Patient?name=Peter",
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.response_text)
    """

    def _step_payload(self, action: MedAgentBenchAction) -> Dict:
        """Convert action to JSON payload for the step message."""
        payload = {
            "action_type": action.action_type.value,
            "url": action.url,
            "raw_response": action.raw_response,
        }
        if action.body is not None:
            payload["body"] = action.body
        if action.answer is not None:
            payload["answer"] = action.answer
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[MedAgentBenchObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = MedAgentBenchObservation(
            task_id=obs_data.get("task_id", ""),
            instruction=obs_data.get("instruction", ""),
            context=obs_data.get("context", ""),
            available_functions=obs_data.get("available_functions", []),
            response_text=obs_data.get("response_text", ""),
            error=obs_data.get("error"),
            task_status=obs_data.get("task_status", "running"),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 8),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return MedAgentBenchState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
