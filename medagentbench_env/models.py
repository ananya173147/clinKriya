# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the MedAgentBench RL Environment.

Wraps MedAgentBench v2's clinical decision-making benchmark as an OpenEnv
environment. Agents interact with a FHIR EHR server via GET/POST requests
and signal completion with FINISH.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """The three action types an agent can take."""

    GET = "GET"
    POST = "POST"
    FINISH = "FINISH"


class TaskStatus(str, Enum):
    """Outcome status for a completed episode."""

    RUNNING = "running"
    COMPLETED = "completed"
    AGENT_CONTEXT_LIMIT = "agent_context_limit"
    AGENT_INVALID_ACTION = "agent_invalid_action"
    TASK_LIMIT_REACHED = "task_limit_reached"
    TASK_ERROR = "task_error"


# ---------------------------------------------------------------------------
# Task / scenario metadata
# ---------------------------------------------------------------------------


class TaskSample(BaseModel):
    """A single task from the MedAgentBench benchmark."""

    id: str = Field(..., description="Task identifier, e.g. 'task1_1'")
    instruction: str = Field(..., description="Natural-language clinical instruction")
    context: str = Field(default="", description="Additional clinical context")
    sol: List[str] = Field(default_factory=list, description="Expected solution values")
    eval_MRN: str = Field(default="", description="Patient MRN used for evaluation")


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in the agent-environment conversation."""

    role: str = Field(..., description="'user' (environment) or 'agent'")
    content: str = Field(..., description="Message text")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class MedAgentBenchAction(Action):
    """Action submitted by the agent each step.

    The agent produces one of:
      - GET <url>            → query the FHIR server
      - POST <url> {json}   → create/update a FHIR resource
      - FINISH([answers])   → end the episode with a result
    """

    action_type: ActionType = Field(..., description="GET, POST, or FINISH")
    url: str = Field(default="", description="FHIR API endpoint (for GET/POST)")
    body: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON payload for POST requests"
    )
    answer: Optional[List[Any]] = Field(
        default=None,
        description="Result list for FINISH actions, e.g. ['S6534835']",
    )
    raw_response: str = Field(
        default="",
        description="The agent's raw text response before parsing",
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


class MedAgentBenchObservation(Observation):
    """Observation returned to the agent after each step.

    On reset: contains the system prompt with task instruction and available
    FHIR functions.
    On step: contains the FHIR server response or an error message.
    On done: includes reward (1.0 = pass, 0.0 = fail) and task status.
    """

    # Task context (populated on reset)
    task_id: str = Field(default="", description="Current task identifier")
    instruction: str = Field(default="", description="Clinical task instruction")
    context: str = Field(default="", description="Additional clinical context")
    available_functions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="FHIR API function definitions available to the agent",
    )

    # Step response
    response_text: str = Field(
        default="",
        description="FHIR server response or environment feedback",
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the action was invalid"
    )

    # Episode outcome
    task_status: TaskStatus = Field(
        default=TaskStatus.RUNNING,
        description="Current status of the episode",
    )
    step_number: int = Field(default=0, description="Current step in the episode")
    max_steps: int = Field(default=8, description="Maximum steps allowed")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class MedAgentBenchState(State):
    """Internal environment state tracked across steps."""

    task_sample: Optional[TaskSample] = Field(
        default=None, description="The current task being solved"
    )
    chat_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Full conversation history for this episode",
    )
    post_requests: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All POST payloads the agent has submitted (used for evaluation)",
    )
    fhir_api_base: str = Field(
        default="http://localhost:8080/fhir/",
        description="Base URL of the FHIR server",
    )
    task_status: TaskStatus = Field(
        default=TaskStatus.RUNNING,
        description="Current episode outcome status",
    )
    agent_answer: Optional[List[Any]] = Field(
        default=None,
        description="The agent's FINISH answer, if provided",
    )
