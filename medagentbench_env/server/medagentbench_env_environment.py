# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MedAgentBench RL Environment Implementation.

Wraps the MedAgentBench v2 clinical decision-making benchmark as an
OpenEnv Gymnasium-style environment.  Each episode corresponds to one
clinical task where the agent interacts with a FHIR EHR server.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests
from openenv.core.env_server.interfaces import Environment

from medagentbench_env.models import (
    ActionType,
    ChatMessage,
    MedAgentBenchAction,
    MedAgentBenchObservation,
    MedAgentBenchState,
    TaskSample,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Paths to MedAgentBench v2 data (relative to this repo)
# ---------------------------------------------------------------------------
_MEDAGENTBENCH_ROOT = Path(__file__).resolve().parents[2]  # medagentbench_env/
_DEFAULT_DATA_DIR = (
    _MEDAGENTBENCH_ROOT
    / "medagentbenchv2"
    / "medagentbench_v2"
    / "src"
    / "MedAgentBench"
    / "data"
    / "medagentbench"
)

# System prompt template (from MedAgentBench v2)
_SYSTEM_PROMPT = """\
You are an expert in using FHIR functions to assist medical professionals. \
You are given a question and a set of possible functions. Based on the question, \
you will need to make one or more function/tool calls to achieve the purpose.

1. If you decide to invoke a GET function, you MUST put it in the format of
GET url?param_name1=param_value1&param_name2=param_value2...

2. If you decide to invoke a POST function, you MUST put it in the format of
POST url
[your payload data in JSON format]

3. If you have got answers for all the questions and finished all the requested \
tasks, you MUST call to finish the conversation in the format of \
(make sure the list is JSON loadable.)
FINISH([answer1, answer2, ...])

Your response must be in the format of one of the three cases, and you can \
call only one function each time. You SHOULD NOT include any other text in \
the response.

Here is a list of functions in JSON format that you can invoke. \
Note that you should use {api_base} as the api_base.
{functions}

Context: {context}
Question: {question}"""


# ---------------------------------------------------------------------------
# FHIR helpers (ported from medagentbenchv2 utils.py)
# ---------------------------------------------------------------------------

def _send_get_request(url: str) -> Dict[str, Any]:
    """Proxy a GET request to the FHIR server."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        data = response.json() if "application/json" in content_type else response.text
        return {"status_code": response.status_code, "data": data}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Evaluation helpers (ported from medagentbenchv2 eval.py / new_refsol.py)
# ---------------------------------------------------------------------------

def _load_eval_module():
    """Try to import the refsol evaluation module from medagentbenchv2."""
    refsol_path = (
        _MEDAGENTBENCH_ROOT.parent
        / "medagentbenchv2"
        / "medagentbench_v2"
        / "src"
        / "MedAgentBench"
        / "src"
        / "server"
        / "tasks"
        / "medagentbench"
    )
    if str(refsol_path) not in sys.path:
        sys.path.insert(0, str(refsol_path))
        # Also add parent paths needed by the refsol module's own imports
        src_root = refsol_path.parents[3]  # .../src
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
    try:
        import importlib
        refsol = importlib.import_module("new_refsol")
        return refsol
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MedAgentBenchEnvironment(
    Environment[MedAgentBenchAction, MedAgentBenchObservation, MedAgentBenchState]
):
    """
    OpenEnv environment wrapping MedAgentBench v2.

    Each episode is one clinical task. The agent sends GET/POST/FINISH
    actions and receives FHIR server responses as observations.  The
    episode terminates when the agent calls FINISH, exceeds max steps,
    or sends an invalid action.  Reward is binary: 1.0 if the task-
    specific evaluation passes, 0.0 otherwise.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        fhir_api_base: str = "http://localhost:8080/fhir/",
        data_file: Optional[str] = None,
        func_file: Optional[str] = None,
        max_steps: int = 8,
    ):
        super().__init__()
        self._fhir_api_base = fhir_api_base
        self._max_steps = max_steps

        # Load task data
        data_path = Path(data_file) if data_file else _DEFAULT_DATA_DIR / "test_data_v2.json"
        with open(data_path) as f:
            self._tasks: List[Dict[str, Any]] = json.load(f)

        # Load FHIR function definitions
        func_path = Path(func_file) if func_file else _DEFAULT_DATA_DIR / "funcs_v1.json"
        with open(func_path) as f:
            self._functions: List[Dict[str, Any]] = json.load(f)

        # Task index for sequential iteration (can be overridden in reset)
        self._task_index = 0

        # Internal state
        self._state = MedAgentBenchState()

        # Evaluation module (lazy-loaded)
        self._refsol = None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MedAgentBenchObservation:
        """Start a new episode with a task from the benchmark.

        Keyword args:
            task_index: int — select a specific task (0-299). Defaults to
                sequential iteration through the dataset.
        """
        task_index = kwargs.get("task_index", self._task_index)
        task_index = task_index % len(self._tasks)
        self._task_index = task_index + 1

        task_data = self._tasks[task_index]
        task_sample = TaskSample(
            id=task_data["id"],
            instruction=task_data["instruction"],
            context=task_data.get("context", ""),
            sol=task_data.get("sol", []),
            eval_MRN=task_data.get("eval_MRN", ""),
        )

        # Build the system prompt
        system_prompt = _SYSTEM_PROMPT.format(
            api_base=self._fhir_api_base,
            functions=json.dumps(self._functions),
            context=task_sample.context,
            question=task_sample.instruction,
        )

        # Initialize state
        self._state = MedAgentBenchState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_sample=task_sample,
            chat_history=[ChatMessage(role="user", content=system_prompt)],
            post_requests=[],
            fhir_api_base=self._fhir_api_base,
            task_status=TaskStatus.RUNNING,
            agent_answer=None,
        )

        return MedAgentBenchObservation(
            done=False,
            reward=0.0,
            task_id=task_sample.id,
            instruction=task_sample.instruction,
            context=task_sample.context,
            available_functions=self._functions,
            response_text=system_prompt,
            task_status=TaskStatus.RUNNING,
            step_number=0,
            max_steps=self._max_steps,
        )

    def step(
        self,
        action: MedAgentBenchAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MedAgentBenchObservation:
        """Process one agent action (GET / POST / FINISH)."""
        self._state.step_count += 1

        # Record the agent's raw response in history
        raw = action.raw_response or self._reconstruct_raw(action)
        self._state.chat_history.append(ChatMessage(role="agent", content=raw))

        # ---- FINISH ----
        if action.action_type == ActionType.FINISH:
            self._state.agent_answer = action.answer
            self._state.task_status = TaskStatus.COMPLETED
            reward = self._evaluate()
            env_msg = "Task completed."
            self._state.chat_history.append(ChatMessage(role="user", content=env_msg))
            return self._make_obs(
                response_text=env_msg,
                done=True,
                reward=reward,
            )

        # ---- GET ----
        if action.action_type == ActionType.GET:
            url = action.url
            if "&_format=json" not in url and "?_format=json" not in url:
                url += "&_format=json" if "?" in url else "?_format=json"

            get_res = _send_get_request(url)

            if "data" in get_res:
                data_str = (
                    json.dumps(get_res["data"])
                    if isinstance(get_res["data"], (dict, list))
                    else str(get_res["data"])
                )
                env_msg = (
                    f"Here is the response from the GET request:\n{data_str}. "
                    "Please call FINISH if you have got answers for all the "
                    "questions and finished all the requested tasks"
                )
            else:
                env_msg = f"Error in sending the GET request: {get_res['error']}"

            self._state.chat_history.append(ChatMessage(role="user", content=env_msg))
            return self._check_step_limit(env_msg)

        # ---- POST ----
        if action.action_type == ActionType.POST:
            if action.body is not None:
                self._state.post_requests.append(action.body)
                env_msg = (
                    "POST request accepted and executed successfully. "
                    "Please call FINISH if you have got answers for all the "
                    "questions and finished all the requested tasks"
                )
            else:
                env_msg = "Invalid POST request"

            self._state.chat_history.append(ChatMessage(role="user", content=env_msg))
            return self._check_step_limit(env_msg)

        # ---- Unknown action type ----
        self._state.task_status = TaskStatus.AGENT_INVALID_ACTION
        env_msg = "Invalid action type."
        self._state.chat_history.append(ChatMessage(role="user", content=env_msg))
        return self._make_obs(response_text=env_msg, done=True, reward=0.0)

    @property
    def state(self) -> MedAgentBenchState:
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reconstruct_raw(self, action: MedAgentBenchAction) -> str:
        """Reconstruct agent text from a structured action."""
        if action.action_type == ActionType.GET:
            return f"GET {action.url}"
        elif action.action_type == ActionType.POST:
            body_str = json.dumps(action.body) if action.body else "{}"
            return f"POST {action.url}\n{body_str}"
        elif action.action_type == ActionType.FINISH:
            return f"FINISH({json.dumps(action.answer)})"
        return ""

    def _check_step_limit(self, response_text: str) -> MedAgentBenchObservation:
        """Return observation, ending episode if step limit reached."""
        if self._state.step_count >= self._max_steps:
            self._state.task_status = TaskStatus.TASK_LIMIT_REACHED
            return self._make_obs(response_text=response_text, done=True, reward=0.0)
        return self._make_obs(response_text=response_text, done=False, reward=0.0)

    def _make_obs(
        self,
        response_text: str = "",
        done: bool = False,
        reward: float = 0.0,
        error: Optional[str] = None,
    ) -> MedAgentBenchObservation:
        task = self._state.task_sample
        return MedAgentBenchObservation(
            done=done,
            reward=reward,
            task_id=task.id if task else "",
            instruction=task.instruction if task else "",
            context=task.context if task else "",
            available_functions=self._functions if not done else [],
            response_text=response_text,
            error=error,
            task_status=self._state.task_status,
            step_number=self._state.step_count,
            max_steps=self._max_steps,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self) -> float:
        """Run the task-specific evaluation and return 1.0 or 0.0."""
        task = self._state.task_sample
        if task is None:
            return 0.0

        # Lazy-load the evaluation module
        if self._refsol is None:
            self._refsol = _load_eval_module()

        if self._refsol is None:
            # Can't evaluate without refsol — return 0 and log
            print("WARNING: refsol module not found, cannot evaluate. Returning 0.")
            return 0.0

        task_type = task.id.split("_")[0]  # e.g. "task1"
        grader_func = getattr(self._refsol, task_type, None)
        if grader_func is None:
            print(f"WARNING: No grader for {task_type}")
            return 0.0

        # Build a results-like object compatible with refsol expectations
        eval_results = _EvalResults(
            history=self._state.chat_history,
            result=json.dumps(self._state.agent_answer)
            if self._state.agent_answer is not None
            else None,
        )

        # Build case_data dict
        case_data = {
            "id": task.id,
            "instruction": task.instruction,
            "context": task.context,
            "sol": task.sol,
            "eval_MRN": task.eval_MRN,
        }

        try:
            if grader_func(case_data, eval_results, self._fhir_api_base) is True:
                return 1.0
        except Exception as e:
            print(f"Evaluation error for {task.id}: {e}")

        return 0.0


class _EvalResults:
    """Lightweight adapter matching the interface expected by refsol graders."""

    def __init__(self, history: List[ChatMessage], result: Any = None):
        self.history = [_ChatAdapter(m.role, m.content) for m in history]
        self.result = result


class _ChatAdapter:
    """Adapts ChatMessage to the attribute-access style refsol expects."""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
