"""
Gymnasium environment for MedAgentBench — multi-task clinical RL.

Wraps FHIR tool execution and reward computation into a standard Gym
interface for multi-turn LLM RL training across all RL-worthy MedAgentBench
task types (task1, task2, task4–task10, v2_task5/9/10).

Each episode, the agent is given a clinical instruction (e.g. "Review this
patient's labs and take any necessary action") and must issue FHIR tool calls
to read the chart and optionally create orders, then call `finish`.  Rewards
are outcome-based: the official task evaluator determines terminal pass/fail,
supplemented by shaped rewards for correct decisions and penalties for
redundant or spurious actions.

Episode flow:
    obs, info = env.reset()           # sample a task from the pool
    for step in range(max_steps):
        action = policy.generate(obs) # LLM generates text + tool calls
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

Action format:
    A dict with the LLM's response:
    {
        "content": "assistant text (optional)",
        "tool_calls": [
            {"name": "fhir_observation_search", "arguments": {...}, "call_id": "..."},
            ...
        ]
    }

Observation format:
    A dict representing the current conversation state:
    {
        "messages": [...],          # OpenAI-format message history (truncated to max_history)
        "tool_schemas": [...],      # available FHIR tool JSON schemas
        "task": {...},              # current task dict (instruction, context, eval_MRN, …)
        "step": int,                # current step number
    }
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "medagentbench_v2" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_HERE.parent / "medagentbench_v2") not in sys.path:
    sys.path.insert(0, str(_HERE.parent / "medagentbench_v2"))

import src.tool.patient_search as _pat_search
import src.tool.observation_search as _obs_search
import src.tool.medication_request_search as _med_search
import src.tool.medication_request_create as _med_create
import src.tool.service_request_create as _svc_create
import src.tool.vitals_search as _vitals_search
import src.tool.vitals_create as _vitals_create
import src.tool.condition_search as _cond_search
import src.tool.procedure_search as _proc_search
import src.tool.calculator as _calc
import src.tool.finish as _finish
from src.tool.base import Tool

from .config import RLConfig, FHIRConfig, RewardConfig, get_fhir_url_map, POST_TOOLS
from .reward import (
    TaskRewardTracker,
    EpisodeRewardTracker,   # backward compat
    UnifiedEpisodeRewards,
    HistoryItem,
    fetch_ground_truth_state,
    fetch_ground_truth_qtc,  # backward compat
)
from .tool_parser import normalize_tool_args, ParsedToolCall
from .fhir_reset import reset_fhir_for_episode


# ─── Environment ────────────────────────────────────────────────────────────


class MedAgentEnv(gym.Env):
    """
    Multi-turn medical agent environment for MedAgentBench RL tasks.

    Each episode:
      1. A task case is selected (randomly or specified).
      2. The agent interacts via FHIR tool calls over multiple steps.
      3. Step rewards are given for correct FHIR interactions.
      4. Terminal reward is given based on the official task evaluator.

    Supports any RL-worthy task type (task1, task2, task4–task10).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        tasks: list[dict],
        config: RLConfig | None = None,
        task_ids: list[str] | None = None,
        auto_reset_fhir: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            tasks: List of task dicts from new_patient_tasks.json or test_data_v2.json.
            config: RL configuration. Uses defaults if None.
            task_ids: If provided, only use these task IDs for sampling.
            auto_reset_fhir: Whether to clean up FHIR state on reset.
            verbose: Print step-by-step info.
        """
        super().__init__()

        self.config = config or RLConfig()
        self.verbose = verbose
        self.auto_reset_fhir = auto_reset_fhir

        # Accept tasks filtered to RL task types (or unfiltered — caller decides)
        rl_prefixes = tuple(f"{t}_" for t in self.config.train.rl_task_types)
        self.all_tasks = {
            t["id"]: t for t in tasks
            if any(t["id"].startswith(p) for p in rl_prefixes)
        }
        if task_ids:
            self.task_pool = [tid for tid in task_ids if tid in self.all_tasks]
        else:
            self.task_pool = list(self.all_tasks.keys())

        if not self.task_pool:
            raise ValueError(
                f"No tasks found for RL types {self.config.train.rl_task_types}. "
                "Check tasks_path in PathConfig."
            )

        # ── Build FHIR tools ────────────────────────────────────────────
        fhir_base = self.config.fhir.api_base
        self.tools_list: list[Tool] = [
            _pat_search.create(fhir_base),
            _obs_search.create(fhir_base),
            _vitals_search.create(fhir_base),
            _vitals_create.create(fhir_base),
            _med_search.create(fhir_base),
            _med_create.create(fhir_base),
            _svc_create.create(fhir_base),
            _cond_search.create(fhir_base),
            _proc_search.create(fhir_base),
            _calc.create(),
            _finish.create(),
        ]
        self.tools_registry = {t.name: t for t in self.tools_list}
        self.tool_schemas = [self._to_chat_schema(t) for t in self.tools_list]

        self.fhir_url_map = get_fhir_url_map(fhir_base)

        # ── Gym spaces (text-based, using Dict) ────────────────────────
        # We use a simple Dict space; actual content is structured data
        self.observation_space = spaces.Dict({
            "messages": spaces.Sequence(spaces.Text(max_length=100000)),
            "step": spaces.Discrete(self.config.train.max_steps_per_episode + 1),
        })
        # Action is an LLM response dict (tool calls or text)
        self.action_space = spaces.Text(max_length=100000)

        # ── Episode state (initialised in reset) ───────────────────────
        self._current_task: Optional[dict] = None
        self._messages: list[dict] = []
        self._history: list[HistoryItem] = []
        self._reward_tracker: Optional[TaskRewardTracker] = None
        self._step_count = 0
        self._finished = False
        self._system_prompt = ""
        self._ground_truth: dict = {}

    @staticmethod
    def _to_chat_schema(tool: Tool) -> dict:
        """Convert tool schema to Chat Completions format (for vLLM/HF)."""
        raw = tool.json_schema()
        return {
            "type": "function",
            "function": {
                "name": raw["name"],
                "description": raw["description"],
                "parameters": raw.get("parameters", {}),
            },
        }

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        try:
            with open(self.config.paths.system_prompt_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "You are an expert medical AI agent."

    # ── Gym interface ────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """
        Reset the environment for a new episode.

        Options:
            task_id (str): Specific task ID to use (otherwise random sample)

        Returns:
            observation, info
        """
        super().reset(seed=seed)

        # Clean up FHIR state from previous episode
        if self.auto_reset_fhir:
            reset_fhir_for_episode(self.config.fhir, verbose=self.verbose)

        # Select task
        options = options or {}
        task_id = options.get("task_id")
        if task_id and task_id in self.all_tasks:
            self._current_task = self.all_tasks[task_id]
        else:
            task_id = random.choice(self.task_pool)
            self._current_task = self.all_tasks[task_id]

        # Derive task type (e.g. "task7" from "task7_3")
        task_type = "_".join(task_id.split("_")[:-1])
        mrn = self._current_task.get("eval_MRN", "")

        # Fetch ground-truth state for reward shaping
        self._ground_truth = fetch_ground_truth_state(
            task_type=task_type,
            mrn=mrn,
            fhir_api_base=self.config.fhir.api_base,
            cutoff_ts=self.config.fhir.cutoff_ts,
            qt_threshold=self.config.reward.qt_threshold,
            task=self._current_task,
        )

        # Initialise unified reward tracker
        self._reward_tracker = TaskRewardTracker(
            cfg=self.config.reward,
            task_type=task_type,
            ground_truth=self._ground_truth,
        )

        # Build initial messages
        self._system_prompt = self._load_system_prompt()
        user_content = f"<instruction>\n{self._current_task['instruction']}\n</instruction>"
        if self._current_task.get("context"):
            user_content += f"\n<context>\n{self._current_task['context']}\n</context>"

        self._messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        self._history = [HistoryItem(role="user", content=user_content)]
        self._step_count = 0
        self._finished = False

        obs = self._get_observation()
        info = {
            "task_id": self._current_task["id"],
            "task_type": task_type,
            "mrn": mrn,
            "ground_truth": self._ground_truth,
        }

        if self.verbose:
            print(
                f"\n{'='*60}\n"
                f"  [ENV] New episode: {self._current_task['id']}  "
                f"MRN={mrn}  should_act={self._ground_truth.get('should_act')}\n"
                f"{'='*60}"
            )

        return obs, info

    def step(
        self, action: dict
    ) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one agent turn.

        Args:
            action: Dict with the LLM's response:
                {
                    "content": str | None,         # assistant text
                    "tool_calls": [ParsedToolCall]  # list of tool calls
                }

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self._finished:
            raise RuntimeError("Episode already finished. Call reset().")

        self._step_count += 1
        step_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {"step": self._step_count}

        # ── Process assistant content ─────────────────────────────────
        assistant_content = action.get("content")
        if assistant_content:
            self._history.append(HistoryItem(role="agent", content=assistant_content))

        # Build the assistant message for the conversation history
        msg_dict: dict = {"role": "assistant", "content": assistant_content}

        tool_calls = action.get("tool_calls", [])

        if tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": getattr(tc, "call_id", "") or f"call_{self._step_count}_{i}",
                    "type": "function",
                    "function": {
                        "name": tc.name if isinstance(tc, ParsedToolCall) else tc.get("name", ""),
                        "arguments": json.dumps(
                            tc.arguments if isinstance(tc, ParsedToolCall)
                            else tc.get("arguments", {})
                        ),
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]
            if msg_dict["content"] is None:
                msg_dict["content"] = None

        self._messages.append(msg_dict)

        # ── Execute tool calls ────────────────────────────────────────
        if not tool_calls:
            # No tool calls → model stopped; truncate
            truncated = True
        else:
            for tc in tool_calls:
                if isinstance(tc, ParsedToolCall):
                    tc_name = tc.name
                    tc_args = tc.arguments
                    tc_id = tc.call_id or f"call_{self._step_count}"
                elif isinstance(tc, dict):
                    tc_name = tc.get("name", "")
                    tc_args = tc.get("arguments", {})
                    tc_id = tc.get("call_id", f"call_{self._step_count}")
                else:
                    continue

                is_post = tc_name in POST_TOOLS
                method, url = self.fhir_url_map.get(tc_name, ("UNKNOWN", ""))

                if self.verbose:
                    args_preview = json.dumps(tc_args)[:100]
                    print(f"  [step {self._step_count}] {tc_name}({args_preview}…)")

                # Handle finish tool
                if tc_name == "finish":
                    terminated = True
                    self._finished = True
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": "Finished.",
                    })

                    # Compute terminal reward
                    terminal_reward = self._reward_tracker.on_episode_end(
                        task=self._current_task,
                        history=self._history,
                        fhir_api_base=self.config.fhir.api_base,
                    )
                    step_reward += terminal_reward

                    info["terminal_pass"] = self._reward_tracker.rewards.terminal > 0
                    info["episode_rewards"] = self._reward_tracker.rewards.to_dict()
                    break

                # Record POST in history for extract_posts() compatibility
                if is_post:
                    post_content = f"POST {url}\n{json.dumps(tc_args)}"
                    self._history.append(HistoryItem(role="agent", content=post_content))

                # Execute the tool
                tool_obj = self.tools_registry.get(tc_name)
                if tool_obj is None:
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": f"Error: Unknown tool '{tc_name}'",
                    })
                    r = self._reward_tracker.on_tool_call(
                        tc_name, tc_args, is_post, url=url, tool_succeeded=False
                    )
                    step_reward += r
                    if is_post:
                        self._history.append(
                            HistoryItem(role="agent", content="POST request failed: unknown tool")
                        )
                    continue

                try:
                    normalized_args = normalize_tool_args(tool_obj, tc_args)
                    input_model = tool_obj.input_schema.model_validate(normalized_args)
                    tool_result = tool_obj(input_model)
                    result_str = (
                        json.dumps(tool_result)
                        if not isinstance(tool_result, str)
                        else tool_result
                    )

                    # Truncate very long FHIR responses to avoid
                    # blowing up the context window
                    _MAX_TOOL_RESPONSE_CHARS = 4000
                    if len(result_str) > _MAX_TOOL_RESPONSE_CHARS:
                        result_str = result_str[:_MAX_TOOL_RESPONSE_CHARS] + "\n... [truncated]"

                    if is_post:
                        self._history.append(
                            HistoryItem(role="agent", content="POST request accepted")
                        )

                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result_str,
                    })

                    # Compute step reward (pass URL for redundancy tracking)
                    r = self._reward_tracker.on_tool_call(
                        tc_name, tc_args, is_post, url=url, tool_succeeded=True
                    )
                    step_reward += r

                    if self.verbose:
                        preview = result_str[:120].replace("\n", " ")
                        print(f"  [tool←] {preview}…  (step_r={r:+.2f})")

                except Exception as exc:
                    err_msg = str(exc)
                    r = self._reward_tracker.on_tool_call(
                        tc_name, tc_args, is_post, url=url, tool_succeeded=False
                    )
                    step_reward += r

                    if is_post:
                        self._history.append(
                            HistoryItem(role="agent", content=f"POST request failed: {err_msg}")
                        )
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": f"Error: {err_msg}",
                    })

                    if self.verbose:
                        print(f"  [tool ERROR] {err_msg}  (step_r={r:+.2f})")

        # ── Check truncation (max steps) ──────────────────────────────
        if self._step_count >= self.config.train.max_steps_per_episode and not terminated:
            truncated = True

        # If truncated without finish, compute terminal reward anyway
        if truncated and not terminated:
            self._finished = True
            terminal_reward = self._reward_tracker.on_episode_end(
                task=self._current_task,
                history=self._history,
                fhir_api_base=self.config.fhir.api_base,
            )
            step_reward += terminal_reward
            info["terminal_pass"] = self._reward_tracker.rewards.terminal > 0
            info["episode_rewards"] = self._reward_tracker.rewards.to_dict()

        info["cumulative_reward"] = self._reward_tracker.get_total_reward()
        obs = self._get_observation()

        return obs, step_reward, terminated, truncated, info

    def _get_observation(self) -> dict:
        """Build the current observation dict.

        To prevent the conversation from growing too large and exceeding the
        model's context window, we keep at most ``max_history_messages``
        messages after the system prompt.  The system prompt (first message) is
        always preserved so the agent retains its instructions.
        """
        max_history = getattr(self, "_max_history_messages", 30)
        msgs = self._messages
        if len(msgs) > max_history + 1:
            # Keep system message (msgs[0]) + last `max_history` messages
            msgs = [msgs[0]] + msgs[-(max_history):]

        return {
            "messages": [m.copy() if isinstance(m, dict) else m for m in msgs],
            "tool_schemas": self.tool_schemas,
            "task": self._current_task,
            "step": self._step_count,
        }

    def get_episode_rewards(self) -> Optional[UnifiedEpisodeRewards]:
        """Get the detailed reward breakdown for the current/last episode."""
        if self._reward_tracker:
            return self._reward_tracker.rewards
        return None

    def render(self, mode: str = "human"):
        """Print the current conversation state."""
        if mode == "human":
            print(f"\n--- Episode Step {self._step_count} ---")
            for msg in self._messages[-3:]:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if content:
                    preview = content[:200].replace("\n", " ")
                    print(f"  [{role}] {preview}")

    def close(self):
        """Clean up resources."""
        pass


# ─── Factory function ───────────────────────────────────────────────────────


def make_env(
    config: RLConfig | None = None,
    task_ids: list[str] | None = None,
    task_types: list[str] | None = None,
    auto_reset_fhir: bool = True,
    verbose: bool = False,
) -> MedAgentEnv:
    """
    Create a MedAgentEnv from configuration.

    Loads task data from the configured path and filters to RL-worthy task types.

    Args:
        config: RL config. Uses defaults if None.
        task_ids: Explicit list of task IDs to include (overrides task_types filter).
        task_types: Task type prefixes to include (e.g. ["task5", "task7", "task10"]).
                    Defaults to config.train.rl_task_types if not specified.
        auto_reset_fhir: Whether to reset FHIR state on each episode.
        verbose: Print step-level debug output.
    """
    config = config or RLConfig()

    with open(config.paths.tasks_path, "r") as f:
        all_tasks = json.load(f)

    # Load v2 tasks (Mg/K+/A1c) with "v2_" prefix to avoid ID collision
    v2_path = config.paths.tasks_path_v2
    _V2_RL_TYPES = {"task5", "task9", "task10"}  # RL-worthy types in v2 file
    try:
        with open(v2_path, "r") as f:
            v2_raw = json.load(f)
        for t in v2_raw:
            ttype = "_".join(t["id"].split("_")[:-1])
            if ttype in _V2_RL_TYPES:
                prefixed = dict(t)
                prefixed["id"] = f"v2_{t['id']}"
                all_tasks.append(prefixed)
    except FileNotFoundError:
        pass  # v2 file optional

    # Filter to RL task types
    allowed = set(task_types or config.train.rl_task_types)
    rl_tasks = [
        t for t in all_tasks
        if any(t["id"].startswith(f"{tt}_") for tt in allowed)
    ]

    return MedAgentEnv(
        tasks=rl_tasks,
        config=config,
        task_ids=task_ids,
        auto_reset_fhir=auto_reset_fhir,
        verbose=verbose,
    )
