#!/usr/bin/env python3
"""
MedAgentBench RL Training Script.

Uses TRL's GRPOTrainer with named FHIR tool calls matching the benchmark
evaluation format (patient_search, fhir_observation_search, etc.) so the
model trains and evaluates on the same interface.

The environment talks directly to the local FHIR cache — no env server needed.

Usage:
    python train.py

    # Or on Northflank with OUTPUT_DIR set:
    python train.py --output-dir /output
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

# Lazy imports: datasets/trl only needed when actually training
try:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    Dataset = None
    GRPOConfig = None
    GRPOTrainer = None

# Import server modules directly via importlib (avoids openenv dependency in __init__.py)
import importlib.util as _ilu
_server_dir = Path(__file__).resolve().parent / "server"
_spec = _ilu.spec_from_file_location("fhir_cache", _server_dir / "fhir_cache.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MockFHIR = _mod.MockFHIR
_spec2 = _ilu.spec_from_file_location("reward", _server_dir / "reward.py")
_mod2 = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
compute_shaped_reward = _mod2.compute_shaped_reward


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent / "data"

_CACHE_PATH = _DATA_DIR / "fhir_cache.json"

_SYSTEM_PROMPT_PATH = _DATA_DIR / "new_system.txt"

_FHIR_API_BASE = "http://localhost:8080/fhir/"


# ---------------------------------------------------------------------------
# History adapter (matches refsol ChatHistoryItem format)
# ---------------------------------------------------------------------------

class _HistoryItem:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


# ---------------------------------------------------------------------------
# Training environment — named FHIR tool calls, no env server
# ---------------------------------------------------------------------------

# Module-level shared MockFHIR (loaded once, reused across episodes)
_MOCK_FHIR: Optional[MockFHIR] = None
_SYSTEM_PROMPT: str = ""
_TASKS: List[Dict] = []
_TASK_INDEX: int = 0

# Global env tracker — Unsloth's patched _calculate_rewards does not forward
# the `environments` kwarg to reward_func, so we track every created env in
# order and pop them in reward_func to match rewards to completions.
_ENV_REGISTRY: List["MedAgentTrainEnv"] = []


def _get_mock_fhir() -> MockFHIR:
    global _MOCK_FHIR
    if _MOCK_FHIR is None:
        if _CACHE_PATH.exists():
            _MOCK_FHIR = MockFHIR.from_cache(str(_CACHE_PATH), _FHIR_API_BASE)
        else:
            raise RuntimeError(
                f"FHIR cache not found at {_CACHE_PATH}. "
                "Build it first: python -m medagentbench_env.server.fhir_cache --build"
            )
    return _MOCK_FHIR


def _get_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if not _SYSTEM_PROMPT:
        if _SYSTEM_PROMPT_PATH.exists():
            _SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text().strip()
        else:
            _SYSTEM_PROMPT = (
                "You are an expert medical AI agent. "
                "Use the available FHIR tools to complete the clinical task. "
                "Always call finish when you are done."
            )
    return _SYSTEM_PROMPT


class MedAgentTrainEnv:
    """Training environment exposing named FHIR tool calls.

    Mirrors the benchmark evaluation interface so training and evaluation
    use the same tool names and argument formats.

    GRPOTrainer's environment_factory creates one instance per rollout.
    """

    def __init__(self):
        _ENV_REGISTRY.append(self)  # register for reward_func fallback
        self._mock = _get_mock_fhir()
        self._history: List[_HistoryItem] = []
        self._post_requests: List[Dict] = []
        self._agent_answer: Optional[List[Any]] = None
        self._step_count: int = 0
        self._max_steps: int = 8
        self._task: Optional[Dict] = None
        self.reward: float = 0.0
        self.done: bool = False

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> str:
        """Start a new episode. Returns the task instruction."""
        global _TASK_INDEX
        tasks = _get_tasks()
        task_index = _TASK_INDEX % len(tasks)
        _TASK_INDEX += 1

        self._task = tasks[task_index]
        self._history = []
        self._post_requests = []
        self._agent_answer = None
        self._step_count = 0
        self.reward = 0.0
        self.done = False

        context_str = f"\nContext: {self._task['context']}" if self._task.get("context") else ""
        instruction = f"{self._task['instruction']}{context_str}"

        # Record system turn in history for refsol evaluation
        self._history.append(_HistoryItem("user", _get_system_prompt()))
        return instruction

    # ------------------------------------------------------------------
    # GET tools
    # ------------------------------------------------------------------

    def fhir_patient_search(
        self,
        family: str = "",
        given: str = "",
        birthdate: str = "",
        identifier: str = "",
    ) -> str:
        """Search for patients in the FHIR EHR.

        Args:
            family: Patient family (last) name.
            given: Patient given (first) name.
            birthdate: Date of birth in YYYY-MM-DD format.
            identifier: Patient MRN or other identifier.

        Returns:
            JSON FHIR Bundle of matching patients.
        """
        if self.done:
            return "Episode already finished."
        params: Dict[str, str] = {}
        if family:
            params["family"] = family
        if given:
            params["given"] = given
        if birthdate:
            params["birthdate"] = birthdate
        if identifier:
            params["identifier"] = identifier
        return self._do_get("Patient", params)

    def fhir_observation_search(
        self,
        patient: str = "",
        code: str = "",
        explanation: str = "",
    ) -> str:
        """Search for clinical observations (labs, vitals) by code.

        Args:
            patient: Patient MRN / identifier.
            code: LOINC or local code to search for (e.g. 'A1C', '4548-4').
            explanation: Optional explanation of why this search is needed.

        Returns:
            JSON FHIR Bundle of Observation resources.
        """
        if self.done:
            return "Episode already finished."
        params: Dict[str, str] = {"_sort": "-date", "_count": "5000"}
        if patient:
            params["patient"] = patient
        if code:
            params["code"] = code
        return self._do_get("Observation", params)

    def fhir_vitals_search(
        self,
        patient: str = "",
        category: str = "vital-signs",
        date: str = "",
    ) -> str:
        """Search for vital signs observations.

        Args:
            patient: Patient MRN / identifier.
            category: Observation category (default 'vital-signs').
            date: Date filter in YYYY-MM-DD format.

        Returns:
            JSON FHIR Bundle of vital sign Observations.
        """
        if self.done:
            return "Episode already finished."
        params: Dict[str, str] = {"category": category}
        if patient:
            params["patient"] = patient
        if date:
            params["date"] = date
        return self._do_get("Observation", params)

    def fhir_condition_search(self, patient: str = "", category: str = "") -> str:
        """Search for patient conditions / diagnoses.

        Args:
            patient: Patient MRN / identifier.
            category: Condition category (e.g. 'problem-list-item').

        Returns:
            JSON FHIR Bundle of Condition resources.
        """
        if self.done:
            return "Episode already finished."
        params: Dict[str, str] = {}
        if patient:
            params["patient"] = patient
        if category:
            params["category"] = category
        return self._do_get("Condition", params)

    def fhir_procedure_search(self, patient: str = "", date: str = "") -> str:
        """Search for procedures performed on a patient.

        Args:
            patient: Patient MRN / identifier.
            date: Date filter in YYYY-MM-DD format.

        Returns:
            JSON FHIR Bundle of Procedure resources.
        """
        if self.done:
            return "Episode already finished."
        params: Dict[str, str] = {}
        if patient:
            params["patient"] = patient
        if date:
            params["date"] = date
        return self._do_get("Procedure", params)

    def fhir_medication_request_search(
        self, patient: str = "", status: str = ""
    ) -> str:
        """Search for medication orders for a patient.

        Args:
            patient: Patient MRN / identifier.
            status: Request status filter (e.g. 'active').

        Returns:
            JSON FHIR Bundle of MedicationRequest resources.
        """
        if self.done:
            return "Episode already finished."
        params: Dict[str, str] = {}
        if patient:
            params["patient"] = patient
        if status:
            params["status"] = status
        return self._do_get("MedicationRequest", params)

    # ------------------------------------------------------------------
    # POST tools
    # ------------------------------------------------------------------

    def fhir_vitals_create(
        self,
        resourceType: str = "Observation",
        category: Optional[List] = None,
        code: Optional[Dict] = None,
        effectiveDateTime: str = "",
        status: str = "final",
        valueString: str = "",
        subject: Optional[Dict] = None,
    ) -> str:
        """Record a vital signs observation in the FHIR EHR.

        Args:
            resourceType: Must be 'Observation'.
            category: FHIR category coding list.
            code: FHIR code element with text/coding.
            effectiveDateTime: ISO datetime of the measurement.
            status: Observation status (default 'final').
            valueString: The vital sign value as a string.
            subject: Patient reference dict, e.g. {'reference': 'Patient/MRN'}.

        Returns:
            Confirmation message.
        """
        if self.done:
            return "Episode already finished."
        payload = {
            "resourceType": resourceType,
            "status": status,
        }
        if category is not None:
            payload["category"] = category
        if code is not None:
            payload["code"] = code
        if effectiveDateTime:
            payload["effectiveDateTime"] = effectiveDateTime
        if valueString:
            payload["valueString"] = valueString
        if subject is not None:
            payload["subject"] = subject
        return self._do_post("Observation", payload)

    def fhir_service_request_create(
        self,
        resourceType: str = "ServiceRequest",
        code: Optional[Dict] = None,
        authoredOn: str = "",
        status: str = "active",
        intent: str = "order",
        priority: str = "stat",
        subject: Optional[Dict] = None,
        note: Optional[Any] = None,
        occurrenceDateTime: str = "",
    ) -> str:
        """Create a service request (referral, order) in the FHIR EHR.

        Args:
            resourceType: Must be 'ServiceRequest'.
            code: FHIR code element with coding list.
            authoredOn: ISO datetime the order was written.
            status: Request status (default 'active').
            intent: Request intent (default 'order').
            priority: Priority (default 'stat').
            subject: Patient reference dict.
            note: Clinical notes as string, dict, or list.
            occurrenceDateTime: When the service should occur.

        Returns:
            Confirmation message.
        """
        if self.done:
            return "Episode already finished."
        payload: Dict[str, Any] = {
            "resourceType": resourceType,
            "status": status,
            "intent": intent,
            "priority": priority,
        }
        if code is not None:
            payload["code"] = code
        if authoredOn:
            payload["authoredOn"] = authoredOn
        if subject is not None:
            payload["subject"] = subject
        if note is not None:
            payload["note"] = note
        if occurrenceDateTime:
            payload["occurrenceDateTime"] = occurrenceDateTime
        return self._do_post("ServiceRequest", payload)

    def fhir_medication_request_create(
        self,
        resourceType: str = "MedicationRequest",
        medicationCodeableConcept: Optional[Dict] = None,
        subject: Optional[Dict] = None,
        status: str = "active",
        intent: str = "order",
        authoredOn: str = "",
        dosageInstruction: Optional[List] = None,
        note: Optional[Any] = None,
    ) -> str:
        """Create a medication order in the FHIR EHR.

        Args:
            resourceType: Must be 'MedicationRequest'.
            medicationCodeableConcept: Medication coding.
            subject: Patient reference dict.
            status: Request status (default 'active').
            intent: Request intent (default 'order').
            authoredOn: ISO datetime the order was written.
            dosageInstruction: List of dosage instruction dicts.
            note: Clinical notes.

        Returns:
            Confirmation message.
        """
        if self.done:
            return "Episode already finished."
        payload: Dict[str, Any] = {
            "resourceType": resourceType,
            "status": status,
            "intent": intent,
        }
        if medicationCodeableConcept is not None:
            payload["medicationCodeableConcept"] = medicationCodeableConcept
        if subject is not None:
            payload["subject"] = subject
        if authoredOn:
            payload["authoredOn"] = authoredOn
        if dosageInstruction is not None:
            payload["dosageInstruction"] = dosageInstruction
        if note is not None:
            payload["note"] = note
        return self._do_post("MedicationRequest", payload)

    # ------------------------------------------------------------------
    # Utility tools
    # ------------------------------------------------------------------

    def calculator(self, expression: str) -> str:
        """Evaluate a mathematical expression safely.

        Args:
            expression: Python math expression, e.g. '(120 + 80) / 2'.

        Returns:
            The numeric result as a string.
        """
        safe_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_names["abs"] = abs
        safe_names["round"] = round
        try:
            result = eval(expression, {"__builtins__": {}}, safe_names)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Calculator error: {e}"

    def finish(self, value: List[Any]) -> str:
        """Signal task completion and provide the final answer.

        Args:
            value: List of answer values, e.g. ['S6534835'] or [10] or [].

        Returns:
            Completion confirmation with reward.
        """
        if self.done:
            return "Episode already finished."

        self._agent_answer = value if isinstance(value, list) else [value]
        raw = f"FINISH({json.dumps(self._agent_answer)})"
        self._history.append(_HistoryItem("agent", raw))
        self._history.append(_HistoryItem("user", "Task completed."))
        self._step_count += 1
        self.done = True
        self.reward = self._evaluate()
        self._print_trace()
        return f"Task completed. Reward: {self.reward:.3f}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_get(self, resource: str, params: Dict[str, str]) -> str:
        self._step_count += 1
        fhir_base = _FHIR_API_BASE.rstrip("/")
        param_str = urlencode(sorted(params.items()))
        url = f"{fhir_base}/{resource}?{param_str}&_format=json" if param_str else f"{fhir_base}/{resource}?_format=json"

        self._history.append(_HistoryItem("agent", f"GET {url}"))

        result = self._mock.get(url)
        if "data" in result:
            data = result["data"]
            response_text = (
                json.dumps(data) if isinstance(data, (dict, list)) else str(data)
            )
            entry_count = len(data.get("entry", [])) if isinstance(data, dict) else "?"
            env_msg = (
                f"Here is the response from the GET request:\n{response_text}. "
                "Please call finish if you have got answers for all the questions "
                "and finished all the requested tasks"
            )
            # Compact trace entry — full bundle is returned to model, but trace shows summary
            trace_msg = f"GET {url} → {entry_count} entries"
        else:
            env_msg = f"Error in GET request: {result.get('error', 'Unknown error')}"
            trace_msg = env_msg

        self._history.append(_HistoryItem("user", trace_msg))

        if self._step_count >= self._max_steps:
            self.done = True
            self.reward = 0.0

        return env_msg

    def _do_post(self, resource: str, payload: Dict) -> str:
        self._step_count += 1
        fhir_base = _FHIR_API_BASE.rstrip("/")
        url = f"{fhir_base}/{resource}"
        payload_str = json.dumps(payload)

        self._history.append(_HistoryItem("agent", f"POST {url}\n{payload_str}"))
        self._post_requests.append(payload)

        env_msg = (
            "POST request accepted and executed successfully. "
            "Please call finish if you have got answers for all the questions "
            "and finished all the requested tasks"
        )
        self._history.append(_HistoryItem("user", env_msg))

        if self._step_count >= self._max_steps:
            self.done = True
            self.reward = 0.0

        return env_msg

    def _print_trace(self) -> None:
        """Print a readable episode trace to stdout."""
        task_id = self._task["id"] if self._task else "unknown"
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"EPISODE TRACE  task={task_id}  steps={self._step_count}  reward={self.reward:.3f}")
        print(sep)
        # Skip index 0 (system prompt — too long to print)
        for i, item in enumerate(self._history[1:], start=1):
            role_label = "AGENT" if item.role == "agent" else "ENV  "
            print(f"  [{i}] {role_label}: {item.content[:300]}")
        print(f"  ANSWER: {self._agent_answer}")
        print(sep)

    def _evaluate(self) -> float:
        if self._task is None:
            return 0.0

        task_type = self._task["id"].split("_")[0]
        case_data = {
            "id": self._task["id"],
            "instruction": self._task["instruction"],
            "context": self._task.get("context", ""),
            "sol": self._task.get("sol", []),
            "eval_MRN": self._task.get("eval_MRN", ""),
        }
        benchmark_type = self._task.get("_benchmark_type", "")

        return compute_shaped_reward(
            task_type=task_type,
            case_data=case_data,
            history=self._history,
            agent_answer=self._agent_answer,
            fhir_api_base=_FHIR_API_BASE,
            step_count=self._step_count,
            max_steps=self._max_steps,
            refsol_pass=False,  # refsol not run during training (no live server)
            benchmark_type=benchmark_type,
        )


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def reward_func(completions, environments=None, **kwargs):
    """Return shaped reward from each episode's environment.

    Standard TRL passes `environments` directly. Unsloth's patched
    _calculate_rewards does not, so we fall back to the global registry
    which tracks every MedAgentTrainEnv in creation order.
    """
    if environments is None:
        environments = kwargs.get("environments")

    if environments is not None:
        return [float(env.reward) for env in environments]

    # Unsloth fallback: pop the oldest N envs from the registry
    n = len(completions)
    envs = _ENV_REGISTRY[:n]
    del _ENV_REGISTRY[:n]
    return [float(env.reward) for env in envs]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _get_tasks() -> List[Dict]:
    global _TASKS
    if not _TASKS:
        data_file = _DATA_DIR / "stratified_benchmark.json"
        with open(data_file) as f:
            _TASKS = json.load(f)
    return _TASKS


def build_dataset(data_dir: Path, num_tasks: Optional[int] = None) -> Dataset:
    """Build training dataset from MedAgentBench stratified benchmark."""
    data_file = data_dir / "stratified_benchmark.json"
    with open(data_file) as f:
        tasks = json.load(f)

    if num_tasks is not None:
        tasks = tasks[:num_tasks]

    system_prompt = _get_system_prompt()

    prompts = []
    for task in tasks:
        context_str = f"\nContext: {task['context']}" if task.get("context") else ""
        user_msg = f"{task['instruction']}{context_str}"
        prompts.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ])

    return Dataset.from_dict({"prompt": prompts})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train on MedAgentBench with GRPO")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-1.7B",
        help="Model name or path",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(_DATA_DIR),
        help="Path to directory containing stratified_benchmark.json",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=None,
        help="Number of tasks to use (default: all 90)",
    )
    parser.add_argument(
        "--max-completion-length", type=int, default=2048,
        help="Max tokens per generation",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("OUTPUT_DIR", "./output"),
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--num-train-epochs", type=int, default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per-device-batch-size", type=int, default=4,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push the final model to HuggingFace Hub after training",
    )
    parser.add_argument(
        "--hub-model-id", type=str, default=None,
        help="HuggingFace repo to push to, e.g. 'username/medagent-qwen3'",
    )
    parser.add_argument(
        "--hub-token", type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    # Pre-load shared resources
    _get_mock_fhir()
    print(f"Loaded FHIR cache from {_CACHE_PATH}")

    dataset = build_dataset(Path(args.data_dir), args.num_tasks)
    print(f"Training dataset: {len(dataset)} tasks")

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        log_completions=True,
        num_completions_to_print=2,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        environment_factory=MedAgentTrainEnv,
        args=grpo_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")

    if args.push_to_hub:
        if not args.hub_model_id:
            # Default repo name: username inferred from token
            model_basename = args.model.split("/")[-1]
            args.hub_model_id = f"medagent-{model_basename}"
            print(f"No --hub-model-id given, using: {args.hub_model_id}")
        print(f"Pushing model to HuggingFace Hub: {args.hub_model_id} ...")
        trainer.push_to_hub(
            repo_id=args.hub_model_id,
            token=args.hub_token,
            private=False,
        )
        print(f"Model pushed to https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
