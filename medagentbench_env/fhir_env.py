"""
MedAgentBench FHIR Training Environment.

Contains MedAgentTrainEnv and all supporting state/helpers so that
train.py is responsible only for model loading and the training loop.

Platform teams can import MedAgentTrainEnv directly to register it as
an SDK environment without pulling in TRL/training dependencies.
"""

import importlib
import importlib.util as _ilu
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

# Lazy: only needed when building the HuggingFace Dataset for training
try:
    from datasets import Dataset
except ImportError:
    Dataset = None  # type: ignore[misc]

# ---------------------------------------------------------------------------
# MockFHIR — loaded via importlib to avoid openenv __init__.py dependency
# ---------------------------------------------------------------------------

_server_dir = Path(__file__).resolve().parent / "server"
_spec = _ilu.spec_from_file_location("fhir_cache", _server_dir / "fhir_cache.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MockFHIR = _mod.MockFHIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent / "data"
_CACHE_PATH = _DATA_DIR / "fhir_cache.json"
_CACHE_GZ_PATH = _DATA_DIR / "fhir_cache.json.gz"

# Auto-decompress if only the .gz is present (e.g. freshly cloned repo)
if not _CACHE_PATH.exists() and _CACHE_GZ_PATH.exists():
    import gzip as _gzip
    print(f"Decompressing {_CACHE_GZ_PATH} → {_CACHE_PATH} ...")
    with _gzip.open(_CACHE_GZ_PATH, "rb") as _f_in, open(_CACHE_PATH, "wb") as _f_out:
        _f_out.write(_f_in.read())
    print("Done.")

_SYSTEM_PROMPT_PATH = _DATA_DIR / "new_system.txt"
_FHIR_API_BASE = "http://localhost:8080/fhir/"
_DEFAULT_HF_TOKEN = ""  # set HF_TOKEN env var instead

# ---------------------------------------------------------------------------
# Module-level shared state (lazy-initialised, mutated by main() at startup)
# ---------------------------------------------------------------------------

_MOCK_FHIR: Optional[MockFHIR] = None
_SYSTEM_PROMPT: str = ""
_TASKS: List[Dict] = []
_TASK_INDEX: int = 0
_TASKS_BY_ID: Dict[str, Dict] = {}
_TASKS_BY_INSTRUCTION: Dict[str, Dict] = {}
_SELECTED_TASK_TYPES: Optional[set] = None
_NEW_REFSOL = None

# Safeguard defaults — overridden by CLI args in main()
_MAX_TOOL_RESPONSE_CHARS = 4000
_MAX_TOOL_RESPONSE_ENTRIES = 24
_MAX_HISTORY_MESSAGES = 64  # includes the initial system item
_MAX_PROMPT_LENGTH = 8192
_MAX_STEPS = 6

# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

from medagentbench_env.verifier import evaluate as _verifier_evaluate  # noqa: E402

# ---------------------------------------------------------------------------
# History adapter (matches refsol ChatHistoryItem format)
# ---------------------------------------------------------------------------

class _HistoryItem:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------

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


def _get_new_refsol():
    """Load medagentbenchevals.new_refsol and patch its HTTP client with MockFHIR.

    This makes training use the same graders as eval (env_environment.py),
    eliminating inline grader duplication.
    """
    global _NEW_REFSOL
    if _NEW_REFSOL is not None:
        return _NEW_REFSOL
    src_dir = (
        Path(__file__).resolve().parent.parent
        / "medagentbenchv2" / "medagentbench_v2" / "src"
    )
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        new_refsol = importlib.import_module("medagentbenchevals.new_refsol")
        mock = _get_mock_fhir()
        new_refsol.send_get_request = (
            lambda url, params=None, headers=None, _m=mock: {
                "status_code": 200,
                "data": json.dumps(_m.get(url).get("data", {})),
            }
        )
        _NEW_REFSOL = new_refsol
        print("Loaded new_refsol graders (single source of truth).")
    except ImportError as e:
        print(f"Warning: could not load medagentbenchevals.new_refsol ({e}); falling back to inline grader.")
    return _NEW_REFSOL


def _norm_text(s: str) -> str:
    """Normalize whitespace for robust prompt/task matching."""
    return " ".join((s or "").split())


def _resolve_task_from_reset_kwargs(kwargs: Dict[str, Any]) -> Optional[Dict]:
    """Best-effort task lookup from GRPO environment reset kwargs."""
    for key in ("task_id", "id"):
        val = kwargs.get(key)
        if isinstance(val, str) and val in _TASKS_BY_ID:
            return _TASKS_BY_ID[val]

    candidate_texts: List[str] = []
    for v in kwargs.values():
        if isinstance(v, str):
            candidate_texts.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and isinstance(item.get("content"), str):
                    candidate_texts.append(item["content"])
                elif isinstance(item, str):
                    candidate_texts.append(item)
        elif isinstance(v, dict):
            content = v.get("content")
            if isinstance(content, str):
                candidate_texts.append(content)
            messages = v.get("messages")
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        candidate_texts.append(msg["content"])

    if not candidate_texts:
        return None

    for text in candidate_texts:
        norm = _norm_text(text)
        m = re.search(r"Task ID:\s*([A-Za-z0-9_]+)", text)
        if m:
            task_id = m.group(1)
            if task_id in _TASKS_BY_ID:
                return _TASKS_BY_ID[task_id]
        for instr_norm, task in _TASKS_BY_INSTRUCTION.items():
            if instr_norm and instr_norm in norm:
                return task

    raise RuntimeError(
        "Could not deterministically resolve task from reset kwargs. "
        "Aborting to prevent prompt/environment mismatch."
    )

# ---------------------------------------------------------------------------
# RL-worthy task types
# ---------------------------------------------------------------------------

# Excludes task3 (always-action HR average) which provides no decision signal.
_RL_TASK_TYPES = {
    "task1", "task2", "task4", "task5", "task6",
    "task7", "task8", "task9", "task10",
    "v2_task5", "v2_task9", "v2_task10",
}


def _get_tasks() -> List[Dict]:
    """Load all RL-worthy tasks from new_patient_tasks.json and test_data_v2.json."""
    global _TASKS, _TASKS_BY_ID, _TASKS_BY_INSTRUCTION
    if not _TASKS:
        with open(_DATA_DIR / "new_patient_tasks.json") as f:
            all_tasks: List[Dict] = json.load(f)

        v2_path = (
            Path(__file__).resolve().parent.parent
            / "medagentbenchv2" / "medagentbench_v2" / "src"
            / "MedAgentBench" / "data" / "medagentbench" / "test_data_v2.json"
        )
        if v2_path.exists():
            with open(v2_path) as f:
                v2_raw: List[Dict] = json.load(f)
            _V2_RL = {"task5", "task9", "task10"}
            for t in v2_raw:
                ttype = "_".join(t["id"].split("_")[:-1])
                if ttype in _V2_RL:
                    prefixed = dict(t)
                    prefixed["id"] = f"v2_{t['id']}"
                    all_tasks.append(prefixed)

        allowed_types = _SELECTED_TASK_TYPES if _SELECTED_TASK_TYPES is not None else _RL_TASK_TYPES
        _TASKS = [
            t for t in all_tasks
            if any(t["id"].startswith(f"{tt}_") for tt in allowed_types)
        ]
        _TASKS_BY_ID = {t["id"]: t for t in _TASKS}
        _TASKS_BY_INSTRUCTION = {
            _norm_text(str(t.get("instruction", ""))): t for t in _TASKS
        }
    return _TASKS


def build_dataset(data_dir: Path, num_tasks: Optional[int] = None) -> "Dataset":
    """Build training dataset from all RL-worthy MedAgentBench tasks."""
    tasks = _get_tasks()
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
# Training environment
# ---------------------------------------------------------------------------

class MedAgentTrainEnv:
    """Training environment exposing named FHIR tool calls.

    Mirrors the benchmark evaluation interface so training and evaluation
    use the same tool names and argument formats.

    GRPOTrainer's environment_factory creates one instance per rollout.
    """

    # Class-level registry — survives module reloads as long as the same
    # class object is used by both environment_factory and reward_func.
    # Unsloth's _calculate_rewards does not forward `environments` to
    # reward_func, so we track instances here and pop them in order.
    _registry: "List[MedAgentTrainEnv]" = []

    def _append_history(self, role: str, content: str) -> None:
        self._history.append(_HistoryItem(role, content))
        if len(self._history) > _MAX_HISTORY_MESSAGES:
            self._history = [self._history[0]] + self._history[-(_MAX_HISTORY_MESSAGES - 1):]

    def __init__(self):
        MedAgentTrainEnv._registry.append(self)
        self._mock = _get_mock_fhir()
        self._history: List[_HistoryItem] = []
        self._post_requests: List[Dict] = []
        self._agent_answer: Optional[List[Any]] = None
        self._step_count: int = 0
        self._max_steps: int = _MAX_STEPS
        self._task: Optional[Dict] = None
        self.reward: float = 0.0
        self.done: bool = False
        self._invalid_fhir_count: int = 0

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> str:
        global _TASK_INDEX
        tasks = _get_tasks()
        task = _resolve_task_from_reset_kwargs(kwargs)
        task_from_kwargs = task is not None
        if task is None:
            task_index = _TASK_INDEX % len(tasks)
            _TASK_INDEX += 1
            task = tasks[task_index]

        self._task = task
        self._history = []
        self._post_requests = []
        self._agent_answer = None
        self._step_count = 0
        self.reward = 0.0
        self.done = False
        self._invalid_fhir_count = 0

        context_str = f"\nContext: {self._task['context']}" if self._task.get("context") else ""
        instruction = f"{self._task['instruction']}{context_str}"
        self._append_history("user", _get_system_prompt())
        if task_from_kwargs:
            return "\nProceed with the provided task."
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
        payload = {"resourceType": resourceType, "status": status}
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
        self._append_history("agent", raw)
        self._append_history("user", "Task completed.")
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
        self._append_history("agent", f"GET {url}")

        result = self._mock.get(url)
        if "data" in result:
            data = result["data"]
            if isinstance(data, dict) and isinstance(data.get("entry"), list):
                entries = data.get("entry", [])
                if len(entries) > _MAX_TOOL_RESPONSE_ENTRIES:
                    data = dict(data)
                    data["entry"] = entries[:_MAX_TOOL_RESPONSE_ENTRIES]
                    data["returned_entry_count"] = len(data["entry"])
                    data["truncated_entry_count"] = max(0, len(entries) - len(data["entry"]))
            response_text = (
                json.dumps(data) if isinstance(data, (dict, list)) else str(data)
            )
            if len(response_text) > _MAX_TOOL_RESPONSE_CHARS:
                response_text = response_text[:_MAX_TOOL_RESPONSE_CHARS] + "\n... [truncated]"
            entry_count = len(data.get("entry", [])) if isinstance(data, dict) else "?"
            env_msg = (
                f"Here is the response from the GET request:\n{response_text}. "
                "Please call finish if you have got answers for all the questions "
                "and finished all the requested tasks"
            )
            trace_msg = f"GET {url} → {entry_count} entries"
        else:
            env_msg = f"Error in GET request: {result.get('error', 'Unknown error')}"
            trace_msg = env_msg
            self._invalid_fhir_count += 1

        self._append_history("user", trace_msg)

        if self._step_count >= self._max_steps:
            self.done = True
            self.reward = self._evaluate()

        return env_msg

    def _do_post(self, resource: str, payload: Dict) -> str:
        self._step_count += 1
        fhir_base = _FHIR_API_BASE.rstrip("/")
        url = f"{fhir_base}/{resource}"
        payload_str = json.dumps(payload)
        self._append_history("agent", f"POST {url}\n{payload_str}")
        self._post_requests.append(payload)

        env_msg = (
            "POST request accepted and executed successfully. "
            "Please call finish if you have got answers for all the questions "
            "and finished all the requested tasks"
        )
        self._append_history("user", env_msg)

        if self._step_count >= self._max_steps:
            self.done = True
            self.reward = self._evaluate()

        return env_msg

    def _print_trace(self) -> None:
        task_id = self._task["id"] if self._task else "unknown"
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"EPISODE TRACE  task={task_id}  steps={self._step_count}  reward={self.reward:.3f}")
        print(sep)
        for i, item in enumerate(self._history[1:], start=1):
            role_label = "AGENT" if item.role == "agent" else "ENV  "
            print(f"  [{i}] {role_label}: {item.content[:300]}")
        print(f"  ANSWER: {self._agent_answer}")
        print(sep)

    def _evaluate(self) -> float:
        if self._task is None:
            return 0.0
        history = [{"role": m.role, "content": m.content} for m in self._history]
        return _verifier_evaluate(
            history,
            self._task,
            _FHIR_API_BASE,
            invalid_fhir_count=self._invalid_fhir_count,
            new_refsol=_get_new_refsol(),
        )
