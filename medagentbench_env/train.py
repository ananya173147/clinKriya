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
import concurrent.futures
import csv
import json
import math
import os
import re
import sys
import types as _types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
_TASKS_BY_ID: Dict[str, Dict] = {}
_TASKS_BY_INSTRUCTION: Dict[str, Dict] = {}
_SELECTED_TASK_TYPES: Optional[set[str]] = None

# Hybrid safeguards inspired by the custom rl_training path
_MAX_TOOL_RESPONSE_CHARS = 4000
_MAX_TOOL_RESPONSE_ENTRIES = 24
_MAX_HISTORY_MESSAGES = 64  # includes the initial system item
_MAX_PROMPT_LENGTH = 8192
_MAX_STEPS = 6
_DEFAULT_HF_TOKEN = ""  # set HF_TOKEN env var instead

# new_refsol module (lazy-loaded once, shared across all episodes)
_NEW_REFSOL = None

# Reward weights and task tables now live in verifier.py — import for
# backward-compat references that remain in this file.
from medagentbench_env.verifier import (  # noqa: E402
    RewardWeights as _RewardWeights,
    ANTICOAG_MEDS as _ANTICOAG_MEDS,
    QT_PROLONGING_MEDS as _TASK7_QT_PROLONGING_MEDS,
    NON_ACTIVE_STATUSES as _TASK_NON_ACTIVE_STATUSES,
    evaluate as _verifier_evaluate,
)

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

    This makes train.py use the same graders as eval (env_environment.py),
    eliminating the inline _compute_refsol_pass() duplication.
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
        import importlib
        new_refsol = importlib.import_module("medagentbenchevals.new_refsol")
        mock = _get_mock_fhir()
        # new_refsol uses `from .utils import *` so patching utils module has no effect.
        # Patch the reference in new_refsol's own namespace directly.
        # MockFHIR.get() returns {"status_code": <int>, "data": <dict>};
        # new_refsol calls json.loads(send_get_request(url)["data"]) so we must
        # serialize the dict back to a JSON string.
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
    """Best-effort task lookup from GRPO environment reset kwargs.

    Newer TRL versions may pass different keys/shapes, so this function accepts:
    - direct task IDs (e.g. task_id/id)
    - prompt/messages payloads with user text
    and maps them back to the canonical task dict.
    """
    # 1) Direct task-id style fields
    for key in ("task_id", "id"):
        val = kwargs.get(key)
        if isinstance(val, str) and val in _TASKS_BY_ID:
            return _TASKS_BY_ID[val]

    # 2) Scan all string-like sources for an instruction match
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

        # Prefer explicit "Task ID: ..." prefix when present.
        m = re.search(r"Task ID:\s*([A-Za-z0-9_]+)", text)
        if m:
            task_id = m.group(1)
            if task_id in _TASKS_BY_ID:
                return _TASKS_BY_ID[task_id]

        # Otherwise, match by embedded instruction text.
        for instr_norm, task in _TASKS_BY_INSTRUCTION.items():
            if instr_norm and instr_norm in norm:
                return task

    # Deterministic task binding: if prompt-like text exists but no match,
    # fail fast to avoid silent prompt/env desynchronization.
    raise RuntimeError(
        "Could not deterministically resolve task from reset kwargs. "
        "Aborting to prevent prompt/environment mismatch."
    )

    return None


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
        """Append history and cap size to avoid unbounded context growth."""
        self._history.append(_HistoryItem(role, content))
        if len(self._history) > _MAX_HISTORY_MESSAGES:
            # Preserve earliest system message + latest interactions.
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
        """Start a new episode. Returns the task instruction."""
        global _TASK_INDEX
        tasks = _get_tasks()

        # Align episode task with the prompt/task passed by TRL when available.
        # This prevents prompt/env desynchronization that can mix instructions.
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

        # Record system turn in history for refsol evaluation
        self._append_history("user", _get_system_prompt())
        # If task came from the prompt kwargs, avoid echoing the same instruction
        # again in the first environment observation.
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
                response_text = (
                    response_text[:_MAX_TOOL_RESPONSE_CHARS]
                    + "\n... [truncated]"
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
            self._invalid_fhir_count += 1

        self._append_history("user", trace_msg)

        if self._step_count >= self._max_steps:
            self.done = True
            # Compute reward even when the model never calls finish().
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
            # Compute reward even when the model never calls finish().
            self.reward = self._evaluate()

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

    # ------------------------------------------------------------------
    # MockFHIR helpers
    # ------------------------------------------------------------------

    def _mock_get_entries(self, resource: str, params: str) -> List[Dict]:
        """Query MockFHIR and return parsed FHIR entry resources."""
        fhir_base = _FHIR_API_BASE.rstrip("/")
        url = f"{fhir_base}/{resource}?{params}&_count=1000&_format=json"
        resp = self._mock.get(url)
        data = resp.get("data", {})
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return []
        return [e.get("resource", {}) for e in data.get("entry", [])]

    @staticmethod
    def _resource_date(r: Dict) -> Optional[str]:
        """Return the best-effort date string from a FHIR resource."""
        for field in ("effectiveDateTime", "performedDateTime", "authoredOn",
                      "recordedDate", "occurrenceDateTime", "date"):
            v = r.get(field)
            if v:
                return v
        period = r.get("performedPeriod") or r.get("occurrencePeriod", {})
        if isinstance(period, dict):
            return period.get("start") or period.get("end")
        return r.get("meta", {}).get("lastUpdated")

    # ------------------------------------------------------------------
    # Ground-truth + refsol
    # ------------------------------------------------------------------

    def _compute_refsol_pass(self, task_type: str, mrn: str) -> bool:
        """
        Offline refsol check using MockFHIR cache — no live server needed.
        Returns True if the agent's actions satisfy the task's acceptance criteria.
        """
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz

        _CUTOFF = "2025-01-01"
        _CUTOFF_DT = _dt.fromisoformat("2025-01-01T00:00:00+00:00")
        patient_ref = f"Patient/{mrn}"

        # ── Parse accepted POSTs ─────────────────────────────────────
        posts: List[Tuple[str, Dict]] = []
        for idx, msg in enumerate(self._history):
            if msg.role == "agent" and msg.content.startswith("POST"):
                if idx + 1 < len(self._history) and "POST request accepted" in self._history[idx + 1].content:
                    try:
                        lines = msg.content.split("\n", 1)
                        url_part = lines[0][4:].strip()
                        payload = json.loads(lines[1]) if len(lines) > 1 else {}
                        posts.append((url_part, payload))
                    except Exception:
                        pass

        def _has_ref(pl: Dict) -> bool:
            subj = pl.get("subject", {})
            if isinstance(subj, list):
                subj = subj[0] if subj else {}
            return subj.get("reference") == patient_ref

        def _parse_dt(s: Optional[str]) -> Optional[_dt]:
            if not s:
                return None
            try:
                d = _dt.fromisoformat(s)
                if d.tzinfo is None:
                    d = d.replace(tzinfo=_tz.utc)
                return d
            except ValueError:
                return None

        # ── task1: CT Abdomen > 12 months ago → order new CT ────────
        if task_type == "task1":
            resources = self._mock_get_entries("Procedure", f"patient={mrn}&code=IMGCT0491,IMGIL0001")
            valid = [r for r in resources if (self._resource_date(r) or "") < _CUTOFF]
            if valid:
                latest_date = _parse_dt(max(self._resource_date(r) or "" for r in valid))
            else:
                latest_date = None
            should_act = latest_date is None or (_CUTOFF_DT - latest_date) > _td(days=365)
            if not should_act:
                return len(posts) == 0
            # Expect ≥1 POST: ServiceRequest with CT code
            return any(
                _has_ref(pl) and pl.get("resourceType") == "ServiceRequest"
                for _, pl in posts
            )

        # ── task2: DVT anticoag count ≠ 1 → fix order ───────────────
        elif task_type == "task2":
            _ANTICOAG = ("heparin", "enoxaparin", "lovenox", "fondaparinux",
                         "rivaroxaban", "apixaban", "dabigatran", "warfarin",
                         "tinzaparin", "dalteparin")
            resources = self._mock_get_entries("MedicationRequest", f"patient={mrn}&status=active")
            active = [r for r in resources
                      if (self._resource_date(r) or "") < _CUTOFF
                      and any(k in json.dumps(r).lower() for k in _ANTICOAG)]
            should_act = len(active) != 1
            if not should_act:
                return len(posts) == 0
            return any(
                _has_ref(pl) and pl.get("resourceType") == "MedicationRequest"
                for _, pl in posts
            )

        # ── task4: urinary catheter > 48h → remove ──────────────────
        elif task_type == "task4":
            # Parse current time from task context
            current_time = _CUTOFF_DT
            ctx = self._task.get("context", "") if self._task else ""
            m = re.search(r"It'?s\s+([\d\-T:+Z]+)\s+now", ctx)
            if m:
                current_time = _parse_dt(m.group(1)) or _CUTOFF_DT

            resources = self._mock_get_entries("Procedure", f"patient={mrn}&code=NUR1373")
            relevant = []
            for r in resources:
                d = _parse_dt(self._resource_date(r))
                if d and d <= current_time:
                    relevant.append((d, r))
            if not relevant:
                return len(posts) == 0
            latest_date, latest_r = max(relevant, key=lambda x: x[0])
            status = (latest_r.get("status") or "").lower()
            if status in ("stopped", "completed", "cancelled", "entered-in-error"):
                return len(posts) == 0
            age_h = (current_time - latest_date).total_seconds() / 3600
            should_act = age_h > 48
            if not should_act:
                return len(posts) == 0
            return any(_has_ref(pl) for _, pl in posts)

        # ── task5: C64.2 + CT > 3 months ago → order CT ─────────────
        elif task_type == "task5":
            conditions = self._mock_get_entries("Condition", f"patient={mrn}&code=C64.2")
            has_c64 = any((self._resource_date(r) or "") < _CUTOFF for r in conditions)
            if not has_c64:
                return len(posts) == 0
            ct_resources = self._mock_get_entries(
                "Procedure", f"patient={mrn}&code=IMGCT0491,IMGIL0001,74177"
            )
            valid_cts = [r for r in ct_resources if (self._resource_date(r) or "") < _CUTOFF]
            if valid_cts:
                latest_ct = _parse_dt(max(self._resource_date(r) or "" for r in valid_cts))
            else:
                latest_ct = None
            should_act = latest_ct is None or (_CUTOFF_DT - latest_ct) > _td(days=90)
            if not should_act:
                return len(posts) == 0
            return any(
                _has_ref(pl) and pl.get("resourceType") in {"ServiceRequest", "MedicationRequest"}
                for _, pl in posts
            )

        # ── task6: TSH > 10 twice ≥ 1 month apart → thyroid rx ──────
        elif task_type == "task6":
            resources = self._mock_get_entries("Observation", f"patient={mrn}&code=TSH")
            elevated_dates: List[_dt] = []
            for r in resources:
                if (self._resource_date(r) or "") >= _CUTOFF:
                    continue
                try:
                    val = float(r.get("valueQuantity", {}).get("value", "nan"))
                    d = _parse_dt(self._resource_date(r))
                    if val > 10 and d:
                        elevated_dates.append(d)
                except (TypeError, ValueError):
                    pass
            elevated_dates.sort()
            should_act = False
            for i in range(len(elevated_dates)):
                for j in range(i + 1, len(elevated_dates)):
                    if (elevated_dates[j] - elevated_dates[i]).days >= 30:
                        should_act = True
                        break
                if should_act:
                    break
            if not should_act:
                return len(posts) == 0
            return any(
                _has_ref(pl) and pl.get("resourceType") == "MedicationRequest"
                for _, pl in posts
            )

        # ── task7: QTc > 500 → stop QT drug + ECG order ─────────────
        elif task_type == "task7":
            resources = self._mock_get_entries("Observation", f"patient={mrn}&code=QTCINTERVAL")
            valid = [r for r in resources if (self._resource_date(r) or "") < _CUTOFF]
            if not valid:
                return False
            latest = max(valid, key=lambda r: self._resource_date(r) or "")
            try:
                qtc_val = float(latest["valueQuantity"]["value"])
            except (KeyError, TypeError, ValueError):
                return False

            if qtc_val <= 500:
                return len(posts) == 0

            _QT_MEDS = {"ondansetron", "prochlorperazine", "haloperidol", "quetiapine",
                        "olanzapine", "risperidone", "ziprasidone", "clozapine", "chlorpromazine"}
            _NON_ACTIVE = {"stopped", "cancelled", "completed", "entered-in-error"}
            found_ecg = any(
                _has_ref(pl) and pl.get("resourceType") == "ServiceRequest"
                and pl.get("status", "").lower() == "active"
                and ("445118002" in str(pl).lower() or "ecg" in str(pl).lower()
                     or "electrocardiogram" in str(pl).lower())
                for _, pl in posts
            )
            found_stop = any(
                _has_ref(pl) and pl.get("resourceType") == "MedicationRequest"
                and pl.get("status", "").lower() in _NON_ACTIVE
                and any(w in str(pl).lower() for w in _QT_MEDS)
                for _, pl in posts
            )
            return found_ecg and found_stop

        # ── task8: active opioid + no naloxone → add naloxone ────────
        elif task_type == "task8":
            _OPIOIDS = ("hydromorphone", "oxycodone", "fentanyl", "hydrocodone", "morphine")
            resources = self._mock_get_entries("MedicationRequest", f"patient={mrn}&status=active")
            active_opioids = [
                r for r in resources
                if (self._resource_date(r) or "") < _CUTOFF
                and any(k in json.dumps(r).lower() for k in _OPIOIDS)
            ]
            has_naloxone = any(
                "naloxone" in json.dumps(r).lower() for r in resources
                if (self._resource_date(r) or "") < _CUTOFF
            )
            should_act = len(active_opioids) > 0 and not has_naloxone
            if not should_act:
                return len(posts) == 0
            return any(
                _has_ref(pl) and pl.get("resourceType") == "MedicationRequest"
                and "naloxone" in json.dumps(pl).lower()
                for _, pl in posts
            )

        # ── task9: flu vaccine > 365 days → order vaccine ───────────
        elif task_type == "task9":
            current_time_str = _CUTOFF
            ctx = self._task.get("context", "") if self._task else ""
            m = re.search(r"It'?s\s+([\d\-T:+Z]+)\s+now", ctx)
            if m:
                current_time_str = m.group(1)
            current_time = _parse_dt(current_time_str) or _CUTOFF_DT

            resources = self._mock_get_entries("Procedure", f"patient={mrn}&code=90686")
            valid = [r for r in resources if (_parse_dt(self._resource_date(r)) or _CUTOFF_DT) <= current_time]
            if valid:
                latest = max((_parse_dt(self._resource_date(r)) for r in valid if self._resource_date(r)),
                             default=None)
            else:
                latest = None
            should_act = latest is None or (current_time - latest) > _td(days=365)
            if not should_act:
                return len(posts) == 0
            return any(_has_ref(pl) and pl.get("resourceType") == "ServiceRequest" for _, pl in posts)

        # ── task10: COVID vaccine > 12 months → order vaccine ────────
        elif task_type == "task10":
            current_time_str = _CUTOFF
            ctx = self._task.get("context", "") if self._task else ""
            m = re.search(r"It'?s\s+([\d\-T:+Z]+)\s+now", ctx)
            if m:
                current_time_str = m.group(1)
            current_time = _parse_dt(current_time_str) or _CUTOFF_DT

            resources = self._mock_get_entries("Procedure", f"patient={mrn}&code=COVIDVACCINE")
            valid = [r for r in resources if (_parse_dt(self._resource_date(r)) or _CUTOFF_DT) <= current_time]
            if valid:
                latest = max((_parse_dt(self._resource_date(r)) for r in valid if self._resource_date(r)),
                             default=None)
            else:
                latest = None
            should_act = latest is None or (current_time - latest) > _td(days=365)
            if not should_act:
                return len(posts) == 0
            return any(_has_ref(pl) and pl.get("resourceType") == "ServiceRequest" for _, pl in posts)

        # ── v2_task5: Mg < 2.0 in last 24h → Mg replacement ─────────
        elif task_type == "v2_task5":
            current_time_str = _CUTOFF
            ctx = self._task.get("context", "") if self._task else ""
            m = re.search(r"It'?s\s+([\d\-T:+Z]+)\s+now", ctx)
            if m:
                current_time_str = m.group(1)
            current_time = _parse_dt(current_time_str) or _CUTOFF_DT
            window_start = current_time - _td(hours=24)

            resources = self._mock_get_entries("Observation", f"patient={mrn}&code=MG")
            recent_vals = []
            for r in resources:
                d = _parse_dt(self._resource_date(r))
                if d and window_start <= d <= current_time:
                    try:
                        val = float(r.get("valueQuantity", {}).get("value", "nan"))
                        recent_vals.append((d, val))
                    except (TypeError, ValueError):
                        pass
            if not recent_vals:
                return len(posts) == 0  # no recent Mg → no action
            # Use latest reading in the window
            _, latest_mg = max(recent_vals, key=lambda x: x[0])
            should_act = latest_mg < 2.0
            if not should_act:
                return len(posts) == 0
            return any(_has_ref(pl) and pl.get("resourceType") == "MedicationRequest" for _, pl in posts)

        # ── v2_task9: K+ < 3.5 → K replacement (MedicationRequest required) ──
        elif task_type == "v2_task9":
            current_time_str = _CUTOFF
            ctx = self._task.get("context", "") if self._task else ""
            m = re.search(r"It'?s\s+([\d\-T:+Z]+)\s+now", ctx)
            if m:
                current_time_str = m.group(1)
            current_time = _parse_dt(current_time_str) or _CUTOFF_DT

            resources = self._mock_get_entries("Observation", f"patient={mrn}&code=K")
            valid = []
            for r in resources:
                d = _parse_dt(self._resource_date(r))
                if d and d <= current_time:
                    try:
                        val = float(r.get("valueQuantity", {}).get("value", "nan"))
                        valid.append((d, val))
                    except (TypeError, ValueError):
                        pass
            if not valid:
                return False  # no data → cannot evaluate
            _, latest_k = max(valid, key=lambda x: x[0])
            should_act = latest_k < 3.5
            if not should_act:
                return len(posts) == 0
            # Require the replacement order (MedicationRequest); follow-up lab is optional
            _K_TERMS = ("potassium", "40032-917-01")
            return any(
                _has_ref(pl)
                and pl.get("resourceType") == "MedicationRequest"
                and any(t in json.dumps(pl).lower() for t in _K_TERMS)
                for _, pl in posts
            )

        # ── v2_task10: A1C > 1 year → order A1C ─────────────────────
        elif task_type == "v2_task10":
            current_time_str = _CUTOFF
            ctx = self._task.get("context", "") if self._task else ""
            m = re.search(r"It'?s\s+([\d\-T:+Z]+)\s+now", ctx)
            if m:
                current_time_str = m.group(1)
            current_time = _parse_dt(current_time_str) or _CUTOFF_DT

            resources = self._mock_get_entries("Observation", f"patient={mrn}&code=A1C")
            valid = []
            for r in resources:
                d = _parse_dt(self._resource_date(r))
                if d and d <= current_time:
                    valid.append(d)
            if not valid:
                latest = None
            else:
                latest = max(valid)
            should_act = latest is None or (current_time - latest) > _td(days=365)
            if not should_act:
                return len(posts) == 0
            return any(_has_ref(pl) and pl.get("resourceType") == "ServiceRequest" for _, pl in posts)

        return False  # unsupported task type

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


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

# def reward_func(completions, environments=None, **kwargs):
#     """Return shaped reward from each episode's environment.

#     Standard TRL passes `environments` directly. Unsloth's patched
#     _calculate_rewards does not forward it, so we fall back to the
#     class-level registry which tracks every instance in creation order.
#     """
#     if environments is None:
#         environments = kwargs.get("environments")

#     if environments is not None:
#         return [float(env.reward) for env in environments]

#     # Unsloth fallback: pop the oldest N envs from the class registry
#     n = len(completions)
#     envs = MedAgentTrainEnv._registry[:n]
#     del MedAgentTrainEnv._registry[:n]
#     return [float(env.reward) for env in envs]

def reward_func(prompts, completions, environments=None, **kwargs):
    """
    GRPO calls this with len(completions) = num_prompts * num_generations.
    Each env executes once per prompt, so we must tile rewards across generations.
    """
    num_completions = len(completions)
    
    if environments is None:
        environments = kwargs.get("environments")

    if environments is not None:
        envs = environments
    else:
        # Unsloth fallback: registry has one env per prompt
        num_envs = len(MedAgentTrainEnv._registry)
        # num_generations = completions / envs
        n_prompts = num_envs  # one env was created per prompt
        envs = MedAgentTrainEnv._registry[:n_prompts]
        del MedAgentTrainEnv._registry[:n_prompts]

    # Tile: each env's reward repeats num_generations times
    n_prompts = len(envs)
    if n_prompts == 0:
        return [0.0] * num_completions

    num_generations = num_completions // n_prompts  # e.g. 4 // 2 = 2
    rewards = []
    for env in envs:
        # Evaluate partial episodes (model stopped without calling finish or
        # hitting max_steps). This ensures GET_CREDIT and action rewards are
        # still visible as a learning signal instead of silently returning 0.
        if not env.done:
            env.reward = env._evaluate()
            env._print_trace()
        rewards.extend([float(env.reward)] * num_generations)
    
    # Safety: pad or truncate to exact length GRPO expects
    if len(rewards) < num_completions:
        rewards.extend([0.0] * (num_completions - len(rewards)))
    return rewards[:num_completions]    


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

# RL-worthy task types: require conditional act/no-act decision.
# Excludes task3 (always-action BP vitals) which provides no decision signal.
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

        # Load v2 tasks (Mg/K+/A1c) with "v2_" prefix to avoid ID collisions
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


def build_dataset(data_dir: Path, num_tasks: Optional[int] = None) -> Dataset:
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


def _export_reward_graph(output_dir: str, log_history: List[Dict[str, Any]]) -> None:
    """Export reward metrics and a reward-curve image."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[int, float, float]] = []
    for item in log_history:
        # Keep only true training log rows that include reward metrics.
        if "rewards/reward_func/mean" not in item and "reward" not in item:
            continue
        step = item.get("step")
        if not isinstance(step, (int, float)):
            continue
        reward_mean = item.get("rewards/reward_func/mean", item.get("reward", 0.0))
        reward_std = item.get("rewards/reward_func/std", item.get("reward_std", 0.0))
        try:
            rows.append((int(step), float(reward_mean), float(reward_std)))
        except Exception:
            continue

    if not rows:
        print("No reward history found; skipped reward graph export.")
        return

    rows.sort(key=lambda x: x[0])
    csv_path = out_dir / "reward_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "reward_mean", "reward_std"])
        writer.writerows(rows)
    print(f"Saved reward metrics CSV: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipped reward_curve.png")
        return

    steps = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [max(0.0, r[2]) for r in rows]
    lows = [m - s for m, s in zip(means, stds)]
    highs = [m + s for m, s in zip(means, stds)]

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, means, linewidth=2.0, label="reward mean")
    plt.fill_between(steps, lows, highs, alpha=0.2, label="reward std")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    png_path = out_dir / "reward_curve.png"
    plt.savefig(png_path, dpi=140)
    plt.close()
    print(f"Saved reward graph: {png_path}")


def _export_completions_debug(output_dir: str) -> None:
    """Auto-export readable CSV/JSONL from rollout parquet logs."""
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except Exception:
        print("pandas/pyarrow unavailable; skipped completion debug export.")
        return

    completions_dir = Path(output_dir) / "completions"
    parquet_files = sorted(completions_dir.glob("completions_*.parquet"))
    if not parquet_files:
        print(f"No completion parquets found under {completions_dir}; skipped debug export.")
        return

    debug_dir = Path(output_dir) / "debug_readable"
    debug_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for file_path in parquet_files:
        table = pq.read_table(file_path)
        cols = table.column_names
        col_data = {c: table[c].to_pylist() for c in cols}
        n_rows = len(next(iter(col_data.values()))) if col_data else 0
        for idx in range(n_rows):
            raw = {c: col_data[c][idx] for c in cols}
            completion = str(raw.get("completion") or "")
            prompt = str(raw.get("prompt") or "")
            tool_names = re.findall(r'"name"\s*:\s*"([^"]+)"', completion)
            rows.append(
                {
                    "source_file": file_path.name,
                    "row_idx": idx,
                    "step": raw.get("step"),
                    "reward_func": raw.get("reward_func"),
                    "advantage": raw.get("advantage"),
                    "tool_calls": completion.count("<tool_call>"),
                    "has_finish": bool(re.search(r'"name"\s*:\s*"finish"', completion)),
                    "tool_names": "|".join(tool_names[:12]),
                    "completion_chars": len(completion),
                    "prompt_chars": len(prompt),
                    "completion_preview": completion[:500].replace("\n", " "),
                }
            )

    if not rows:
        print("Completion parquet files had no rows; skipped debug export.")
        return

    df = pd.DataFrame(rows)
    csv_path = debug_dir / "completions_readable.csv"
    jsonl_path = debug_dir / "completions_readable.jsonl"
    summary_path = debug_dir / "summary.json"

    df.to_csv(csv_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "num_files": len(parquet_files),
        "num_rows": int(len(df)),
        "finish_rate": float(df["has_finish"].mean()),
        "mean_tool_calls": float(df["tool_calls"].mean()),
        "mean_completion_chars": float(df["completion_chars"].mean()),
        "files": [p.name for p in parquet_files],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved completion debug CSV: {csv_path}")
    print(f"Saved completion debug JSONL: {jsonl_path}")
    print(f"Saved completion debug summary: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _SELECTED_TASK_TYPES, _TASKS, _TASKS_BY_ID, _TASKS_BY_INSTRUCTION, _TASK_INDEX
    global _MAX_TOOL_RESPONSE_CHARS, _MAX_TOOL_RESPONSE_ENTRIES, _MAX_HISTORY_MESSAGES, _MAX_PROMPT_LENGTH, _MAX_STEPS
    parser = argparse.ArgumentParser(description="Train on MedAgentBench with GRPO")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-1.7B",
        help="Model name or path",
    )
    parser.add_argument(
        "--disable-qwen-thinking",
        action="store_true",
        help="Force chat_template_kwargs enable_thinking=False (Qwen3 or local checkpoints without 'qwen3' in path)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(_DATA_DIR),
        help="Path to directory containing new_patient_tasks.json",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=None,
        help="Number of tasks to use (default: all tasks from selected categories)",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=sorted(_RL_TASK_TYPES),
        help="Task categories to include, e.g. task1 task2 v2_task5",
    )
    parser.add_argument(
        "--max-completion-length", type=int, default=8000,
        help="Max tokens per generation. max_tool_calling_iterations prevents runaway loops.",
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
        "--max-prompt-length", type=int, default=8192,
        help="Max prompt tokens passed to TRL before generation/loss truncation",
    )
    parser.add_argument(
        "--max-history-messages", type=int, default=64,
        help="Max in-episode history messages kept (includes initial system item)",
    )
    parser.add_argument(
        "--max-tool-response-chars", type=int, default=4000,
        help="Max chars kept from tool responses before truncation",
    )
    parser.add_argument(
        "--max-tool-response-entries", type=int, default=24,
        help="Max FHIR Bundle entries returned to the model per GET response",
    )
    parser.add_argument(
        "--max-steps", type=int, default=6,
        help="Max tool actions per episode before forced evaluation",
    )
    parser.add_argument(
        "--num-generations", type=int, default=4,
        help="GRPO num_generations (must divide per-device-batch-size)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce VRAM (recommended)",
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
        default=os.environ.get("HF_TOKEN") or _DEFAULT_HF_TOKEN,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--resume-from-checkpoint", type=str, default=None,
        help="Path to checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--beta", type=float, default=0.001,
        help="KL penalty coefficient (default: 0.001)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.5,
        help="Sampling temperature for generation (default: 1.5)",
    )
    args = parser.parse_args()

    _SELECTED_TASK_TYPES = set(args.task_types)
    _MAX_PROMPT_LENGTH = max(512, int(args.max_prompt_length))
    _MAX_HISTORY_MESSAGES = max(8, int(args.max_history_messages))
    _MAX_TOOL_RESPONSE_CHARS = max(512, int(args.max_tool_response_chars))
    _MAX_TOOL_RESPONSE_ENTRIES = max(4, int(args.max_tool_response_entries))
    _MAX_STEPS = max(2, int(args.max_steps))
    print(
        f"Safeguards: max_prompt_length={_MAX_PROMPT_LENGTH} "
        f"(trainer compatibility mode), "
        f"max_history_messages={_MAX_HISTORY_MESSAGES}, "
        f"max_tool_response_chars={_MAX_TOOL_RESPONSE_CHARS}, "
        f"max_tool_response_entries={_MAX_TOOL_RESPONSE_ENTRIES}, "
        f"max_steps={_MAX_STEPS}"
    )
    # Rebuild task caches with selected task categories for this run.
    _TASKS = []
    _TASKS_BY_ID = {}
    _TASKS_BY_INSTRUCTION = {}
    _TASK_INDEX = 0

    # Pre-load shared resources
    _get_mock_fhir()
    print(f"Loaded FHIR cache from {_CACHE_PATH}")

    dataset = build_dataset(Path(args.data_dir), args.num_tasks)
    print(f"Training dataset: {len(dataset)} tasks")
    if len(dataset) == 0:
        raise RuntimeError(
            "No tasks selected. Check --task-types and --num-tasks settings."
        )
    total_train_steps = max(1, len(dataset) * args.num_train_epochs)
    effective_batch_size = max(1, min(args.per_device_batch_size, len(dataset)))
    effective_grad_accum = max(
        1, min(args.gradient_accumulation_steps, len(dataset))
    )
    if effective_batch_size != args.per_device_batch_size:
        print(
            f"Adjusted per-device batch size from {args.per_device_batch_size} "
            f"to {effective_batch_size} for small dataset."
        )
    if effective_grad_accum != args.gradient_accumulation_steps:
        print(
            f"Adjusted gradient accumulation from {args.gradient_accumulation_steps} "
            f"to {effective_grad_accum} for small dataset."
        )

    # Load model with standard transformers + PEFT (no Unsloth).
    # Unsloth's GRPOTrainer has a hardcoded fp16 autocaster in
    # grpo_accumulated_loss that cannot be overridden by bf16/fp16 flags,
    # causing Half/BFloat16 mismatches.  Standard TRL respects bf16=True.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,  # match Unsloth's fp16 autocaster
        device_map="auto",
    )
    # GRPO generates long sequences; caching KV during training is very VRAM heavy.
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Qwen3: disable chat-template "thinking" so generations favor immediate tool calls.
    if args.per_device_batch_size % int(args.num_generations) != 0:
        raise ValueError(
            f"--per-device-batch-size ({args.per_device_batch_size}) must be divisible by "
            f"--num-generations ({args.num_generations}) for GRPO."
        )

    _grpo_kwargs: Dict[str, Any] = dict(
        output_dir=args.output_dir,
        max_steps=total_train_steps,
        num_train_epochs=args.num_train_epochs,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=effective_batch_size,
        gradient_accumulation_steps=effective_grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=10,
        log_completions=True,
        num_completions_to_print=2,
        logging_steps=1,
        save_steps=10,
        save_total_limit=6,
        fp16=True,
        bf16=False,
        num_generations=int(args.num_generations),
        beta=args.beta,
        temperature=args.temperature,
        max_tool_calling_iterations=_MAX_STEPS + 4,
    )
    if "qwen3" in args.model.lower() or args.disable_qwen_thinking:
        _grpo_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        print("chat_template_kwargs: enable_thinking=False", flush=True)

    grpo_config = GRPOConfig(**_grpo_kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        environment_factory=MedAgentTrainEnv,
        processing_class=tokenizer,
        args=grpo_config,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    _export_reward_graph(args.output_dir, trainer.state.log_history)
    _export_completions_debug(args.output_dir)
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
