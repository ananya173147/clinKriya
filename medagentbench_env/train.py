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

    # Class-level registry — survives module reloads as long as the same
    # class object is used by both environment_factory and reward_func.
    # Unsloth's _calculate_rewards does not forward `environments` to
    # reward_func, so we track instances here and pop them in order.
    _registry: "List[MedAgentTrainEnv]" = []

    def __init__(self):
        MedAgentTrainEnv._registry.append(self)
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
            return pl.get("subject", {}).get("reference") == patient_ref

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
            recent = []
            for r in resources:
                d = _parse_dt(self._resource_date(r))
                if d and window_start <= d <= current_time:
                    try:
                        val = float(r.get("valueQuantity", {}).get("value", "nan"))
                        recent.append(val)
                    except (TypeError, ValueError):
                        pass
            if not recent:
                return len(posts) == 0  # no recent Mg → no action
            should_act = max(recent) < 2.0  # latest Mg low
            # Use latest value
            latest_mg = recent[-1]
            should_act = latest_mg < 2.0
            if not should_act:
                return len(posts) == 0
            return any(_has_ref(pl) and pl.get("resourceType") == "MedicationRequest" for _, pl in posts)

        # ── v2_task9: K+ < 3.5 → K replacement ──────────────────────
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
                return False  # can't determine
            _, latest_k = max(valid, key=lambda x: x[0])
            should_act = latest_k < 3.5
            if not should_act:
                return len(posts) == 0
            return any(_has_ref(pl) and pl.get("resourceType") == "MedicationRequest" for _, pl in posts)

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

        task_id = self._task["id"]
        # v2_ tasks: type is everything before the numeric suffix, e.g. "v2_task5"
        parts = task_id.rsplit("_", 1)
        task_type = parts[0] if len(parts) == 2 and parts[1].isdigit() else task_id
        mrn = self._task.get("eval_MRN", "")

        refsol_pass = self._compute_refsol_pass(task_type, mrn)

        # ── Count GETs and unique GET URLs ──────────────────────────
        get_urls: List[str] = []
        num_posts = 0
        for msg in self._history:
            if msg.role != "agent":
                continue
            if msg.content.startswith("GET "):
                url = msg.content.split()[1] if len(msg.content.split()) > 1 else ""
                get_urls.append(url)
            elif msg.content.startswith("POST "):
                num_posts += 1

        # ── Reward ──────────────────────────────────────────────────
        reward = 1.0 if refsol_pass else 0.0

        # Bonus: agent read the chart before deciding
        if get_urls:
            reward += 0.2

        # Penalty: repeated identical GET URLs
        seen: set = set()
        redundant = 0
        for url in get_urls:
            if url in seen:
                redundant += 1
            seen.add(url)
        reward -= min(0.05 * redundant, 0.2)

        return max(-0.3, min(1.2, reward))


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
    global _TASKS
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

        _TASKS = [
            t for t in all_tasks
            if any(t["id"].startswith(f"{tt}_") for tt in _RL_TASK_TYPES)
        ]
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
        help="Path to directory containing new_patient_tasks.json",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=None,
        help="Number of tasks to use (default: all RL-worthy tasks)",
    )
    parser.add_argument(
        "--max-completion-length", type=int, default=512,
        help="Max tokens per generation (tool calls are short; 512 is enough)",
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

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=10,
        log_completions=True,
        num_completions_to_print=2,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        # Group size: 4 completions per prompt keeps epochs completable.
        # Default (8) means 8×90=720 full episodes before any gradient update.
        num_generations=4,
        # KL penalty against the frozen reference policy.
        # Without this (beta=0 default) the policy can collapse freely.
        beta=0.01,
        # Temperature for diverse rollouts within each group.
        # Greedy (temp=0) makes all 4 completions identical → zero variance
        # in advantages → GRPO gradient is zero on every step.
        temperature=0.9,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        environment_factory=MedAgentTrainEnv,
        processing_class=tokenizer,
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
