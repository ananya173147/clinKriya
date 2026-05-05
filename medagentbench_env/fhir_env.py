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

# Set by MedAgentTrainEnv._evaluate() before calling the grader so the patched
# send_get_request can consult the current episode's overlay (cache mutations).
# Cleared after the grader returns. Reads cross-thread are safe-by-construction
# because GRPO calls _evaluate sequentially per rollout.
_CURRENT_GRADING_ENV: "Optional[MedAgentTrainEnv]" = None
_SYSTEM_PROMPT: str = ""
_TASKS: List[Dict] = []
_TASK_INDEX: int = 0
_TASKS_BY_ID: Dict[str, Dict] = {}
_TASKS_BY_INSTRUCTION: Dict[str, Dict] = {}
_SELECTED_TASK_TYPES: Optional[set] = None
_NEW_REFSOL = None
_V1_REFSOL = None
_TASKS_FILE: Optional[Path] = None  # overrides default new_patient_tasks.json when set

# Safeguard defaults — overridden by CLI args in main()
_MAX_TOOL_RESPONSE_CHARS = 16000
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
        # Auto-merge synthetic-patient additions if present (build_synth.py output).
        # This is opt-in by file presence, so production users without the synth
        # data file are unaffected.
        synth_path = _DATA_DIR / "synth_cache_additions.json"
        if synth_path.exists():
            import json as _json
            with open(synth_path) as _sf:
                synth_entries = _json.load(_sf)
            _MOCK_FHIR._cache.update(synth_entries)
            # Reset name+DOB index since new Patient entries were added
            _MOCK_FHIR._patient_name_index = None
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
        # The grader's send_get_request first checks the current episode's
        # overlay (set by _CURRENT_GRADING_ENV), so cache mutations made by
        # tools like fhir_medication_request_update during the episode are
        # visible to the canonical grader at episode end.
        # NOTE: graders intentionally read from MockFHIR baseline (NOT the
        # episode overlay). The grader's notion of "active anticoagulants" is
        # the pre-action baseline; the agent's discontinue actions are checked
        # via extract_posts on the agent's POST history. If the grader saw the
        # overlay, the baseline would shrink as the agent stops orders, and
        # the grader would mistakenly think "no action was needed" → reject.
        def _send(url, params=None, headers=None, _m=mock):
            return {"status_code": 200, "data": json.dumps(_m.get(url).get("data", {}))}
        new_refsol.send_get_request = _send
        _NEW_REFSOL = new_refsol
        print("Loaded new_refsol graders (single source of truth).")
    except ImportError as e:
        print(f"Warning: could not load medagentbenchevals.new_refsol ({e}); falling back to inline grader.")
    return _NEW_REFSOL


def _get_v1_refsol():
    """Load canonical v1 refsol.py (MedAgentBench v1 graders) with MockFHIR patching.

    v1 graders expect raw-HTTP agent history ("POST {url}\\n{payload}", "GET {url}",
    "POST request accepted" markers) — the env already emits this format, so no
    history translation is required.

    We load refsol.py and its utils.py directly via importlib.util rather than
    going through the v1 package system, because the v1 package __init__.py pulls
    in heavy Task/Session machinery unrelated to grading.
    """
    global _V1_REFSOL
    if _V1_REFSOL is not None:
        return _V1_REFSOL
    import importlib.util
    base = (
        Path(__file__).resolve().parent.parent
        / "medagentbenchv2" / "medagentbench_v2" / "src"
        / "MedAgentBench" / "src" / "server" / "tasks" / "medagentbench"
    )
    try:
        # Create parent package shim so `from .utils import *` resolves.
        pkg_name = "_v1_medagentbench_refsol_pkg"
        pkg = type(sys)("__shim__")
        pkg.__path__ = [str(base)]
        sys.modules[pkg_name] = pkg
        # Load utils.py first
        utils_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.utils", base / "utils.py"
        )
        utils_mod = importlib.util.module_from_spec(utils_spec)
        sys.modules[f"{pkg_name}.utils"] = utils_mod
        utils_spec.loader.exec_module(utils_mod)
        # Load refsol.py with the shim as its package
        refsol_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.refsol", base / "refsol.py"
        )
        v1_refsol = importlib.util.module_from_spec(refsol_spec)
        sys.modules[f"{pkg_name}.refsol"] = v1_refsol
        refsol_spec.loader.exec_module(v1_refsol)
        mock = _get_mock_fhir()
        _patched = (
            lambda url, params=None, headers=None, _m=mock: {
                "status_code": 200,
                "data": json.dumps(_m.get(url).get("data", {})),
            }
        )
        v1_refsol.send_get_request = _patched
        utils_mod.send_get_request = _patched
        _V1_REFSOL = v1_refsol
        print("Loaded v1 refsol graders (canonical MedAgentBench v1).")
    except Exception as e:
        print(f"Warning: could not load v1 refsol ({type(e).__name__}: {e}).")
    return _V1_REFSOL


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

# Excludes v1_task3 (always-action HR average) which provides no decision signal.
# Includes both bare-prefix (legacy test_data_v2.json) and corpus-prefixed
# (clinkriya_train/test.json) IDs so either dataset works as --tasks-file.
_RL_TASK_TYPES = {
    # Legacy bare names
    "task1", "task2", "task4", "task5", "task6",
    "task7", "task8", "task9", "task10",
    "v2_task5", "v2_task9", "v2_task10",
    # clinKriya corpus-prefixed names (v1_ for v1 tasks, v2new_ for v2-new)
    "v1_task1", "v1_task2", "v1_task4", "v1_task5", "v1_task6",
    "v1_task7", "v1_task8", "v1_task9", "v1_task10",
    "v2new_task1", "v2new_task2", "v2new_task3", "v2new_task4", "v2new_task5",
    "v2new_task6", "v2new_task7", "v2new_task8", "v2new_task9", "v2new_task10",
}


def _get_tasks() -> List[Dict]:
    """Load all RL-worthy tasks from new_patient_tasks.json and test_data_v2.json."""
    global _TASKS, _TASKS_BY_ID, _TASKS_BY_INSTRUCTION
    if not _TASKS:
        tasks_file = _TASKS_FILE if _TASKS_FILE is not None else (_DATA_DIR / "new_patient_tasks.json")
        with open(tasks_file) as f:
            all_tasks: List[Dict] = json.load(f)

        # Only append raw v2 tasks when using the default file; custom task files
        # (e.g. train_tasks.json) already include v2 entries with v2_ prefixes.
        if _TASKS_FILE is None:
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
        # Episode-local cache overlay — mutations made by tools like
        # fhir_medication_request_update live here, NOT in MockFHIR.
        # Keyed by normalized URL → response dict matching MockFHIR.get() shape.
        # Cleared on reset() so mutations never leak across episodes/rollouts.
        self._episode_overlay: Dict[str, Any] = {}

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
        self._episode_overlay = {}  # discard prior episode's mutations

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
            code: FHIR code with coding list, e.g.
                {'coding': [{'code': '8867-4', 'system': 'http://loinc.org'}]}.
            effectiveDateTime: ISO datetime of the measurement.
            status: Observation status (default 'final').
            valueString: The vital sign value as a string.
            subject: Patient reference — exactly {'reference': 'Patient/<MRN>'}.

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
        """Create a service request (lab order, imaging order, or referral) in the FHIR EHR.

        Args:
            resourceType: Must be 'ServiceRequest'.
            code: FHIR code with coding list. For CPT codes use
                {'coding': [{'code': '74177', 'system': 'http://www.ama-assn.org/go/cpt'}]}.
                For LOINC codes use
                {'coding': [{'code': '4548-4', 'system': 'http://loinc.org'}]}.
            authoredOn: ISO datetime the order was written.
            status: Request status — 'active' for a new order (default 'active').
            intent: Request intent (default 'order').
            priority: Priority (default 'stat').
            subject: Patient reference — exactly {'reference': 'Patient/<MRN>'}.
            note: Clinical indication or reason as a list of dicts, e.g.
                [{'text': 'Renal mass follow-up'}]. Used for the order indication.
            occurrenceDateTime: When the service should occur (ISO datetime).

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
        """Create or discontinue a medication order in the FHIR EHR.

        Args:
            resourceType: Must be 'MedicationRequest'.
            medicationCodeableConcept: Medication identity including full name, dose, route and
                frequency as free text, e.g. {'text': 'heparin 5000 units SC q8h'} or
                {'text': 'ondansetron 4 mg IV'}. Include the numeric dose — e.g. '5000 units',
                not just the drug name.
            subject: Patient reference — exactly {'reference': 'Patient/<MRN>'}.
            status: 'active' to create a new order; 'stopped' to discontinue an existing one
                (default 'active').
            intent: Request intent (default 'order').
            authoredOn: ISO datetime the order was written.
            dosageInstruction: List of dosage instruction dicts (optional).
            note: Clinical notes as a list of dicts, e.g. [{'text': 'DVT prophylaxis'}].

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

    def fhir_medication_request_update(self, id: str, status: str) -> str:
        """Update an existing MedicationRequest's status (e.g. 'stopped',
        'cancelled', 'completed'). Used to discontinue active orders.

        The mutation is recorded in the episode-local cache overlay and is
        VISIBLE for subsequent GETs in the same episode AND to the canonical
        grader at episode end. The mutation is automatically discarded on the
        next env.reset(), so no state leaks across rollouts.

        Args:
            id: The id of the existing MedicationRequest to update (from a
                prior fhir_medication_request_search response's entry[i].resource.id).
            status: The new status — typically 'stopped', 'cancelled', or 'completed'.
        Returns:
            Confirmation message.
        """
        if self.done:
            return "Episode already finished."
        if not self._task or not self._task.get("eval_MRN"):
            return "Error: no current patient context."
        mrn = self._task["eval_MRN"]
        # Apply mutation to both the active-filtered and unfiltered MedicationRequest
        # cache keys for this patient. The active-filtered key MUST drop the
        # entry if the new status is non-active (so the grader's status=active
        # query no longer finds it). The unfiltered key keeps the entry but
        # with the new status field.
        fhir_base = _FHIR_API_BASE.rstrip("/")
        active_url = f"{fhir_base}/MedicationRequest?_count=5000&_format=json&patient={mrn}&status=active"
        unfilt_url = f"{fhir_base}/MedicationRequest?_count=5000&_format=json&patient={mrn}"
        non_active = {"stopped", "cancelled", "completed", "entered-in-error"}
        for url in (active_url, unfilt_url):
            key = _mod._normalize_url(url)
            # Read current state — overlay takes precedence over cache
            current = self._episode_overlay.get(key) or self._mock.get(url)
            data = current.get("data", {}) if isinstance(current, dict) else {}
            if isinstance(data, str):
                try: data = json.loads(data)
                except Exception: data = {}
            entries = list(data.get("entry", []) or [])
            new_entries: List[Dict[str, Any]] = []
            mutated = False
            for entry in entries:
                res = entry.get("resource", {}) or {}
                if res.get("id") == id:
                    mutated = True
                    if "status=active" in url and status.lower() in non_active:
                        # Drop from active-filtered Bundle
                        continue
                    # Mutate status in place (deep copy to avoid touching MockFHIR)
                    import copy
                    new_res = copy.deepcopy(res)
                    new_res["status"] = status
                    new_entry = dict(entry)
                    new_entry["resource"] = new_res
                    new_entries.append(new_entry)
                else:
                    new_entries.append(entry)
            new_data = dict(data)
            new_data["entry"] = new_entries
            new_data["total"] = len(new_entries)
            self._episode_overlay[key] = {"status_code": 200, "data": new_data}

        msg = f"MedicationRequest {id} status updated to '{status}'."
        if not mutated:
            msg = f"Warning: MedicationRequest {id} not found for patient {mrn}; no update applied."
        self._step_count += 1
        # Emit POST-formatted history so the canonical graders (which parse the
        # agent's history for POST blocks via extract_posts) see this as a
        # status-change submission. The mutation is also recorded as a "post"
        # in self._post_requests so verifier-side action shaping fires.
        update_payload = {
            "resourceType": "MedicationRequest",
            "id": id,
            "status": status,
            "intent": "order",
            "subject": {"reference": f"Patient/{mrn}"},
        }
        post_line = f"POST {_FHIR_API_BASE.rstrip('/')}/MedicationRequest\n{json.dumps(update_payload)}"
        self._append_history("agent", post_line)
        self._post_requests.append(update_payload)
        self._append_history("user",
            "POST request accepted and executed successfully. Please call finish "
            "if you have got answers for all the questions and finished all the requested tasks")
        return msg

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

    @staticmethod
    def _coerce_numeric(v):
        """Normalize string numerals → int/float so grader's strict `== ` check passes.
        Qwen3-8B has a strong JSON-stringify prior despite explicit "numbers not
        strings" instructions. This coercion matches what any production inference
        wrapper would do before sending to a grader. -1 stays -1, "-1" → -1."""
        if isinstance(v, str):
            s = v.strip()
            try:
                i = int(s)
                return i
            except ValueError:
                pass
            try:
                return float(s)
            except ValueError:
                return v
        return v

    def finish(self, value: List[Any]) -> str:
        """Signal task completion and provide the final answer.

        Args:
            value: List of answer values, e.g. ['S6534835'] or [10] or [].

        Returns:
            Completion confirmation with reward.
        """
        if self.done:
            return "Episode already finished."
        # Robust unwrapping: tolerate string-wrapped JSON forms the model emits
        # under tool-call confusion. e.g. value="[-1]" → [-1]; value="[]" → [].
        if isinstance(value, str):
            s = value.strip()
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        value = parsed
                except Exception:
                    pass
        raw = value if isinstance(value, list) else [value]
        # Also unwrap inner string-wrapped list elements: ["[-1]"] → [-1].
        unwrapped = []
        for v in raw:
            if isinstance(v, str):
                s = v.strip()
                if (s.startswith("[") and s.endswith("]")):
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, list):
                            unwrapped.extend(parsed)
                            continue
                    except Exception:
                        pass
            unwrapped.append(v)
        raw = unwrapped
        self._agent_answer = [self._coerce_numeric(v) for v in raw]
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

    def _get_with_overlay(self, url: str) -> Dict[str, Any]:
        """Look up url in the episode overlay first, else delegate to MockFHIR.
        Overlay keys are normalized via the same _normalize_url logic.
        We use the already-loaded _mod (server.fhir_cache) to avoid triggering
        the medagentbench_env.server package __init__ (which pulls openenv)."""
        key = _mod._normalize_url(url)
        if key in self._episode_overlay:
            return self._episode_overlay[key]
        if url in self._episode_overlay:
            return self._episode_overlay[url]
        return self._mock.get(url)

    def _do_get(self, resource: str, params: Dict[str, str]) -> str:
        self._step_count += 1
        fhir_base = _FHIR_API_BASE.rstrip("/")
        param_str = urlencode(sorted(params.items()))
        url = f"{fhir_base}/{resource}?{param_str}&_format=json" if param_str else f"{fhir_base}/{resource}?_format=json"
        self._append_history("agent", f"GET {url}")

        result = self._get_with_overlay(url)
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
        # Make this episode's overlay visible to the grader's send_get_request
        # (see _get_new_refsol). Always restore on exit, even on exceptions.
        global _CURRENT_GRADING_ENV
        prev = _CURRENT_GRADING_ENV
        _CURRENT_GRADING_ENV = self
        try:
            return _verifier_evaluate(
                history,
                self._task,
                _FHIR_API_BASE,
                invalid_fhir_count=self._invalid_fhir_count,
                new_refsol=_get_new_refsol(),
                v1_refsol=_get_v1_refsol(),
                agent_answer=self._agent_answer,
            )
        finally:
            _CURRENT_GRADING_ENV = prev
