"""
MedAgentBench Verifier — decoupled reward/grading logic.

This module is the single source of truth for episode scoring.
It has no dependency on the training loop, TRL, or the environment class.

Public API
----------
evaluate(history, task_spec, fhir_base_url, *, invalid_fhir_count=0,
         new_refsol=None, weights=None) -> float

    Score a completed (or partial) episode.

    Parameters
    ----------
    history : list of {"role": str, "content": str}
        Full conversation history.  role is "agent" or "env".
    task_spec : dict
        Keys: id, instruction, context (optional), sol (optional), eval_MRN.
    fhir_base_url : str
        e.g. "http://localhost:8080/fhir/"
    invalid_fhir_count : int
        Number of malformed/rejected FHIR calls during the episode.
    new_refsol : module | None
        Pre-loaded medagentbenchevals.new_refsol module.  If None the
        verifier falls back to the lightweight inline grader.
    weights : RewardWeights | None
        Override default reward weights (useful for ablations / platform tuning).

    Returns
    -------
    float in [-1.0, 2.0]
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Reward weights — override via RewardWeights for platform experiments
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    terminal: float       = 1.00   # refsol grader passes
    get_credit: float     = 0.20   # agent read the chart before deciding
    action_a: float       = 0.25   # correct primary POST detected
    action_b: float       = 0.25   # correct secondary POST (task5/7/v2_task9)
    spurious_post: float  = -0.15  # POST that matched neither action_a nor action_b
    invalid_fhir: float   = -0.10  # per malformed/rejected FHIR call
    redundant_lookup: float     = -0.05
    redundant_lookup_cap: float = -0.20
    offtarget_lookup: float     = -0.05
    offtarget_lookup_cap: float = -0.20


_DEFAULT_WEIGHTS = RewardWeights()


# ---------------------------------------------------------------------------
# Task-specific knowledge tables
# ---------------------------------------------------------------------------

# FHIR resources each task is expected to GET (off-target GETs are penalised)
ALLOWED_GET_RESOURCES: Dict[str, set] = {
    "task1":     {"Procedure"},
    "task2":     {"MedicationRequest"},
    "task4":     {"Procedure"},
    "task5":     {"Condition", "Procedure"},
    "task6":     {"Observation"},
    "task7":     {"Observation", "MedicationRequest"},
    "task8":     {"MedicationRequest"},
    "task9":     {"Procedure"},
    "task10":    {"Procedure"},
    "v2_task5":  {"Observation"},
    "v2_task9":  {"Observation"},
    "v2_task10": {"Observation"},
}

NON_ACTIVE_STATUSES = frozenset({
    "stopped", "cancelled", "completed", "entered-in-error",
})

QT_PROLONGING_MEDS = (
    "ondansetron", "prochlorperazine", "haloperidol", "quetiapine",
    "olanzapine", "risperidone", "ziprasidone", "clozapine", "chlorpromazine",
)

ANTICOAG_MEDS = (
    "heparin", "enoxaparin", "lovenox", "fondaparinux",
    "rivaroxaban", "apixaban", "dabigatran", "warfarin",
    "tinzaparin", "dalteparin",
)


# ---------------------------------------------------------------------------
# Inline fallback grader (used when new_refsol is unavailable)
# ---------------------------------------------------------------------------

def _inline_pass(task_type: str, mrn: str, history: List[Dict]) -> bool:
    """Minimal pass/fail check when new_refsol is not available."""
    posts = _accepted_posts(history)
    patient_ref = f"Patient/{mrn}"

    def has_ref(pl: Dict) -> bool:
        subj = pl.get("subject", {})
        if isinstance(subj, list):
            subj = subj[0] if subj else {}
        return subj.get("reference") == patient_ref

    for pl in posts:
        blob = json.dumps(pl).lower()
        rtype = pl.get("resourceType", "")
        if task_type == "task1" and rtype == "ServiceRequest":
            if has_ref(pl) and "74177" in blob:
                return True
        elif task_type == "task2" and rtype == "MedicationRequest":
            if any(k in blob for k in ANTICOAG_MEDS):
                return True
        elif task_type == "task4" and rtype == "ServiceRequest":
            if has_ref(pl) and ("catheter" in blob or "removal" in blob):
                return True
        # Tasks 3/5/6/7/8/9/10 and v2 variants are too complex for inline grading
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _accepted_posts(history: List[Dict]) -> List[Dict]:
    """Return parsed payloads of POST requests that were accepted by the env."""
    posts = []
    msgs = list(history)
    for i, msg in enumerate(msgs):
        if msg.get("role") != "agent":
            continue
        if not msg.get("content", "").startswith("POST"):
            continue
        next_msg = msgs[i + 1] if i + 1 < len(msgs) else None
        if next_msg and "POST request accepted" in next_msg.get("content", ""):
            try:
                lines = msg["content"].split("\n", 1)
                payload = json.loads(lines[1]) if len(lines) > 1 else {}
                posts.append(payload)
            except Exception:
                pass
    return posts


def _get_urls(history: List[Dict]) -> List[str]:
    """Return list of GET URLs the agent called."""
    urls = []
    for msg in history:
        if msg.get("role") == "agent" and msg.get("content", "").startswith("GET "):
            parts = msg["content"].split()
            if len(parts) > 1:
                urls.append(parts[1])
    return urls


def _action_rewards(
    task_type: str, mrn: str, posts: List[Dict], w: RewardWeights
) -> tuple[bool, bool]:
    """Return (found_a, found_b) for task-specific action detection."""
    patient_ref = f"Patient/{mrn}"
    found_a = found_b = False

    def has_ref(pl: Dict) -> bool:
        subj = pl.get("subject", {})
        if isinstance(subj, list):
            subj = subj[0] if subj else {}
        return subj.get("reference") == patient_ref

    for pl in posts:
        blob = json.dumps(pl).lower()
        rtype = pl.get("resourceType", "")

        if task_type == "task1":
            if rtype == "ServiceRequest" and (has_ref(pl) or "74177" in blob):
                found_a = True

        elif task_type == "task2":
            if rtype == "MedicationRequest" and any(k in blob for k in ANTICOAG_MEDS):
                found_a = True

        elif task_type == "task4":
            if rtype == "ServiceRequest" and has_ref(pl):
                if "nur1373" in blob or "catheter" in blob or "removal" in blob:
                    found_a = True

        elif task_type == "task5":
            if rtype == "ServiceRequest" and has_ref(pl):
                if "74177" in blob:
                    found_a = True
                if "con417" in blob or "interventional radiology" in blob:
                    found_b = True

        elif task_type == "task6":
            if rtype == "MedicationRequest" and has_ref(pl):
                if "levothyroxine" in blob:
                    found_a = True
            if rtype == "ServiceRequest" and has_ref(pl):
                if "tsh" in blob or "ft4" in blob or "thyroid" in blob:
                    found_b = True

        elif task_type == "task7":
            status = (pl.get("status") or "").lower()
            if rtype == "MedicationRequest" and has_ref(pl):
                if status in NON_ACTIVE_STATUSES and any(m in blob for m in QT_PROLONGING_MEDS):
                    found_a = True
            if rtype == "ServiceRequest" and has_ref(pl):
                if "445118002" in blob or "ecg" in blob:
                    found_b = True

        elif task_type == "task8":
            if rtype == "MedicationRequest" and has_ref(pl):
                if "naloxone" in blob:
                    found_a = True

        elif task_type == "task9":
            if rtype == "ServiceRequest" and has_ref(pl):
                if "90686" in blob or "influenza" in blob:
                    found_a = True

        elif task_type == "task10":
            if rtype == "ServiceRequest" and has_ref(pl):
                if "91320" in blob or "covid" in blob:
                    found_a = True

        elif task_type == "v2_task5":
            if rtype == "MedicationRequest" and has_ref(pl):
                if "0338-1715-40" in blob or "magnesium" in blob:
                    found_a = True

        elif task_type == "v2_task9":
            if rtype == "MedicationRequest" and has_ref(pl):
                if "40032-917-01" in blob or "potassium" in blob:
                    found_a = True
            if rtype == "ServiceRequest" and has_ref(pl):
                if "2823-3" in blob or "potassium" in blob:
                    found_b = True

        elif task_type == "v2_task10":
            if rtype == "ServiceRequest" and has_ref(pl):
                if "4548-4" in blob or "a1c" in blob or "hba1c" in blob:
                    found_a = True

    return found_a, found_b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    history: List[Dict[str, str]],
    task_spec: Dict[str, Any],
    fhir_base_url: str,
    *,
    invalid_fhir_count: int = 0,
    new_refsol=None,
    weights: Optional[RewardWeights] = None,
    grader_timeout: float = 60.0,
) -> float:
    """Score a completed or partial episode.

    Parameters
    ----------
    history : list of {"role": str, "content": str}
    task_spec : dict  — keys: id, instruction, context, sol, eval_MRN
    fhir_base_url : str
    invalid_fhir_count : int
    new_refsol : module | None  — pre-loaded grader module
    weights : RewardWeights | None
    grader_timeout : float  — seconds before refsol grader is abandoned

    Returns
    -------
    float in [-1.0, 2.0]
    """
    w = weights if weights is not None else _DEFAULT_WEIGHTS

    task_id = task_spec.get("id", "")
    parts = task_id.rsplit("_", 1)
    task_type = parts[0] if len(parts) == 2 and parts[1].isdigit() else task_id
    mrn = task_spec.get("eval_MRN", "")

    # ── 1. Terminal grader ───────────────────────────────────────────────────
    refsol_pass = False

    def _run_grader() -> bool:
        if new_refsol is not None:
            grader_fn = getattr(new_refsol, task_type, None)
            if grader_fn is not None:
                import types as _types
                case_data = {
                    "id": task_id,
                    "instruction": task_spec.get("instruction", ""),
                    "context": task_spec.get("context", ""),
                    "sol": task_spec.get("sol", []),
                    "eval_MRN": mrn,
                }
                eval_results = _types.SimpleNamespace(history=history, result=None)
                try:
                    return grader_fn(case_data, eval_results, fhir_base_url) is True
                except Exception as e:
                    print(f"[verifier] new_refsol grader error for {task_id}: {e}")
        return _inline_pass(task_type, mrn, history)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_run_grader)
            try:
                refsol_pass = fut.result(timeout=grader_timeout)
            except concurrent.futures.TimeoutError:
                print(f"[verifier] grader timeout for {task_id} — reward_pass=False")
    except Exception as e:
        print(f"[verifier] grader exception for {task_id}: {e}")

    # ── 2. Parse history ─────────────────────────────────────────────────────
    get_urls = _get_urls(history)
    posts = _accepted_posts(history)
    num_posts = sum(
        1 for msg in history
        if msg.get("role") == "agent" and msg.get("content", "").startswith("POST ")
    )

    # ── 3. Build reward ──────────────────────────────────────────────────────
    reward = 0.0

    if refsol_pass:
        reward += w.terminal

    if get_urls:
        reward += w.get_credit

    # Redundant GET penalty
    seen: set = set()
    redundant = sum(1 for url in get_urls if url in seen or seen.add(url))  # type: ignore[func-returns-value]
    reward += max(w.redundant_lookup_cap, w.redundant_lookup * redundant)

    # Invalid FHIR call penalty
    reward += w.invalid_fhir * invalid_fhir_count

    # Off-target GET penalty
    allowed = ALLOWED_GET_RESOURCES.get(task_type)
    if allowed:
        offtarget = sum(
            1 for url in get_urls
            if url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] not in allowed
        )
        reward += max(w.offtarget_lookup_cap, w.offtarget_lookup * offtarget)

    # Action rewards
    found_a, found_b = _action_rewards(task_type, mrn, posts, w)
    if found_a:
        reward += w.action_a
    if found_b:
        reward += w.action_b

    # Spurious POST penalty
    if not refsol_pass and posts and not found_a and not found_b:
        reward += w.spurious_post

    return max(-1.0, min(2.0, reward))
