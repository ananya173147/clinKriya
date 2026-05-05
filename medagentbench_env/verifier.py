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
        Full conversation history.  role is "agent" or "user".
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
    terminal: float            = 1.00   # refsol grader passes
    # Shape-rewards reduced (v27): previous 0.25/0.25/0.10/0.12 caused 44–52%
    # of high-reward trajectories to FAIL the strict grader (RL reinforced
    # right-shape-wrong-answer rollouts). Terminal now dominates the signal.
    get_credit: float          = 0.05   # GET before accepted POST (guards GET→FINISH hack)
    action_a: float            = 0.10   # correct primary POST — full code+type match
    action_b: float            = 0.10   # correct secondary POST (task5/6/7/v2_task9)
    partial_action: float      = 0.05   # right resourceType + patient ref but wrong code
    spurious_post: float       = -0.05  # off-target POST when neither action_a/b nor partial
    skip_finish_penalty: float = -0.20  # finish() with no GET or POST tool use
    invalid_fhir: float        = -0.10  # per malformed/rejected FHIR call
    redundant_lookup: float     = -0.05
    redundant_lookup_cap: float = -0.10  # softened from -0.20 (v24): capped lookup
                                          # penalties were stacking up to -0.40 per
                                          # episode for SFT-style careful exploration,
                                          # contributing to GRPO drift away from
                                          # multi-GET trajectories.
    offtarget_lookup: float     = -0.05
    offtarget_lookup_cap: float = -0.10  # symmetric softening


_DEFAULT_WEIGHTS = RewardWeights()


# ---------------------------------------------------------------------------
# Task-specific knowledge tables
# ---------------------------------------------------------------------------

# FHIR resources each task is expected to GET (off-target GETs are penalised)
ALLOWED_GET_RESOURCES: Dict[str, set] = {
    "task1":      {"Procedure"},
    "task2":      {"MedicationRequest"},
    "task4":      {"Procedure", "ServiceRequest"},
    # task5 = v2new CT+IR malignant neoplasm (checks Condition + prior CT Procedure)
    "task5":      {"Condition", "Procedure"},
    "task6":      {"Observation"},
    "task7":      {"Observation", "MedicationRequest"},
    "task8":      {"MedicationRequest"},
    # task9 = v2new flu vaccine (checks Procedure for CPT 90686)
    "task9":      {"Procedure"},
    # task10 = v1 HbA1C reorder (checks Observation/A1C)
    "task10":     {"Observation"},
    # v1-specific: same task numbers, different clinical content
    "v1_task4":   {"Observation"},          # v1 Mg level lookup (checks Observation/MG)
    "v1_task5":   {"Observation"},          # v1 Mg replacement (checks Observation/MG)
    "v1_task9":   {"Observation"},          # v1 K replacement (checks Observation/K)
    # v2new task10 = COVID vaccine (checks Procedure for COVIDVACCINE)
    "v2_task10":  {"Procedure"},
}

# Expected resource types for partial credit. Each slot may be a single type
# or a tuple of accepted alternatives (matches whatever the grader will accept).
_TASK_ACTION_RTYPES: Dict[str, tuple] = {
    "task1":      ("ServiceRequest",                              None),
    "task2":      ("MedicationRequest",                           None),
    "task4":      ("ServiceRequest",                              None),
    # task5 = v2new CT+IR: two ServiceRequests (CT scan + IR referral)
    "task5":      ("ServiceRequest",                              "ServiceRequest"),
    "task6":      ("MedicationRequest",                           "ServiceRequest"),
    "task7":      ("MedicationRequest",                           "ServiceRequest"),
    "task8":      (("MedicationRequest", "ServiceRequest"),       None),
    # task9 = v2new flu vaccine: ServiceRequest or MedicationRequest
    "task9":      (("ServiceRequest", "MedicationRequest"),       None),
    "task10":     (("ServiceRequest", "MedicationRequest"),       None),
    # v1-specific task types (v1 and v2new use same task numbers for different tasks)
    "v1_task5":   ("MedicationRequest",                           None),
    "v1_task9":   ("MedicationRequest",                           "ServiceRequest"),
    # v2new task10 = COVID vaccine: ServiceRequest or MedicationRequest
    "v2_task10":  (("ServiceRequest", "MedicationRequest"),       None),
}

# Maps internal task_type (used for rewards) to grader function name when they differ.
# v1_taskN types resolve back to refsol.taskN; v2_task10 resolves to new_refsol.task10.
_GRADER_TASK_REMAP: Dict[str, str] = {
    "v1_task4":  "task4",
    "v1_task5":  "task5",
    "v1_task9":  "task9",
    "v2_task10": "task10",
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


def _is_finish_no_tools(history: List[Dict]) -> bool:
    """True if agent called finish() with no preceding GET or POST tool use."""
    has_finish = any(
        msg.get("role") == "agent" and msg.get("content", "").startswith("FINISH(")
        for msg in history
    )
    if not has_finish:
        return False
    return not _get_urls(history) and not any(
        msg.get("role") == "agent" and msg.get("content", "").startswith("POST ")
        for msg in history
    )


def _action_rewards(
    task_type: str, mrn: str, posts: List[Dict], w: RewardWeights
) -> tuple[float, float]:
    """Return (credit_a, credit_b) as floats in [0, action_a/b weight].

    Two-tier credit:
      partial_action (0.10) — correct resourceType + patient reference present
      action_a/b    (0.25) — full code/content match for the clinical action
    """
    patient_ref = f"Patient/{mrn}"
    credit_a: float = 0.0
    credit_b: float = 0.0
    exp_rtypes = _TASK_ACTION_RTYPES.get(task_type, (None, None))

    def has_ref(pl: Dict) -> bool:
        subj = pl.get("subject", {})
        if isinstance(subj, list):
            subj = subj[0] if subj else {}
        return subj.get("reference") == patient_ref

    def _matches(rtype: str, expected) -> bool:
        if expected is None:
            return False
        if isinstance(expected, (tuple, list, set)):
            return rtype in expected
        return rtype == expected

    def _has_authored(pl: Dict) -> bool:
        return bool(pl.get("authoredOn"))

    def _has_dosage(pl: Dict) -> bool:
        d = pl.get("dosageInstruction")
        return isinstance(d, list) and len(d) > 0 and "route" in d[0]

    def _is_stat(pl: Dict) -> bool:
        return (pl.get("priority") or "").lower() == "stat"

    def _active_order(pl: Dict) -> bool:
        return ((pl.get("status") or "").lower() == "active"
                and (pl.get("intent") or "").lower() == "order")

    for pl in posts:
        blob = json.dumps(pl).lower()
        rtype = pl.get("resourceType", "")

        # Tier-1 partial credit: right resource type and patient reference
        if _matches(rtype, exp_rtypes[0]) and has_ref(pl):
            credit_a = max(credit_a, w.partial_action)
        if _matches(rtype, exp_rtypes[1]) and has_ref(pl):
            credit_b = max(credit_b, w.partial_action)

        # Tier-2 full credit: task-specific code/content checks
        if task_type == "task1":
            if rtype == "ServiceRequest" and has_ref(pl) and "74177" in blob:
                credit_a = max(credit_a, w.action_a)

        elif task_type == "task2":
            if rtype == "MedicationRequest" and has_ref(pl) and any(k in blob for k in ANTICOAG_MEDS):
                credit_a = max(credit_a, w.action_a)

        elif task_type == "task4":
            if rtype == "ServiceRequest" and has_ref(pl):
                if "nur1373" in blob or "catheter" in blob or "removal" in blob:
                    credit_a = max(credit_a, w.action_a)

        elif task_type == "task5":
            if rtype == "ServiceRequest" and has_ref(pl):
                if "74177" in blob:
                    credit_a = max(credit_a, w.action_a)
                if "con417" in blob or "interventional radiology" in blob:
                    credit_b = max(credit_b, w.action_b)

        elif task_type == "task6":
            # task6 (thyroid protocol — both v1 and v2new): grader requires
            # "25" in med text AND ≥2 lab orders for TSH/FT4 (branch A) or
            # 2 lab orders only (branch B). Tighten med credit to require
            # "25" presence; lab credit unchanged (any tsh/ft4 ServiceRequest).
            if rtype == "MedicationRequest" and has_ref(pl):
                if "levothyroxine" in blob and "25" in blob and _active_order(pl) and _has_authored(pl):
                    credit_a = max(credit_a, w.action_a)
            if rtype == "ServiceRequest" and has_ref(pl):
                if "tsh" in blob or "ft4" in blob or "thyroid" in blob:
                    if _active_order(pl) and _has_authored(pl):
                        credit_b = max(credit_b, w.action_b)

        elif task_type == "task7":
            status = (pl.get("status") or "").lower()
            if rtype == "MedicationRequest" and has_ref(pl):
                if status in NON_ACTIVE_STATUSES and any(m in blob for m in QT_PROLONGING_MEDS):
                    credit_a = max(credit_a, w.action_a)
            if rtype == "ServiceRequest" and has_ref(pl):
                if "445118002" in blob or "ecg" in blob:
                    credit_b = max(credit_b, w.action_b)

        elif task_type == "task8":
            # v1_task8: orthopedic referral. Grader requires priority=stat + authoredOn + note text.
            if rtype == "ServiceRequest" and has_ref(pl):
                if "306181000000106" in blob or "orthopedic" in blob:
                    if _active_order(pl) and _has_authored(pl) and _is_stat(pl):
                        credit_a = max(credit_a, w.action_a)

        elif task_type == "task9":
            # v2new_task9: flu vaccine order (CPT 90686).
            if rtype in ("ServiceRequest", "MedicationRequest") and has_ref(pl):
                if "90686" in blob or "influenza" in blob or "flu" in blob:
                    if _active_order(pl) and _has_authored(pl):
                        credit_a = max(credit_a, w.action_a)

        elif task_type == "task10":
            # v1_task10: HbA1C reorder. Grader requires priority=stat + authoredOn.
            if rtype == "ServiceRequest" and has_ref(pl):
                if "4548-4" in blob or "a1c" in blob or "hba1c" in blob:
                    if _active_order(pl) and _has_authored(pl) and _is_stat(pl):
                        credit_a = max(credit_a, w.action_a)

        elif task_type == "v1_task5":
            # v1_task5: Mg replacement. Same content check as new_refsol.v2_task5.
            if rtype == "MedicationRequest" and has_ref(pl):
                if "0338-1715-40" in blob or "magnesium" in blob:
                    if _active_order(pl) and _has_authored(pl) and _has_dosage(pl):
                        credit_a = max(credit_a, w.action_a)

        elif task_type == "v1_task9":
            # v1_task9: K replacement (action_a) + serum-K lab order (action_b).
            if rtype == "MedicationRequest" and has_ref(pl):
                if "40032-917-01" in blob or "potassium" in blob:
                    if _active_order(pl) and _has_authored(pl) and _has_dosage(pl):
                        credit_a = max(credit_a, w.action_a)
            if rtype == "ServiceRequest" and has_ref(pl):
                if "2823-3" in blob or "potassium" in blob or "serum" in blob:
                    if _active_order(pl) and _has_authored(pl):
                        credit_b = max(credit_b, w.action_b)

        elif task_type == "v2_task10":
            # v2new_task10: COVID booster. new_refsol.task10 accepts ServiceRequest or MedicationRequest.
            if rtype in ("ServiceRequest", "MedicationRequest") and has_ref(pl):
                if "covidvaccine" in blob or "covid" in blob:
                    if _active_order(pl) and _has_authored(pl):
                        credit_a = max(credit_a, w.action_a)

    return credit_a, credit_b


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
    v1_refsol=None,
    agent_answer=None,
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

    raw_id = task_spec.get("id", "")
    # Corpus routing: "v1_taskN_M" → v1 refsol; "v2new_taskN_M" → new_refsol;
    # bare "taskN_M"/"v2_taskN_M" → new_refsol (backward compat).
    # ART corpus: tasks carry "ground_truth.task_type" — route to ART grader.
    corpus = "v2new"
    task_id = raw_id
    if raw_id.startswith("v1_"):
        corpus = "v1"
        task_id = raw_id[len("v1_"):]
    elif raw_id.startswith("v2new_"):
        task_id = raw_id[len("v2new_"):]
    elif "ground_truth" in task_spec and task_spec["ground_truth"].get("task_type") in {
        "magnesium_tiered", "potassium_linear", "a1c_timebased",
        "absence_action", "safety_cap", "compound_conditional",
    }:
        corpus = "art"
        task_id = raw_id
    parts = task_id.rsplit("_", 1)
    task_type = parts[0] if len(parts) == 2 and parts[1].isdigit() else task_id
    # v1 corpus: task5 = Mg replacement, task9 = K replacement — differ from v2new.
    # Remap to v1_taskN so their intermediate rewards and GET-allowlists are correct.
    if corpus == "v1" and task_type in {"task4", "task5", "task9"}:
        task_type = f"v1_{task_type}"
    # v2new corpus: task10 = COVID vaccine — route to v2_task10.
    # task5/9 stay as-is (CT+IR and flu vaccine reuse the non-prefixed logic).
    elif corpus == "v2new" and f"v2_{task_type}" in _TASK_ACTION_RTYPES:
        task_type = f"v2_{task_type}"
    mrn = task_spec.get("eval_MRN", "")

    # ── 1. Terminal grader ───────────────────────────────────────────────────
    refsol_pass = False

    def _run_grader() -> bool:
        if corpus == "art":
            try:
                from medagentbench_env.art.grader import grade as _art_grade
                import types as _types
                ns_history = [_types.SimpleNamespace(**msg) for msg in history]
                result_str = json.dumps(agent_answer) if agent_answer is not None else None
                eval_results = _types.SimpleNamespace(history=ns_history, result=result_str)
                case_data = {
                    "id": raw_id,
                    "instruction": task_spec.get("instruction", ""),
                    "context": task_spec.get("context", ""),
                    "sol": task_spec.get("sol", []),
                    "eval_MRN": mrn,
                    "ground_truth": task_spec.get("ground_truth", {}),
                }
                return _art_grade(case_data, eval_results, fhir_base_url) is True
            except Exception as e:
                print(f"[verifier] ART grader error for {raw_id}: {e}")
                return False
        grader_mod = v1_refsol if corpus == "v1" else new_refsol
        if grader_mod is not None:
            grader_fn_name = _GRADER_TASK_REMAP.get(task_type, task_type)
            grader_fn = getattr(grader_mod, grader_fn_name, None)
            if grader_fn is not None:
                import types as _types
                case_data = {
                    "id": task_id,
                    "instruction": task_spec.get("instruction", ""),
                    "context": task_spec.get("context", ""),
                    "sol": task_spec.get("sol", []),
                    "eval_MRN": mrn,
                }
                ns_history = [_types.SimpleNamespace(**msg) for msg in history]
                # v1 refsol reads results.result (JSON-encoded finish answer).
                result_str = json.dumps(agent_answer) if agent_answer is not None else None
                eval_results = _types.SimpleNamespace(history=ns_history, result=result_str)
                try:
                    return grader_fn(case_data, eval_results, fhir_base_url) is True
                except Exception as e:
                    print(f"[verifier] {corpus} refsol grader error for {raw_id}: {e}")
        return _inline_pass(task_type, mrn, history)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_run_grader)
            try:
                refsol_pass = fut.result(timeout=grader_timeout)
            except concurrent.futures.TimeoutError:
                print(f"[verifier] GRADER_TIMEOUT task={task_id} after {grader_timeout}s — terminal=False")
    except Exception as e:
        print(f"[verifier] grader exception for {task_id}: {e}")

    # ── 2. Parse history ─────────────────────────────────────────────────────
    get_urls = _get_urls(history)
    posts = _accepted_posts(history)

    # ── 3. Build reward ──────────────────────────────────────────────────────
    reward = 0.0

    if refsol_pass:
        # Gate full terminal credit on the agent having actually called finish().
        # Without this, silent-action graders return pass=True for "did nothing"
        # rollouts (no finish call), which corrupts the RL signal toward laziness.
        # Half-credit (0.5) if grader passed but the model never declared completion.
        if agent_answer is not None:
            reward += w.terminal
        else:
            reward += 0.5 * w.terminal

    # GET credit: only when agent looked up chart AND placed an accepted order
    # AND actually called finish() with an answer.
    # The agent_answer gate (v27) blocks the GET→POST→never-finish hack, where
    # the model collects shape rewards on a truncated rollout that never
    # produces an answer the strict grader can score.
    if get_urls and posts and agent_answer is not None:
        reward += w.get_credit

    # Redundant GET penalty
    seen: set = set()
    redundant = sum(1 for url in get_urls if url in seen or seen.add(url))  # type: ignore[func-returns-value]
    reward += max(w.redundant_lookup_cap, w.redundant_lookup * redundant)

    # Invalid FHIR call penalty
    reward += w.invalid_fhir * invalid_fhir_count

    # Off-target GET penalty.
    # Patient is a benign meta-query (resolve name+DOB → MRN, fetch demographics)
    # and is exempt globally — penalizing it would punish v1_task1/task2 which
    # legitimately need a Patient GET. Bare-key collisions in ALLOWED_GET_RESOURCES
    # (v1_task1 vs v2-new task1) are also worked around here.
    allowed = ALLOWED_GET_RESOURCES.get(task_type)
    if allowed:
        offtarget = sum(
            1 for url in get_urls
            if (
                (rt := url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]) != "Patient"
                and rt not in allowed
            )
        )
        reward += max(w.offtarget_lookup_cap, w.offtarget_lookup * offtarget)

    # Dense action rewards: partial (right type+patient) and full (code match).
    # Gated on agent_answer (v27): only fire when finish() was called, so RL
    # cannot reinforce truncated POST-only rollouts that the strict grader
    # cannot score.
    if agent_answer is not None:
        credit_a, credit_b = _action_rewards(task_type, mrn, posts, w)
    else:
        credit_a, credit_b = 0.0, 0.0
    reward += credit_a
    reward += credit_b

    # Spurious POST: posted but nothing task-relevant in any POST
    if not refsol_pass and posts and credit_a == 0.0 and credit_b == 0.0:
        reward += w.spurious_post

    # Skip-tool finish: called finish() with no GET or POST — only penalise when
    # wrong (correct no-action branches still pass through refsol and get terminal).
    if _is_finish_no_tools(history) and not refsol_pass:
        reward += w.skip_finish_penalty

    return max(-1.0, min(2.0, reward))
