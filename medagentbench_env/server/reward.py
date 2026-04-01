"""
Shaped reward for MedAgentBench server environment.

This mirrors the dense reward used in medagentbench_env/train.py so that
offline evaluation and training are scored consistently.

The primary signal is `refsol_pass` (binary, from new_refsol graders).
Dense components give intermediate feedback for tool use quality.
"""

import json
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Reward weights — keep in sync with train.py
# ---------------------------------------------------------------------------
_W_TERMINAL = 1.00        # full credit for passing the refsol grader
_W_GET_CREDIT = 0.20      # agent looked up chart before deciding
_W_REDUNDANT = -0.05      # per repeated identical GET URL
_W_REDUNDANT_CAP = -0.20
_W_INVALID = -0.10        # per rejected FHIR call
_W_OFFTARGET = -0.05      # per GET on a resource irrelevant to the task
_W_OFFTARGET_CAP = -0.20
_W_ACTION_A = 0.25        # primary required action detected in a POST
_W_ACTION_B = 0.25        # secondary required action (task5 / task7 / v2_task9)
_W_SPURIOUS = -0.15       # completely off-target POST(s) when neither A nor B found

# Resources each task type should query (used for off-target penalty)
_TASK_RESOURCES: Dict[str, set] = {
    "task1":    {"Procedure"},
    "task2":    {"MedicationRequest"},
    "task4":    {"Procedure"},
    "task5":    {"Condition", "Procedure"},
    "task6":    {"Observation"},
    "task7":    {"Observation", "MedicationRequest"},
    "task8":    {"MedicationRequest"},
    "task9":    {"Procedure"},
    "task10":   {"Procedure"},
    "v2_task5": {"Observation"},
    "v2_task9": {"Observation"},
    "v2_task10": {"Observation"},
}

_NON_ACTIVE = {"stopped", "cancelled", "completed", "entered-in-error"}
_QT_MEDS = (
    "ondansetron", "prochlorperazine", "haloperidol", "quetiapine",
    "olanzapine", "risperidone", "ziprasidone", "clozapine", "chlorpromazine",
)
_ANTICOAG = (
    "heparin", "enoxaparin", "lovenox", "fondaparinux",
    "rivaroxaban", "apixaban", "dabigatran", "warfarin",
    "tinzaparin", "dalteparin",
)


def compute_shaped_reward(
    task_type: str,
    history: List[Any],
    refsol_pass: bool,
    step_count: int = 0,
    max_steps: int = 8,
    invalid_fhir_count: int = 0,
) -> float:
    """
    Compute dense shaped reward for one completed episode.

    Parameters
    ----------
    task_type       : canonical task identifier, e.g. "task1" or "v2_task5"
    history         : list of objects with .role and .content attributes
    refsol_pass     : True if the new_refsol grader accepted the agent's actions
    step_count      : number of steps taken
    max_steps       : episode step budget
    invalid_fhir_count : number of FHIR calls rejected by the server

    Returns
    -------
    float in [-1.0, 2.0]
    """
    reward = 0.0

    # ── 1 · Terminal success ──────────────────────────────────────────────
    if refsol_pass:
        reward += _W_TERMINAL

    # ── 2 · Parse history ────────────────────────────────────────────────
    get_urls: List[str] = []
    posts: List[Dict] = []

    for idx, msg in enumerate(history):
        if msg.role != "agent":
            continue
        content = msg.content or ""
        if content.startswith("GET "):
            parts = content.split()
            if len(parts) > 1:
                get_urls.append(parts[1])
        elif content.startswith("POST "):
            # Only count accepted POSTs
            if idx + 1 < len(history) and "POST request accepted" in (history[idx + 1].content or ""):
                try:
                    payload = json.loads(content.split("\n", 1)[1]) if "\n" in content else {}
                    posts.append(payload)
                except Exception:
                    pass

    # ── 3 · GET credit (agent looked at chart before deciding) ───────────
    if get_urls:
        reward += _W_GET_CREDIT

    # ── 4 · Redundant GET penalty ─────────────────────────────────────────
    seen: set = set()
    redundant = 0
    for url in get_urls:
        if url in seen:
            redundant += 1
        seen.add(url)
    r_red = max(_W_REDUNDANT_CAP, _W_REDUNDANT * redundant)
    reward += r_red

    # ── 5 · Invalid FHIR penalty ──────────────────────────────────────────
    reward += _W_INVALID * float(invalid_fhir_count)

    # ── 6 · Off-target GET penalty ────────────────────────────────────────
    allowed = _TASK_RESOURCES.get(task_type)
    if allowed:
        offtarget = 0
        for url in get_urls:
            base = url.split("?", 1)[0].rstrip("/")
            resource = base.rsplit("/", 1)[-1]
            if resource not in allowed:
                offtarget += 1
        reward += max(_W_OFFTARGET_CAP, _W_OFFTARGET * offtarget)

    # ── 7 · Required-action partial credit ───────────────────────────────
    found_a = False
    found_b = False

    for pl in posts:
        blob = json.dumps(pl).lower()
        rtype = pl.get("resourceType", "")
        status = pl.get("status", "").lower()

        if task_type == "task1":
            if rtype == "ServiceRequest" and "74177" in blob:
                found_a = True

        elif task_type == "task2":
            if rtype == "MedicationRequest" and any(k in blob for k in _ANTICOAG):
                found_a = True

        elif task_type == "task4":
            if rtype == "ServiceRequest" and ("nur1373" in blob or "catheter" in blob):
                found_a = True

        elif task_type == "task5":
            if rtype == "ServiceRequest":
                if "74177" in blob:
                    found_a = True
                if "con417" in blob or "interventional radiology" in blob:
                    found_b = True

        elif task_type == "task6":
            if rtype == "MedicationRequest" and "levothyroxine" in blob:
                found_a = True
            if rtype == "ServiceRequest" and ("tsh" in blob or "ft4" in blob or "thyroid" in blob):
                found_b = True

        elif task_type == "task7":
            if rtype == "MedicationRequest" and status in _NON_ACTIVE and any(m in blob for m in _QT_MEDS):
                found_a = True
            if rtype == "ServiceRequest" and ("445118002" in blob or "ecg" in blob or "electrocardiogram" in blob):
                found_b = True

        elif task_type == "task8":
            if rtype == "MedicationRequest" and "naloxone" in blob:
                found_a = True

        elif task_type == "task9":
            if rtype == "ServiceRequest" and ("90686" in blob or "influenza" in blob or "flu" in blob):
                found_a = True

        elif task_type == "task10":
            if rtype in {"ServiceRequest", "MedicationRequest"} and "covid" in blob:
                found_a = True

        elif task_type == "v2_task5":
            if rtype == "MedicationRequest" and ("magnesium" in blob or "0338-1715-40" in blob):
                found_a = True

        elif task_type == "v2_task9":
            if rtype == "MedicationRequest" and ("potassium" in blob or "40032-917-01" in blob):
                found_a = True
            if rtype == "ServiceRequest" and ("potassium" in blob or "2823-3" in blob):
                found_b = True

        elif task_type == "v2_task10":
            if rtype == "ServiceRequest" and ("4548-4" in blob or "a1c" in blob or "hba1c" in blob):
                found_a = True

    if found_a:
        reward += _W_ACTION_A
    if found_b:
        reward += _W_ACTION_B

    # ── 8 · Spurious-POST penalty: posted but nothing task-relevant ───────
    if not refsol_pass and posts and not found_a and not found_b:
        reward += _W_SPURIOUS

    return max(-1.0, min(2.0, reward))
