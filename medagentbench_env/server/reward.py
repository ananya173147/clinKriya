"""
Shaped reward verifier for MedAgentBench RL training.

Covers the 5 task types served by the HF Space (stratified_benchmark.json):
  task3  – Vital-signs documentation: record blood pressure (always-action)
  task5  – CT Oncology Follow-up: order CT & IR referral if scan overdue (action-required)
  task7  – QTc Interval Review: ECG + med discontinuation if QTc prolonged (action-required)
  task8  – Specialist Referral: orthopedic surgery ServiceRequest (always-action)
  task10 – Chronic Disease Review: A1C ServiceRequest if lab overdue (action-required)

Reward components (summed, range ~-0.3 to 1.0):
  - Correctness  (0.0 – 0.4): refsol pass/fail
  - Structure    (0.0 – 0.2): right endpoint + right resource type
  - Field credit (0.0 – 0.1): partial credit for correct payload fields
  - Patient ref  (0.0 – 0.1): correct patient MRN in payload
  - Efficiency   (0.0 – 0.1): fewer steps = bonus
  - Redundancy   (-0.1/call):  penalty per extra POST/GET
  - Format       (-0.1):       penalty for invalid action format
  - Completion   (+0.05):      agent called FINISH
"""

import json
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def _extract_posts_from_history(history: list) -> List[Tuple[str, Dict]]:
    posts = []
    for idx, msg in enumerate(history):
        if msg.role == "agent" and "POST" in msg.content:
            if idx + 1 < len(history) and "POST request accepted" in history[idx + 1].content:
                try:
                    raw = msg.content
                    url = raw.split("\n")[0][4:].strip()
                    payload = json.loads("\n".join(raw.split("\n")[1:]))
                    posts.append((url, payload))
                except Exception:
                    pass
    return posts


def _count_get_requests(history: list) -> int:
    return sum(1 for msg in history if msg.role == "agent" and msg.content.strip().startswith("GET"))


def _has_any_post(history: list) -> bool:
    return any(msg.role == "agent" and "POST" in msg.content for msg in history)


# ---------------------------------------------------------------------------
# Task-specific field checkers (return 0.0 – 1.0 fraction of correct fields)
# ---------------------------------------------------------------------------

def _check_task3_fields(payload: Dict, case_data: Dict) -> float:
    """Task 3: Record blood pressure Observation (always-action)."""
    checks = []
    checks.append(payload.get("resourceType") == "Observation")
    cats = payload.get("category", [])
    if cats and isinstance(cats, list):
        codings = cats[0].get("coding", [])
        checks.append(bool(codings) and codings[0].get("code") == "vital-signs")
    else:
        checks.append(False)
    checks.append(payload.get("code", {}).get("text") == "BP")
    checks.append("2023-11-13T10:15" in str(payload.get("effectiveDateTime", "")))
    checks.append(payload.get("status") == "final")
    checks.append("118/77" in str(payload.get("valueString", "")))
    expected_ref = f"Patient/{case_data['eval_MRN']}"
    checks.append(payload.get("subject", {}).get("reference") == expected_ref)
    return sum(checks) / len(checks)


def _check_task5_fields(payload: Dict, case_data: Dict) -> float:
    """Task 5: CT Abdomen/Pelvis ServiceRequest (CPT 74177) or IR referral (CON417)."""
    checks = []
    checks.append(payload.get("resourceType") in {"ServiceRequest", "MedicationRequest"})
    checks.append(payload.get("status") == "active")
    checks.append(payload.get("intent") == "order")
    code_str = str(payload.get("code", {}))
    checks.append("74177" in code_str or "CON417" in code_str)
    expected_ref = f"Patient/{case_data['eval_MRN']}"
    checks.append(payload.get("subject", {}).get("reference") == expected_ref)
    return sum(checks) / len(checks)


def _check_task7_fields(payload: Dict, case_data: Dict) -> float:
    """Task 7: ECG ServiceRequest (SNOMED 445118002) or QT-prolonging med discontinuation."""
    checks = []
    checks.append(payload.get("resourceType") in {"ServiceRequest", "MedicationRequest"})
    expected_ref = f"Patient/{case_data['eval_MRN']}"
    checks.append(payload.get("subject", {}).get("reference") == expected_ref)
    text_blob = str(payload).lower()
    ecg_ok = "445118002" in text_blob or "ecg" in text_blob or "electrocardiogram" in text_blob
    stop_status = payload.get("status", "").lower() in {"stopped", "cancelled", "completed", "entered-in-error"}
    qt_drug_ok = any(w in text_blob for w in [
        "ondansetron", "prochlorperazine", "haloperidol", "quetiapine",
        "olanzapine", "risperidone", "ziprasidone", "clozapine", "chlorpromazine",
    ])
    checks.append(ecg_ok or (stop_status and qt_drug_ok))
    return sum(checks) / len(checks)


def _check_task8_fields(payload: Dict, case_data: Dict) -> float:
    """Task 8: Orthopedic surgery ServiceRequest (SNOMED 306181000000106, SBAR note)."""
    checks = []
    checks.append(payload.get("resourceType") == "ServiceRequest")
    code_codings = payload.get("code", {}).get("coding", [])
    if code_codings:
        checks.append(code_codings[0].get("code") == "306181000000106")
        checks.append(code_codings[0].get("system") == "http://snomed.info/sct")
    else:
        checks.append(False)
        checks.append(False)
    checks.append("2023-11-13T10:15" in str(payload.get("authoredOn", "")))
    checks.append(payload.get("status") == "active")
    checks.append(payload.get("intent") == "order")
    checks.append(payload.get("priority") == "stat")
    expected_ref = f"Patient/{case_data['eval_MRN']}"
    checks.append(payload.get("subject", {}).get("reference") == expected_ref)
    note = payload.get("note", {})
    if isinstance(note, list):
        note_text = " ".join(str(n.get("text", "")) if isinstance(n, dict) else str(n) for n in note)
    elif isinstance(note, dict):
        note_text = str(note.get("text", ""))
    else:
        note_text = str(note)
    checks.append("ACL tear" in note_text or "orthopedic" in note_text.lower())
    return sum(checks) / len(checks)


def _check_task10_post_fields(payload: Dict, case_data: Dict) -> float:
    """Task 10: HbA1C ServiceRequest (LOINC 4548-4)."""
    checks = []
    checks.append(payload.get("resourceType") == "ServiceRequest")
    code_codings = payload.get("code", {}).get("coding", [])
    if code_codings:
        checks.append(code_codings[0].get("code") == "4548-4")
        checks.append(code_codings[0].get("system") == "http://loinc.org")
    else:
        checks.append(False)
        checks.append(False)
    checks.append("2023-11-13T10:15" in str(payload.get("authoredOn", "")))
    checks.append(payload.get("status") == "active")
    checks.append(payload.get("intent") == "order")
    checks.append(payload.get("priority") == "stat")
    expected_ref = f"Patient/{case_data['eval_MRN']}"
    checks.append(payload.get("subject", {}).get("reference") == expected_ref)
    return sum(checks) / len(checks)


# ---------------------------------------------------------------------------
# Expected endpoint + field checkers per in-scope task type
# ---------------------------------------------------------------------------

_EXPECTED_ENDPOINTS = {
    "task3":  "Observation",
    "task5":  "ServiceRequest",
    "task7":  "ServiceRequest",
    "task8":  "ServiceRequest",
    "task10": "ServiceRequest",
}

_FIELD_CHECKERS = {
    "task3":  _check_task3_fields,
    "task5":  _check_task5_fields,
    "task7":  _check_task7_fields,
    "task8":  _check_task8_fields,
    "task10": _check_task10_post_fields,
}

# task7 may need 2 POSTs (ECG order + med discontinuation)
_EXPECTED_POST_COUNT = {
    "task7": 2,
}


# ---------------------------------------------------------------------------
# Main shaped reward function
# ---------------------------------------------------------------------------

def compute_shaped_reward(
    task_type: str,
    case_data: Dict[str, Any],
    history: list,
    agent_answer: Optional[List[Any]],
    fhir_api_base: str,
    step_count: int,
    max_steps: int,
    refsol_pass: bool,
    benchmark_type: str = "",
) -> float:
    """Compute a shaped reward for one completed episode.

    Args:
        task_type: e.g. "task3", "task8"
        case_data: Task definition dict (must include eval_MRN)
        history: Chat history (list of objects with .role and .content)
        agent_answer: The agent's FINISH answer list (or None)
        fhir_api_base: FHIR server base URL
        step_count: Number of steps the agent took
        max_steps: Maximum allowed steps
        refsol_pass: Whether the binary refsol grader passed
        benchmark_type: "always-action", "action-required", or "no-action-required"

    Returns:
        Float reward, clamped to [-0.3, 1.0]
    """
    reward = 0.0
    posts = _extract_posts_from_history(history)
    num_gets = _count_get_requests(history)
    has_post = _has_any_post(history)

    no_action_task = benchmark_type == "no-action-required"
    action_required = not no_action_task

    # ---- 1. Binary correctness (0.0 or 0.4) ----
    if refsol_pass:
        reward += 0.4

    # ---- 2. Structural + field correctness (0.0 – 0.2) ----
    expected_endpoint = _EXPECTED_ENDPOINTS.get(task_type)

    if action_required and posts:
        post_url, payload = posts[0]
        if expected_endpoint and expected_endpoint in post_url:
            reward += 0.05
        if payload.get("resourceType") == expected_endpoint:
            reward += 0.05
        checker = _FIELD_CHECKERS.get(task_type)
        if checker:
            reward += 0.1 * checker(payload, case_data)

    elif no_action_task and not has_post:
        reward += 0.15

    # ---- 3. Patient reference in first POST (0.0 or 0.1) ----
    if posts:
        _, payload = posts[0]
        expected_ref = f"Patient/{case_data.get('eval_MRN', '')}"
        if payload.get("subject", {}).get("reference") == expected_ref:
            reward += 0.1

    # ---- 4. Efficiency bonus (0.0 – 0.1) ----
    if step_count > 0 and max_steps > 0:
        reward += 0.1 * max(0.0, 1.0 - (step_count / max_steps))

    # ---- 5. Redundancy penalties ----
    if action_required:
        expected_posts = _EXPECTED_POST_COUNT.get(task_type, 1)
        extra_posts = max(0, len(posts) - expected_posts)
        reward -= 0.1 * extra_posts
    else:
        if has_post:
            reward -= 0.15

    if num_gets > 3:
        reward -= 0.05 * (num_gets - 3)

    # ---- 6. Format penalty ----
    for msg in history:
        if msg.role == "agent":
            content = msg.content.strip()
            if not (content.startswith("GET") or content.startswith("POST") or content.startswith("FINISH")):
                reward -= 0.1
                break

    # ---- 7. Completion bonus ----
    if agent_answer is not None:
        reward += 0.05

    return max(-0.3, min(1.0, reward))
