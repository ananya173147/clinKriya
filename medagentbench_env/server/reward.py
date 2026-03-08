"""
Shaped reward verifier for MedAgentBench RL training.

Provides dense, step-aware rewards instead of binary pass/fail.
Scores partial credit for correct fields, penalizes redundant/wrong
calls, and rewards efficiency.

Reward components (summed, range ~-0.3 to 1.0):
  - Correctness  (0.0 – 0.4): refsol pass/fail + partial field credit
  - Structure    (0.0 – 0.2): right endpoint, right resource type
  - Patient ref  (0.0 – 0.1): correct patient MRN in payload
  - Efficiency   (0.0 – 0.1): fewer steps = bonus
  - Redundancy   (-0.1/call):  penalty per unnecessary POST/GET
  - Format       (-0.1):       penalty for invalid action format
"""

import json
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Post extraction (mirrors refsol logic)
# ---------------------------------------------------------------------------

def _extract_posts_from_history(history: list) -> List[Tuple[str, Dict]]:
    """Extract successful POST requests from chat history."""
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
    """Count GET requests made by the agent."""
    return sum(1 for msg in history if msg.role == "agent" and msg.content.strip().startswith("GET"))


def _has_any_post(history: list) -> bool:
    for msg in history:
        if msg.role == "agent" and "POST" in msg.content:
            return True
    return False


# ---------------------------------------------------------------------------
# Task-specific field checkers (return fraction of correct fields)
# ---------------------------------------------------------------------------

def _check_task3_fields(payload: Dict, case_data: Dict) -> float:
    """Task 3: Record blood pressure observation. Returns 0.0-1.0."""
    checks = []
    checks.append(payload.get("resourceType") == "Observation")
    # Category
    cats = payload.get("category", [])
    if cats and isinstance(cats, list) and len(cats) > 0:
        codings = cats[0].get("coding", [])
        if codings:
            checks.append(codings[0].get("code") == "vital-signs")
        else:
            checks.append(False)
    else:
        checks.append(False)
    # Code
    checks.append(payload.get("code", {}).get("text") == "BP")
    # DateTime
    checks.append("2023-11-13T10:15" in str(payload.get("effectiveDateTime", "")))
    # Status
    checks.append(payload.get("status") == "final")
    # Value
    checks.append("118/77" in str(payload.get("valueString", "")))
    # Subject
    expected_ref = f"Patient/{case_data['eval_MRN']}"
    checks.append(payload.get("subject", {}).get("reference") == expected_ref)
    return sum(checks) / len(checks) if checks else 0.0


def _check_task8_fields(payload: Dict, case_data: Dict) -> float:
    """Task 8: Orthopedic referral ServiceRequest. Returns 0.0-1.0."""
    checks = []
    checks.append(payload.get("resourceType") == "ServiceRequest")
    # Code
    code_codings = payload.get("code", {}).get("coding", [])
    if code_codings:
        checks.append(code_codings[0].get("code") == "306181000000106")
        checks.append(code_codings[0].get("system") == "http://snomed.info/sct")
    else:
        checks.append(False)
        checks.append(False)
    # Date
    checks.append("2023-11-13T10:15" in str(payload.get("authoredOn", "")))
    # Status + intent + priority
    checks.append(payload.get("status") == "active")
    checks.append(payload.get("intent") == "order")
    checks.append(payload.get("priority") == "stat")
    # Subject
    expected_ref = f"Patient/{case_data['eval_MRN']}"
    checks.append(payload.get("subject", {}).get("reference") == expected_ref)
    # Note (SBAR comment)
    note = payload.get("note", {})
    if isinstance(note, list):
        note_text = " ".join(str(n.get("text", "")) if isinstance(n, dict) else str(n) for n in note)
    elif isinstance(note, dict):
        note_text = str(note.get("text", ""))
    else:
        note_text = str(note)
    checks.append("ACL tear" in note_text or "orthopedic" in note_text.lower())
    return sum(checks) / len(checks) if checks else 0.0


def _check_task10_post_fields(payload: Dict, case_data: Dict) -> float:
    """Task 10: A1C ServiceRequest. Returns 0.0-1.0."""
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
    return sum(checks) / len(checks) if checks else 0.0


# ---------------------------------------------------------------------------
# Expected endpoint per task type
# ---------------------------------------------------------------------------

_EXPECTED_ENDPOINTS = {
    "task3": "Observation",
    "task8": "ServiceRequest",
    "task10": "ServiceRequest",
}

_FIELD_CHECKERS = {
    "task3": _check_task3_fields,
    "task8": _check_task8_fields,
    "task10": _check_task10_post_fields,
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
        task_type: e.g. "task3", "task8", "task10"
        case_data: Task definition dict
        history: Chat history (list of objects with .role, .content)
        agent_answer: The agent's FINISH answer list (or None)
        fhir_api_base: FHIR server base URL
        step_count: Number of steps the agent took
        max_steps: Maximum allowed steps
        refsol_pass: Whether the binary refsol grader passed
        benchmark_type: "always-action", "action-required", "no-action-required"

    Returns:
        Float reward, roughly in range [-0.3, 1.0]
    """
    reward = 0.0
    posts = _extract_posts_from_history(history)
    num_gets = _count_get_requests(history)
    has_post = _has_any_post(history)

    # ---- 1. Binary correctness (0.0 or 0.4) ----
    if refsol_pass:
        reward += 0.4

    # ---- 2. Structural correctness of POSTs (0.0 – 0.2) ----
    expected_endpoint = _EXPECTED_ENDPOINTS.get(task_type)
    action_required = benchmark_type in ("always-action", "action-required")

    if action_required and posts:
        # Check if the POST hit the right endpoint
        post_url, payload = posts[0]
        if expected_endpoint and expected_endpoint in post_url:
            reward += 0.05  # Correct endpoint
        if payload.get("resourceType") == expected_endpoint:
            reward += 0.05  # Correct resourceType

        # Field-level partial credit (0.0 – 0.1)
        checker = _FIELD_CHECKERS.get(task_type)
        if checker:
            field_score = checker(payload, case_data)
            reward += 0.1 * field_score

    elif not action_required and not has_post:
        # Correctly did nothing — structural bonus
        reward += 0.15

    # ---- 3. Patient reference (0.0 or 0.1) ----
    if posts:
        post_url, payload = posts[0]
        expected_ref = f"Patient/{case_data.get('eval_MRN', '')}"
        actual_ref = payload.get("subject", {}).get("reference", "")
        if actual_ref == expected_ref:
            reward += 0.1

    # ---- 4. Efficiency bonus (0.0 – 0.1) ----
    # Fewer steps relative to max = better
    if step_count > 0 and max_steps > 0:
        efficiency = max(0.0, 1.0 - (step_count / max_steps))
        reward += 0.1 * efficiency

    # ---- 5. Redundancy penalties ----
    if action_required:
        # Penalize extra POSTs beyond what's needed (usually 1)
        expected_posts = 1
        extra_posts = max(0, len(posts) - expected_posts)
        reward -= 0.1 * extra_posts
    else:
        # No action needed — penalize any POST
        if has_post:
            reward -= 0.15

    # Penalize excessive GET requests (more than 3 is likely redundant)
    if num_gets > 3:
        reward -= 0.05 * (num_gets - 3)

    # ---- 6. Format penalty ----
    # Check if agent ever produced an invalid action (non GET/POST/FINISH)
    for msg in history:
        if msg.role == "agent":
            content = msg.content.strip()
            if not (content.startswith("GET") or content.startswith("POST") or content.startswith("FINISH")):
                reward -= 0.1
                break  # Only penalize once

    # ---- 7. Completion bonus ----
    # Agent called FINISH (not timed out)
    if agent_answer is not None:
        reward += 0.05

    # Clamp to reasonable range
    return max(-0.3, min(1.0, reward))
