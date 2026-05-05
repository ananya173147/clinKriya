"""
Shaped reward for MedAgentBench server environment.

Delegates to medagentbench_env.verifier for consistent scoring with training.
All reward weights and action-detection logic are shared — no divergence possible.
"""

from typing import Any, List, Optional

from medagentbench_env.verifier import (
    _action_rewards,
    _accepted_posts,
    _get_urls,
    _is_finish_no_tools,
    ALLOWED_GET_RESOURCES,
    _DEFAULT_WEIGHTS,
    RewardWeights,
)


def compute_shaped_reward(
    task_type: str,
    mrn: str,
    history: List[Any],
    refsol_pass: bool,
    step_count: int = 0,
    max_steps: int = 8,
    invalid_fhir_count: int = 0,
    weights: Optional[RewardWeights] = None,
) -> float:
    """Compute dense shaped reward for one completed episode.

    Parameters
    ----------
    task_type       : canonical task identifier, e.g. "task1" or "v2_task5"
    mrn             : patient MRN string
    history         : list of objects with .role and .content attributes
    refsol_pass     : True if the new_refsol grader accepted the agent's actions
    step_count      : number of steps taken (unused, kept for API compat)
    max_steps       : episode step budget (unused, kept for API compat)
    invalid_fhir_count : number of FHIR calls rejected / returned error
    weights         : optional RewardWeights override

    Returns
    -------
    float in [-1.0, 2.0]
    """
    w = weights if weights is not None else _DEFAULT_WEIGHTS

    # Convert attr-access history items to plain dicts for verifier helpers
    history_dicts = [
        {"role": getattr(m, "role", ""), "content": getattr(m, "content", "") or ""}
        for m in history
    ]

    reward = 0.0

    if refsol_pass:
        reward += w.terminal

    get_urls = _get_urls(history_dicts)
    posts = _accepted_posts(history_dicts)

    # GET credit: only when agent looked up chart AND placed an accepted order
    if get_urls and posts:
        reward += w.get_credit

    # Redundant GET penalty
    seen: set = set()
    redundant = sum(1 for url in get_urls if url in seen or seen.add(url))  # type: ignore[func-returns-value]
    reward += max(w.redundant_lookup_cap, w.redundant_lookup * redundant)

    # Invalid FHIR penalty
    reward += w.invalid_fhir * float(invalid_fhir_count)

    # Off-target GET penalty
    allowed = ALLOWED_GET_RESOURCES.get(task_type)
    if allowed:
        offtarget = sum(
            1 for url in get_urls
            if url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] not in allowed
        )
        reward += max(w.offtarget_lookup_cap, w.offtarget_lookup * offtarget)

    # Dense action rewards (partial + full)
    credit_a, credit_b = _action_rewards(task_type, mrn, posts, w)
    reward += credit_a
    reward += credit_b

    # Spurious POST penalty
    if not refsol_pass and posts and credit_a == 0.0 and credit_b == 0.0:
        reward += w.spurious_post

    # Skip-tool finish penalty
    if _is_finish_no_tools(history_dicts) and not refsol_pass:
        reward += w.skip_finish_penalty

    return max(-1.0, min(2.0, reward))
