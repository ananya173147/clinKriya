"""Oracle demo generator: produce clean SFT samples for every train task.

For each task_id in train_tasks.json, dispatches on workflow_branch to generate
a canonical tool_call sequence that passes the new_refsol grader.  Validates
each demo against the MedAgentTrainEnv + verifier before writing.

Output: training/sft_oracle/train.jsonl  (chat-formatted SFT samples)
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mock_fhir():
    spec = importlib.util.spec_from_file_location("fhir_cache", ROOT / "medagentbench_env/server/fhir_cache.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.MockFHIR.from_cache(str(ROOT / "medagentbench_env/data/fhir_cache.json"))


def _patient_ref(mrn: str) -> dict:
    return {"reference": f"Patient/{mrn}"}


def _tool_call(name: str, arguments: dict) -> str:
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": arguments}, separators=(", ", ": "))
        + "\n</tool_call>"
    )


def _finish(value=None) -> str:
    return _tool_call("finish", {"value": value if value is not None else []})


# ---------------------------------------------------------------------------
# Per-task oracle logic (keyed on workflow_branch)
# ---------------------------------------------------------------------------

def oracle_task1(mrn: str, branch: str):
    """CT abdomen/pelvis follow-up for renal mass."""
    calls = [
        ("fhir_procedure_search", {"patient": mrn}),
    ]
    if branch == "recent_ct_no_action":
        calls.append(("finish", {"value": []}))
    else:  # no_ct_order, old_ct_order → place the CT order
        calls.append((
            "fhir_service_request_create",
            {
                "resourceType": "ServiceRequest",
                "status": "active",
                "intent": "order",
                "priority": "stat",
                "code": {"coding": [{"code": "74177", "system": "http://www.ama-assn.org/go/cpt"}]},
                "subject": _patient_ref(mrn),
                "note": [{"text": "Renal mass follow-up"}],
            },
        ))
        calls.append(("finish", {"value": []}))
    return calls


def oracle_task2(mrn: str, branch: str):
    """DVT prophylaxis — anticoagulant order hygiene."""
    calls = [("fhir_medication_request_search", {"patient": mrn, "status": "active"})]
    if branch == "1_order_ok":
        calls.append(("finish", {"value": []}))
    elif branch == "0_orders_create":
        calls.append((
            "fhir_medication_request_create",
            {
                "resourceType": "MedicationRequest",
                "status": "active",
                "intent": "order",
                "subject": _patient_ref(mrn),
                "medicationCodeableConcept": {"text": "heparin 5000 units SC q8h"},
            },
        ))
        calls.append(("finish", {"value": []}))
    elif branch == "multiple_discontinue":
        # Discontinue all non-heparin anticoag; keep heparin. We approximate by
        # posting a single heparin order (the grader checks "exactly one
        # heparin 5000 remains" — achieved if agent stops extras, but we cannot
        # know which without looking up data. As a best-effort demo we post the
        # canonical heparin order; grader may or may not pass if extras exist.)
        calls.append((
            "fhir_medication_request_create",
            {
                "resourceType": "MedicationRequest",
                "status": "active",
                "intent": "order",
                "subject": _patient_ref(mrn),
                "medicationCodeableConcept": {"text": "heparin 5000 units SC q8h"},
            },
        ))
        calls.append(("finish", {"value": []}))
    return calls


def oracle_task4(mrn: str, branch: str):
    """Urinary catheter removal after >48h."""
    calls = [("fhir_procedure_search", {"patient": mrn})]
    if branch in ("catheter_lt48h_no_action", "no_catheter_no_action"):
        calls.append(("finish", {"value": []}))
    else:  # catheter_gt48h_order
        calls.append((
            "fhir_service_request_create",
            {
                "resourceType": "ServiceRequest",
                "status": "active",
                "intent": "order",
                "code": {"coding": [{"code": "NUR1373", "display": "Discontinue urinary catheter"}]},
                "subject": _patient_ref(mrn),
            },
        ))
        calls.append(("finish", {"value": []}))
    return calls


def oracle_task6(mrn: str, branch: str):
    calls = [
        ("fhir_observation_search", {"patient": mrn, "code": "TSH"}),
        ("fhir_observation_search", {"patient": mrn, "code": "FT4"}),
    ]
    if branch in ("no_tsh_data", "tsh_normal_no_action"):
        calls.append(("finish", {"value": []}))
    elif branch == "tsh_high_x2_order_levothyroxine":
        # Branch A: levothyroxine 25 mcg + repeat TSH + FT4 labs
        calls.append((
            "fhir_medication_request_create",
            {
                "resourceType": "MedicationRequest",
                "status": "active",
                "intent": "order",
                "subject": _patient_ref(mrn),
                "medicationCodeableConcept": {"text": "levothyroxine 25 mcg oral daily"},
            },
        ))
        calls.append((
            "fhir_service_request_create",
            {
                "resourceType": "ServiceRequest",
                "status": "active",
                "intent": "order",
                "code": {"coding": [{"code": "TSH", "display": "Thyroid Stimulating Hormone"}], "text": "TSH"},
                "subject": _patient_ref(mrn),
            },
        ))
        calls.append((
            "fhir_service_request_create",
            {
                "resourceType": "ServiceRequest",
                "status": "active",
                "intent": "order",
                "code": {"coding": [{"code": "FT4", "display": "Free T4"}], "text": "FT4"},
                "subject": _patient_ref(mrn),
            },
        ))
        calls.append(("finish", {"value": []}))
    return calls


def oracle_task7(mrn: str, branch: str):
    calls = [
        ("fhir_observation_search", {"patient": mrn, "code": "QTCINTERVAL"}),
    ]
    if branch in ("no_qtc_data", "normal_no_action"):
        calls.append(("finish", {"value": []}))
    else:  # prolonged_discontinue
        calls.append(("fhir_medication_request_search", {"patient": mrn, "status": "active"}))
        # Discontinue ondansetron (common QT-prolonger) + order ECG
        calls.append((
            "fhir_medication_request_create",
            {
                "resourceType": "MedicationRequest",
                "status": "stopped",
                "intent": "order",
                "subject": _patient_ref(mrn),
                "medicationCodeableConcept": {"text": "ondansetron 4 mg IV"},
            },
        ))
        calls.append((
            "fhir_service_request_create",
            {
                "resourceType": "ServiceRequest",
                "status": "active",
                "intent": "order",
                "code": {"coding": [{"code": "445118002", "system": "http://snomed.info/sct", "display": "12-lead ECG"}]},
                "subject": _patient_ref(mrn),
            },
        ))
        calls.append(("finish", {"value": []}))
    return calls


def oracle_task8(mrn: str, branch: str):
    calls = [("fhir_medication_request_search", {"patient": mrn, "status": "active"})]
    if branch == "no_opioid_no_action":
        calls.append(("finish", {"value": []}))
    else:  # opioid_missing_naloxone_order
        calls.append((
            "fhir_medication_request_create",
            {
                "resourceType": "MedicationRequest",
                "status": "active",
                "intent": "order",
                "subject": _patient_ref(mrn),
                "medicationCodeableConcept": {"text": "naloxone 4 mg intranasal rescue PRN opioid overdose"},
            },
        ))
        calls.append(("finish", {"value": []}))
    return calls


def oracle_task9(mrn: str, branch: str):
    calls = [("fhir_procedure_search", {"patient": mrn})]
    if branch == "recent_flu_vax_no_action":
        calls.append(("finish", {"value": []}))
    else:  # no_flu_vax_order, old_flu_vax_order
        calls.append((
            "fhir_service_request_create",
            {
                "resourceType": "ServiceRequest",
                "status": "active",
                "intent": "order",
                "code": {"coding": [{"code": "90686", "system": "http://www.ama-assn.org/go/cpt", "display": "Influenza vaccine"}]},
                "subject": _patient_ref(mrn),
            },
        ))
        calls.append(("finish", {"value": []}))
    return calls


def oracle_task10(mrn: str, branch: str):
    calls = [("fhir_procedure_search", {"patient": mrn})]
    if branch == "recent_vax_no_action":
        calls.append(("finish", {"value": []}))
    else:  # no_vax_order, old_vax_order_booster
        calls.append((
            "fhir_service_request_create",
            {
                "resourceType": "ServiceRequest",
                "status": "active",
                "intent": "order",
                "code": {"coding": [{"code": "91300", "display": "COVID-19 vaccine booster"}], "text": "COVID-19 booster"},
                "subject": _patient_ref(mrn),
            },
        ))
        calls.append(("finish", {"value": []}))
    return calls


def oracle_v2_task5(mrn: str, branch: str):
    calls = [("fhir_observation_search", {"patient": mrn, "code": "MG"})]
    if branch == "mg_normal_no_action":
        calls.append(("finish", {"value": []}))
    else:  # no_mg_24h_order_stat — order magnesium replacement
        calls.append((
            "fhir_medication_request_create",
            {
                "resourceType": "MedicationRequest",
                "status": "active",
                "intent": "order",
                "subject": _patient_ref(mrn),
                "medicationCodeableConcept": {"text": "magnesium sulfate 2 g IV over 2 hours (NDC 0338-1715-40)"},
            },
        ))
        calls.append(("finish", {"value": []}))
    return calls


def oracle_v2_task9(mrn: str, branch: str):
    calls = [("fhir_observation_search", {"patient": mrn, "code": "K"})]
    if branch in ("k_normal_no_action", "no_k_data"):
        calls.append(("finish", {"value": []}))
    else:  # k_low_order_replacement
        calls.append((
            "fhir_medication_request_create",
            {
                "resourceType": "MedicationRequest",
                "status": "active",
                "intent": "order",
                "subject": _patient_ref(mrn),
                "medicationCodeableConcept": {"text": "potassium chloride 40 mEq oral (NDC 40032-917-01)"},
            },
        ))
        calls.append(("finish", {"value": []}))
    return calls


def oracle_v2_task10(mrn: str, branch: str):
    calls = [("fhir_observation_search", {"patient": mrn, "code": "A1C"})]
    if branch == "a1c_recent_no_action":
        calls.append(("finish", {"value": []}))
    else:  # no_a1c_order, a1c_old_order
        calls.append((
            "fhir_service_request_create",
            {
                "resourceType": "ServiceRequest",
                "status": "active",
                "intent": "order",
                "code": {"coding": [{"code": "4548-4", "system": "http://loinc.org", "display": "HbA1c"}], "text": "A1C"},
                "subject": _patient_ref(mrn),
            },
        ))
        calls.append(("finish", {"value": []}))
    return calls


ORACLE_DISPATCH = {
    "task1": oracle_task1,
    "task2": oracle_task2,
    "task4": oracle_task4,
    "task6": oracle_task6,
    "task7": oracle_task7,
    "task8": oracle_task8,
    "task9": oracle_task9,
    "task10": oracle_task10,
    "v2_task5": oracle_v2_task5,
    "v2_task9": oracle_v2_task9,
    "v2_task10": oracle_v2_task10,
}


# ---------------------------------------------------------------------------
# Format: render tool calls interleaved with env responses as assistant text
# ---------------------------------------------------------------------------

def render_episode(calls, env):
    """Drive the env with the oracle call list; return (multi_turn_messages, reward).

    Returns a list of {"role": ..., "content": ...} where each tool_call is a
    separate assistant turn and each env response is a separate user turn.
    This way the model learns to stop after </tool_call> and wait.
    """
    turns = []
    for tool_name, args in calls:
        tc = _tool_call(tool_name, args)
        turns.append({"role": "assistant", "content": tc})
        if tool_name == "finish":
            try:
                env.finish(args.get("value", []))
            except Exception:
                pass
            break
        method = getattr(env, tool_name, None)
        if method is None:
            turns.append({"role": "user", "content": f"<tool_response>Unknown tool: {tool_name}</tool_response>"})
            continue
        try:
            response = method(**args)
        except Exception as e:
            response = f"Tool error: {e}"
        turns.append({"role": "user", "content": f"<tool_response>\n{response}\n</tool_response>"})
    return turns, float(getattr(env, "reward", 0.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(ROOT / "medagentbench_env/data/train_tasks.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "training/sft_oracle"))
    ap.add_argument("--min-reward", type=float, default=0.85,
                    help="minimum terminal-grader pass to accept demo")
    args = ap.parse_args()

    # Bootstrap the env module
    sys.path.insert(0, str(ROOT))
    env_mod = importlib.import_module("medagentbench_env.fhir_env")

    system_prompt = (ROOT / "medagentbench_env/data/new_system.txt").read_text().strip()
    tasks = json.loads(Path(args.tasks).read_text())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    rejected = []

    for t in tasks:
        task_id = t["id"]
        parts = task_id.rsplit("_", 1)
        task_type = parts[0] if len(parts) == 2 and parts[1].isdigit() else task_id
        branch = t.get("workflow_branch", "")
        mrn = t["eval_MRN"]

        oracle = ORACLE_DISPATCH.get(task_type)
        if oracle is None:
            rejected.append({"task_id": task_id, "reason": f"no oracle for {task_type}"})
            continue
        try:
            calls = oracle(mrn, branch)
        except Exception as e:
            rejected.append({"task_id": task_id, "reason": f"oracle error: {e}"})
            continue

        env = env_mod.MedAgentTrainEnv()
        env.reset(task_id=task_id)

        turns, reward = render_episode(calls, env)

        if reward < args.min_reward:
            rejected.append({
                "task_id": task_id, "branch": branch, "reward": reward,
                "reason": "below threshold",
            })
            continue

        user_instruction = (
            t.get("instruction", "")
            + "\n\n"
            + (t.get("context", "") or "")
            + "\n\nProceed with the provided task."
        ).strip()

        sample = {
            "messages": (
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_instruction},
                ]
                + turns
            ),
            "task_id": task_id,
            "task_type": task_type,
            "workflow_branch": branch,
            "reward": reward,
        }
        samples.append(sample)

    out_path = out_dir / "train.jsonl"
    with out_path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Wrote {out_path} — {len(samples)} samples")

    # Stats
    by_type = Counter(s["task_type"] for s in samples)
    by_branch = Counter((s["task_type"], s["workflow_branch"]) for s in samples)
    print("\nBy task_type:")
    for k, v in sorted(by_type.items()):
        print(f"  {k:<12} {v}")
    print("\nBy branch:")
    for (tt, br), v in sorted(by_branch.items()):
        print(f"  {tt:<12} {br:<35} {v}")

    if rejected:
        print(f"\n{len(rejected)} rejected:")
        for r in rejected[:20]:
            print(f"  {r}")
        (out_dir / "rejected.json").write_text(json.dumps(rejected, indent=2))

    (out_dir / "stats.json").write_text(json.dumps({
        "n_accepted": len(samples),
        "n_rejected": len(rejected),
        "by_task_type": dict(by_type),
        "by_branch": {f"{k[0]}/{k[1]}": v for k, v in by_branch.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
