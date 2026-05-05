#!/usr/bin/env python3
"""
Create a stratified train/test holdout split of MedAgentBench tasks.

Within each (task_type, workflow_branch) group, ~20% of tasks are held out
for test. Groups with <5 examples stay entirely in train to avoid leaking
unique clinical scenarios.

Outputs:
  data/train_tasks.json  — training set
  data/test_tasks.json   — held-out test set (never seen during training)

Usage:
  python -m medagentbench_env.create_holdout [--test-ratio 0.2] [--seed 42]
"""

import argparse
import json
import math
import re
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"
_V2_DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "medagentbenchv2" / "medagentbench_v2" / "src"
    / "MedAgentBench" / "data" / "medagentbench" / "test_data_v2.json"
)
_BASE = "http://localhost:8080/fhir"
_NOW = datetime(2024, 1, 1)
_FLU_REF = datetime(2024, 1, 9)   # reference date used in task9 context
_V2_REF = datetime(2023, 11, 13, 10, 15)  # reference date used in v2 task contexts


def _get_entries(cache, mrn, resource):
    key = f"{_BASE}/{resource}?_count=5000&_format=json&patient={mrn}"
    data = cache.get(key, {}).get("data", {})
    return data.get("entry", []) if isinstance(data, dict) else []


def _ct_age_months(cache, mrn):
    CT_CODES = {"IMGCT0491", "IMGIL0001", "74177"}
    dates = []
    for e in _get_entries(cache, mrn, "Procedure"):
        res = e.get("resource", {})
        codes = {c.get("code", "") for c in res.get("code", {}).get("coding", [])}
        if codes & CT_CODES:
            d = res.get("performedDateTime", res.get("performedPeriod", {}).get("start", ""))
            if d:
                dates.append(d[:10])
    if not dates:
        return None
    try:
        return (_NOW - datetime.strptime(max(dates), "%Y-%m-%d")).days / 30
    except ValueError:
        return None


def _active_med_count(cache, mrn, keywords):
    count = 0
    for e in _get_entries(cache, mrn, "MedicationRequest"):
        res = e.get("resource", {})
        if res.get("status") == "active":
            text = str(res.get("medicationCodeableConcept", {})).lower()
            if any(kw in text for kw in keywords):
                count += 1
    return count


def _latest_obs_value(cache, mrn, code_substr):
    obs = []
    for e in _get_entries(cache, mrn, "Observation"):
        res = e.get("resource", {})
        coding = res.get("code", {}).get("coding", [])
        if any(code_substr.lower() in (c.get("code", "") + c.get("display", "")).lower()
               for c in coding):
            v = res.get("valueQuantity", {}).get("value")
            dt = res.get("effectiveDateTime", "")
            if v is not None:
                obs.append((dt, float(v)))
    # Sort by effectiveDateTime (ISO strings sort correctly) to get most recent value
    return sorted(obs)[-1][1] if obs else None


def _has_condition_code(cache, mrn, code_prefix):
    for e in _get_entries(cache, mrn, "Condition"):
        res = e.get("resource", {})
        for c in res.get("code", {}).get("coding", []):
            if c.get("code", "").startswith(code_prefix):
                return True
    return False


def _covid_vax_age_months(cache, mrn):
    """Match v2-new task10 grader: Procedure code COVIDVACCINE OR
    MedicationRequest with status="completed" AND text containing "covid-19 vac".
    Earlier this matched any text-mention of covid keywords, but the grader
    only looks at these strict patterns, so over-broad matches caused the
    classifier to label patients as `recent_vax_no_action` when the grader
    saw no qualifying vaccine → expected POST → no-action oracle rejected."""
    dates = []
    for e in _get_entries(cache, mrn, "Procedure"):
        res = e.get("resource", {})
        codes = {c.get("code", "") for c in res.get("code", {}).get("coding", [])}
        if "COVIDVACCINE" in codes:
            for field in ("performedDateTime", "performedPeriod"):
                d = res.get(field, "")
                if isinstance(d, dict):
                    d = d.get("end") or d.get("start") or ""
                if d: dates.append(d[:10])
    for e in _get_entries(cache, mrn, "MedicationRequest"):
        res = e.get("resource", {})
        if res.get("status", "").lower() != "completed":
            continue
        txt = (res.get("medicationCodeableConcept", {}) or {}).get("text", "").lower()
        if "covid-19 vac" in txt:
            d = res.get("authoredOn", "")
            if d: dates.append(d[:10])
    if not dates:
        return None
    try:
        return (_NOW - datetime.strptime(max(dates), "%Y-%m-%d")).days / 30
    except ValueError:
        return None


def _catheter_max_hours(cache, mrn, ref_time):
    hours = []
    for e in _get_entries(cache, mrn, "Procedure"):
        res = e.get("resource", {})
        codes = {c.get("code", "") for c in res.get("code", {}).get("coding", [])}
        if "NUR1373" in codes:
            d = res.get("performedDateTime", res.get("performedPeriod", {}).get("start", ""))
            if d:
                try:
                    dt = datetime.fromisoformat(d.replace("Z", "+00:00").replace("+00:00", ""))
                    hours.append((ref_time - dt).total_seconds() / 3600)
                except ValueError:
                    pass
    return max(hours) if hours else None


def _tsh_values(cache, mrn):
    vals = []
    for e in _get_entries(cache, mrn, "Observation"):
        res = e.get("resource", {})
        if any(c.get("code", "") == "TSH" for c in res.get("code", {}).get("coding", [])):
            v = res.get("valueQuantity", {}).get("value")
            if v is not None:
                vals.append(float(v))
    return vals


def _flu_vax_age_days(cache, mrn):
    """Match the v2-new task9 grader, which queries strictly by CPT 90686.
    Earlier this also matched on text "influenza"/"flu vaccine" — but those
    text-matched procedures are invisible to the grader, so a classifier
    that found them would mis-label patients as `recent_flu_vax_no_action`
    when the grader still considered them un-vaccinated → grader expected
    a POST and rejected the no-action oracle at reward 0.0."""
    dates = []
    for resource in ("Procedure", "Observation"):
        for e in _get_entries(cache, mrn, resource):
            res = e.get("resource", {})
            codes = {c.get("code", "") for c in res.get("code", {}).get("coding", [])}
            if "90686" in codes:
                for field in ("performedDateTime", "effectiveDateTime", "authoredOn"):
                    d = res.get(field, "")
                    if d:
                        dates.append(d[:10])
    if not dates:
        return None
    try:
        return (_FLU_REF - datetime.strptime(max(dates), "%Y-%m-%d")).days
    except ValueError:
        return None


def classify_branch(task, cache):
    ttype = "_".join(task["id"].split("_")[:-1])
    mrn = task["eval_MRN"]
    ctx = task.get("context", "")

    if ttype == "task1":
        ct = _ct_age_months(cache, mrn)
        if ct is None:
            return "no_ct_order"
        return "recent_ct_no_action" if ct < 12 else "old_ct_order"

    if ttype == "task2":
        DVT_KWS = ["heparin", "enoxaparin", "rivaroxaban", "apixaban", "fondaparinux", "dvt"]
        n = _active_med_count(cache, mrn, DVT_KWS)
        return "0_orders_create" if n == 0 else ("1_order_ok" if n == 1 else "multiple_discontinue")

    if ttype == "task3":
        # Always a calculation; branch by whether HR data exists in the time window
        obs = _get_entries(cache, mrn, "Observation")
        has_hr = any(
            any(c.get("code", "") == "HEARTRATE" for c in e.get("resource", {}).get("code", {}).get("coding", []))
            for e in obs
        )
        return "has_hr_data" if has_hr else "no_hr_data"

    if ttype == "task4":
        m = re.search(r"It's ([\d\-T:+]+) now", ctx)
        ref = datetime.fromisoformat(m.group(1).replace("+00:00", "")) if m else datetime(2023, 11, 16, 10, 0)
        max_h = _catheter_max_hours(cache, mrn, ref)
        if max_h is None:
            return "no_catheter_no_action"
        return "catheter_gt48h_order" if max_h > 48 else "catheter_lt48h_no_action"

    if ttype == "task5":
        has_kidney_ca = _has_condition_code(cache, mrn, "C64")
        ct = _ct_age_months(cache, mrn)
        if not has_kidney_ca:
            return "no_kidney_ca_no_action"
        if ct is None or ct >= 12:
            return "kidney_ca_order_ct"
        return "kidney_ca_recent_ct_no_action"

    if ttype == "task6":
        tsh_vals = _tsh_values(cache, mrn)
        if not tsh_vals:
            return "no_tsh_data"
        high_count = sum(1 for v in tsh_vals if v > 10)
        latest = tsh_vals[-1]
        if latest < 0.5:
            return "tsh_low_hyperthyroid"
        if high_count >= 2:
            return "tsh_high_x2_order_levothyroxine"
        if latest > 10:
            return "tsh_high_x1_monitor"
        return "tsh_normal_no_action"

    if ttype == "task7":
        qtc = _latest_obs_value(cache, mrn, "qtc") or _latest_obs_value(cache, mrn, "QTc")
        if qtc is None:
            return "no_qtc_data"
        return "prolonged_discontinue" if qtc > 450 else "normal_no_action"

    if ttype == "task8":
        OPIOIDS = ["hydromorphone", "oxycodone", "fentanyl", "hydrocodone", "morphine"]
        active = [
            e.get("resource", {}) for e in _get_entries(cache, mrn, "MedicationRequest")
            if e.get("resource", {}).get("status") == "active"
        ]
        has_opioid = any(
            any(op in str(r.get("medicationCodeableConcept", {})).lower() for op in OPIOIDS)
            for r in active
        )
        has_naloxone = any(
            "naloxone" in str(r.get("medicationCodeableConcept", {})).lower()
            for r in active
        )
        if not has_opioid:
            return "no_opioid_no_action"
        return "opioid_with_naloxone_ok" if has_naloxone else "opioid_missing_naloxone_order"

    if ttype == "task9":
        age_days = _flu_vax_age_days(cache, mrn)
        if age_days is None:
            return "no_flu_vax_order"
        return "recent_flu_vax_no_action" if age_days <= 365 else "old_flu_vax_order"

    if ttype == "task10":
        age = _covid_vax_age_months(cache, mrn)
        if age is None:
            return "no_vax_order"
        return "recent_vax_no_action" if age < 6 else "old_vax_order_booster"

    # ── v2 task types ────────────────────────────────────────────────────────

    if ttype == "v2_task5":
        # Mg in last 24h; none/low → order replacement; normal → no action
        cutoff = (_V2_REF - timedelta(hours=24)).isoformat()[:19]
        mg_in_window = []
        for e in _get_entries(cache, mrn, "Observation"):
            res = e.get("resource", {})
            if any(c.get("code", "") == "MG" for c in res.get("code", {}).get("coding", [])):
                dt = res.get("effectiveDateTime", "")
                if dt and dt[:19] >= cutoff:
                    v = res.get("valueQuantity", {}).get("value")
                    if v is not None:
                        mg_in_window.append(float(v))
        if not mg_in_window:
            return "no_mg_24h_order_stat"
        return "mg_low_order_replacement" if min(mg_in_window) < 1.8 else "mg_normal_no_action"

    if ttype == "v2_task9":
        # Most recent K (by effectiveDateTime); low (<3.5) → order replacement + morning K check
        k_obs = []
        for e in _get_entries(cache, mrn, "Observation"):
            res = e.get("resource", {})
            if any(c.get("code", "") == "K" for c in res.get("code", {}).get("coding", [])):
                v = res.get("valueQuantity", {}).get("value")
                dt = res.get("effectiveDateTime", "")
                if v is not None:
                    k_obs.append((dt, float(v)))
        if not k_obs:
            return "no_k_data"
        latest_k = sorted(k_obs)[-1][1]  # ISO datetime strings sort correctly
        return "k_low_order_replacement" if latest_k < 3.5 else "k_normal_no_action"

    if ttype == "v2_task10":
        # Most recent A1C; if >1 year old or absent → order new
        a1c_entries = []
        for e in _get_entries(cache, mrn, "Observation"):
            res = e.get("resource", {})
            if any(c.get("code", "") == "A1C" for c in res.get("code", {}).get("coding", [])):
                dt = res.get("effectiveDateTime", "")
                v = res.get("valueQuantity", {}).get("value")
                if dt and v is not None:
                    a1c_entries.append((dt[:10], float(v)))
        if not a1c_entries:
            return "no_a1c_order"
        latest_dt = max(a1c_entries)[0]
        try:
            age_days = (_V2_REF.date() - datetime.strptime(latest_dt, "%Y-%m-%d").date()).days
        except ValueError:
            return "no_a1c_order"
        return "a1c_recent_no_action" if age_days <= 365 else "a1c_old_order"

    return "unknown"


# ---------------------------------------------------------------------------
# Curation config — derived from branch entropy + action-branch analysis.
#
# EXCLUDED task types (not worth training on):
#   task3   — pure calculation, no conditional decision, not in _RL_TASK_TYPES
#   task5   — 93% trivial (no_kidney_ca→no_action); only 2 action examples
#
# For retained task types, dominant (easy) branches are capped so the model
# sees roughly balanced training signal. Rare action branches are always kept
# in full.
# ---------------------------------------------------------------------------

_EXCLUDED_TASK_TYPES = {"task3", "task5"}

# max examples to keep from each branch per task type.
# None = keep all.  Branches not listed get capped at _DEFAULT_MAJORITY_CAP.
_DEFAULT_MAJORITY_CAP = 5   # hard ceiling for any dominant branch not listed below

_BRANCH_CAPS: dict = {
    # task1: 24 no_ct dominates — cap it; keep all rare conditional branches
    "task1": {
        "no_ct_order":          4,
        "recent_ct_no_action":  None,
        "old_ct_order":         None,
    },
    # task2: well-distributed 17/7/6 — keep all
    "task2": {},
    # task4: well-distributed 17/10/3 — keep all
    "task4": {},
    # task6: only 1 action example (tsh_hi_x2) — keep it; cap no-data branch
    "task6": {
        "tsh_normal_no_action":              None,
        "no_tsh_data":                       4,
        "tsh_high_x2_order_levothyroxine":   None,
        "tsh_low_hyperthyroid":              None,
        "tsh_high_x1_monitor":               None,
    },
    # task7: well-distributed 17/9/4 — keep all
    "task7": {},
    # task8: 21/9 — cap dominant, keep full minority
    "task8": {
        "opioid_missing_naloxone_order": 12,
        "no_opioid_no_action":           None,
        "opioid_with_naloxone_ok":       None,
    },
    # task9: 25 no_flu trivial — cap hard; keep all conditional branches
    "task9": {
        "no_flu_vax_order":        4,
        "old_flu_vax_order":       None,
        "recent_flu_vax_no_action": None,
    },
    # task10: 25 no_covid trivial — cap; keep all conditional branches
    "task10": {
        "no_vax_order":          4,
        "old_vax_order_booster": None,
        "recent_vax_no_action":  None,
    },
    # v2_task5: near-50/50 — keep all
    "v2_task5": {},
    # v2_task9: ~90% k_normal trivial — cap hard; keep all action examples
    "v2_task9": {
        "k_normal_no_action":    4,
        "k_low_order_replacement": None,
        "no_k_data":             3,
    },
    # v2_task10: best-distributed 13/13/4 — keep all
    "v2_task10": {},
}


def _curate(tasks, cache, seed):
    """Apply curation caps per task type / branch. Returns curated task list."""
    rng = random.Random(seed)

    # Group by (task_type, branch)
    groups: dict = defaultdict(list)
    for t in tasks:
        ttype = "_".join(t["id"].split("_")[:-1])
        if ttype in _EXCLUDED_TASK_TYPES:
            continue
        b = classify_branch(t, cache)
        t = dict(t)
        t["workflow_branch"] = b
        groups[(ttype, b)].append(t)

    curated = []
    for (ttype, b), group in sorted(groups.items()):
        rng.shuffle(group)
        caps = _BRANCH_CAPS.get(ttype, {})
        cap = caps.get(b, _DEFAULT_MAJORITY_CAP if b not in caps else None)
        if cap is None:
            curated.extend(group)
        else:
            curated.extend(group[:cap])

    return curated


def build_split(tasks, cache, test_ratio, seed, min_group_size):
    rng = random.Random(seed + 1)  # different seed from curation shuffle

    groups: dict = defaultdict(list)
    for t in tasks:
        ttype = "_".join(t["id"].split("_")[:-1])
        b = t.get("workflow_branch") or classify_branch(t, cache)
        groups[(ttype, b)].append(t)

    train, test = [], []
    for (ttype, b), group in sorted(groups.items()):
        rng.shuffle(group)
        n_test = math.floor(len(group) * test_ratio) if len(group) >= min_group_size else 0
        test.extend(group[:n_test])
        train.extend(group[n_test:])

    return train, test


def main():
    parser = argparse.ArgumentParser(description="Stratified train/test split for MedAgentBench")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-group-size", type=int, default=5,
                        help="Min branch size to put any examples in test (default: 5)")
    parser.add_argument("--data-dir", type=str, default=str(_DATA_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    with open(data_dir / "new_patient_tasks.json") as f:
        tasks = json.load(f)
    with open(data_dir / "fhir_cache.json") as f:
        cache = json.load(f)

    # Add v2_task5, v2_task9, v2_task10 with prefixed IDs (mirrors fhir_env._get_tasks)
    _V2_RL = {"task5", "task9", "task10"}
    if _V2_DATA_PATH.exists():
        with open(_V2_DATA_PATH) as f:
            v2_raw = json.load(f)
        for t in v2_raw:
            ttype = "_".join(t["id"].split("_")[:-1])
            if ttype in _V2_RL:
                prefixed = dict(t)
                prefixed["id"] = f"v2_{t['id']}"
                tasks.append(prefixed)
    else:
        print(f"Warning: v2 data not found at {_V2_DATA_PATH}")

    # Step 1: curate — drop excluded tasks, cap dominant branches
    curated = _curate(tasks, cache, args.seed)

    # Step 2: stratified train/test split on the curated set
    train, test = build_split(curated, cache, args.test_ratio, args.seed, args.min_group_size)

    out_train = data_dir / "train_tasks.json"
    out_test = data_dir / "test_tasks.json"
    out_train.write_text(json.dumps(train, indent=2))
    out_test.write_text(json.dumps(test, indent=2))

    # Summary
    from collections import Counter
    all_by_branch: dict = defaultdict(lambda: defaultdict(lambda: {"train": 0, "test": 0}))
    for t in train:
        tt = "_".join(t["id"].split("_")[:-1])
        all_by_branch[tt][t["workflow_branch"]]["train"] += 1
    for t in test:
        tt = "_".join(t["id"].split("_")[:-1])
        all_by_branch[tt][t["workflow_branch"]]["test"] += 1

    print(f"\nCurated: {len(curated)} tasks (from {len(tasks)} raw) → {len(train)} train / {len(test)} test")
    print(f"Excluded: {sorted(_EXCLUDED_TASK_TYPES)}\n")
    print(f"{'Task type':<12} {'Branch':<42} {'train':>6} {'test':>6}")
    print("-" * 70)
    for tt in sorted(all_by_branch):
        for i, (b, counts) in enumerate(sorted(all_by_branch[tt].items())):
            label = tt if i == 0 else ""
            print(f"  {label:<10} {b:<42} {counts['train']:>6} {counts['test']:>6}")
        tt_train = sum(c["train"] for c in all_by_branch[tt].values())
        tt_test  = sum(c["test"]  for c in all_by_branch[tt].values())
        print(f"  {'':10} {'TOTAL':42} {tt_train:>6} {tt_test:>6}")
        print()
    print(f"Wrote {out_train}")
    print(f"Wrote {out_test}")


if __name__ == "__main__":
    main()
