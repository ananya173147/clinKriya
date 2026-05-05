"""Construct the clinKriya-Fair benchmark from MedAgentBench v1 + v2-new.

Design (Option B, K=15):
  - Take ALL action-required instances (silent-finish fails) from every task type.
  - Cap no-action instances at K=15 per (corpus, task_type).
  - Every task type from v1 + v2-new (20 total) stays represented.

The cap is on Axis 2 (per-instance silent-finish label), not Axis 1 (task type).
Axis 1 is fully preserved at max (up to 30) per task type.

Outputs:
  data/benchmark_fair.json           — 528 tasks (the clinKriya-Fair benchmark, original v1/v2-new contexts)
  data/benchmark_fair.index.json     — companion {corpus, task_type, task_id, action_label}
  data/benchmark_fair_augmented.json — 528 tasks with two task-type contexts patched (v1 task7, v2-new task3)
                                       to document grader conventions that were omitted in the public dataset;
                                       see CONTEXT_PATCHES below for exact text.
  data/benchmark_full.json           — 600 tasks (v1+v2-new full, reference for inflation comparison)
  data/benchmark_full.index.json     — companion

Silent-finish labels are grader-authoritative, from silent_finish_ceiling.json.
"""

import copy
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "medagentbench_env" / "data"

V1_TASKS = Path(
    "/home/azureuser/RL/clinKriya/medagentbenchv2/medagentbench_v2/src/"
    "MedAgentBench/data/medagentbench/test_data_v2.json"
)
V2_TASKS = DATA / "new_patient_tasks.json"
SILENT = DATA / "silent_finish_ceiling.json"

K_NO_ACTION = 15  # per (corpus, task_type) cap on no-action instances


# Context patches for the clinKriya-Fair-augmented variant.
# Every patch appends a line that documents a format convention the canonical
# strict grader enforces but the public task `context` field never mentions.
# None of them change the task's clinical content or relax any assertion;
# they only tell the model what JSON shape the grader will accept.
#
# Catalogue of strict-grader format conventions (derived by reading
# `MedAgentBench/src/server/tasks/medagentbench/refsol.py` line-by-line and
# tracing actual frontier-model failures — see `docs/cleanup_log.md`):
#
#   - v1 task7 (CBG query): grader computes ref_sol = [last_value] or [-1]
#     when no GLU entries exist. Public context gives the GLU code but omits
#     the "-1 for no-data" clause (task4 / task6 include it verbatim).
#   - v1 task5 (Mg replacement): grader requires dosageInstruction[0].route
#     as a bare string "IV" (grader does `== "IV"`). FHIR-R4-compliant agents
#     emit `{"text": "IV"}`, which fails the equality. Grader also checks
#     FINISH([Mg_value]) or FINISH([]).
#   - v1 task9 (K replacement): grader requires dosageInstruction[0].route as
#     a bare string "oral" (grader does `.lower().strip() == "oral"` — calling
#     .lower() on a dict raises AttributeError). Plus FINISH([K_value]) or [].
#   - v1 task10 (A1C check+reorder): FINISH shape is [value, time_string] when
#     data exists, [-1] when missing. Public context documents [-1] but not
#     the pair.
#   - v2-new task3 (HR averages): grader expects a length-2 list and
#     normalizes -1 / null → None. Public context documents neither the
#     required shape nor the null convention.
CONTEXT_PATCHES: Dict[Tuple[str, str], str] = {
    ("v1", "task5"): (
        " Select the magnesium observation with the most recent effectiveDateTime. "
        "In the MedicationRequest, set the route field to the literal string "
        "\"IV\" — e.g. `\"route\": \"IV\"`; do NOT use a CodeableConcept like "
        "`\"route\": {\"text\": \"IV\"}`. The finish answer should be [value] (a "
        "list with a single number) if the most-recent magnesium measurement is "
        "available, and [-1] if no measurement exists in the last 24 hours. "
        "Returning [] is also accepted."
    ),
    ("v1", "task7"): (
        " The answer should be a single number converted to a unit of mg/dL, "
        "and it should be -1 if a measurement is not available."
    ),
    ("v1", "task9"): (
        " Select the potassium observation with the most recent effectiveDateTime "
        "(not the first entry in the bundle). In the MedicationRequest, set the "
        "route field to the literal string \"oral\" — e.g. `\"route\": \"oral\"`; "
        "do NOT use a CodeableConcept like `\"route\": {\"text\": \"oral\"}`. The "
        "finish answer should be [value] (a list with the most-recent potassium "
        "value as a single number), or [-1] if no measurement exists. Returning "
        "[] is also accepted."
    ),
    # task10 intentionally NOT patched — existing context already documents the
    # "-1 for no measurement" convention, and the [value, effectiveDateTime] pair
    # shape is easy for models to wrap incorrectly (GPT-4o emitted
    # FINISH([[value, time]]) under an earlier patch wording). Leave as-is; the
    # current SR (46.7% GPT-4o / 23.3% Claude 3.7 under fair-orig) is the honest
    # number.
    ("v2_new", "task3"): (
        " The answer should be a list [avg_6h, avg_12h] of two numbers in bpm; "
        "use -1 (or null) for a window that has no heart-rate observations."
    ),
}


def load_tasks():
    v1 = json.load(V1_TASKS.open())
    v2n = json.load(V2_TASKS.open())
    silent = json.load(SILENT.open())["per_task"]
    silent_by_id = {(r["corpus"], r["id"]): r["silent_pass"] for r in silent}
    rows = []
    for t in v1:
        tid = t["id"]
        label = "no_action" if silent_by_id[("v1", tid)] else "action"
        rows.append({"corpus": "v1", "task_type": tid.rsplit("_", 1)[0],
                     "task_id": tid, "action_label": label, "task": t})
    for t in v2n:
        tid = t["id"]
        label = "no_action" if silent_by_id[("v2_new", tid)] else "action"
        rows.append({"corpus": "v2_new", "task_type": tid.rsplit("_", 1)[0],
                     "task_id": tid, "action_label": label, "task": t})
    return rows


def build_fair(rows, k=K_NO_ACTION):
    """Take all action + up to k no-action per (corpus, task_type). Deterministic slice."""
    buckets = defaultdict(lambda: {"action": [], "no_action": []})
    for r in rows:
        buckets[(r["corpus"], r["task_type"])][r["action_label"]].append(r)
    out = []
    for (corpus, tt), b in sorted(buckets.items()):
        out.extend(b["action"])              # all action instances
        out.extend(b["no_action"][:k])       # cap no-action at k
    return out


def summarize(name, tier):
    labels = Counter(r["action_label"] for r in tier)
    types = Counter((r["corpus"], r["task_type"]) for r in tier)
    ceiling = labels["no_action"] / max(1, len(tier))
    print(f"\n=== {name} ===")
    print(f"  total:         {len(tier)}")
    print(f"  action:        {labels['action']}")
    print(f"  no_action:     {labels['no_action']}")
    print(f"  ceiling:       {100*ceiling:.1f}%")
    print(f"  task-type cells: {len(types)}")
    print(f"  per-type sizes (sorted):")
    # show every (corpus, task_type, #action, #no_action, subtotal)
    per = defaultdict(lambda: {"action": 0, "no_action": 0})
    for r in tier:
        per[(r["corpus"], r["task_type"])][r["action_label"]] += 1
    for (corpus, tt), c in sorted(per.items()):
        total = c["action"] + c["no_action"]
        print(f"    {corpus:7s} {tt:10s}  action={c['action']:2d}  no_action={c['no_action']:2d}  total={total:3d}")


def apply_context_patches(rows):
    """Return a deep copy of rows with CONTEXT_PATCHES appended to matching task types."""
    patched = []
    patched_count = defaultdict(int)
    for r in rows:
        key = (r["corpus"], r["task_type"])
        if key in CONTEXT_PATCHES:
            new_r = copy.deepcopy(r)
            old_ctx = new_r["task"].get("context", "")
            new_r["task"]["context"] = (old_ctx.rstrip() + CONTEXT_PATCHES[key]).lstrip()
            patched.append(new_r)
            patched_count[key] += 1
        else:
            patched.append(r)
    print(f"\n  context patches applied:")
    for key, n in sorted(patched_count.items()):
        print(f"    {key[0]:7s} {key[1]:10s}  {n} instances")
    return patched


def main():
    rows = load_tasks()
    assert len(rows) == 600, f"expected 600 tasks, got {len(rows)}"

    full = list(rows)
    fair = build_fair(rows, k=K_NO_ACTION)
    fair_augmented = apply_context_patches(fair)

    summarize("Full v1 + v2-new (reference)", full)
    summarize(f"clinKriya-Fair (K={K_NO_ACTION})", fair)

    # Delete stale three-tier outputs from the prior design
    for stale in ("benchmark_action_only.json", "benchmark_action_only.index.json",
                  "benchmark_discrimination.json", "benchmark_discrimination.index.json"):
        p = DATA / stale
        if p.exists():
            p.unlink()
            print(f"  removed stale {p.name}")

    # Name → (rows, embed-body flag). We embed the full task body in the index
    # for the augmented variant because its contexts are patched vs. the
    # v1/v2-new source of truth; downstream consumers (t2_baseline_v2.py's
    # --tasks-file bypass) must use the patched body, not the raw dataset body.
    outputs = [
        ("benchmark_fair", fair, False),
        ("benchmark_fair_augmented", fair_augmented, True),
        ("benchmark_full", full, False),
    ]
    for name, tier, embed_body in outputs:
        path = DATA / f"{name}.json"
        idx_path = DATA / f"{name}.index.json"
        json.dump([r["task"] for r in tier], path.open("w"), indent=2)
        if embed_body:
            idx = [dict(r) for r in tier]  # keep "task" field
        else:
            idx = [{k: v for k, v in r.items() if k != "task"} for r in tier]
        json.dump(idx, idx_path.open("w"), indent=2)
        print(f"  wrote {path.name} ({len(tier)} tasks) + {idx_path.name}"
              + ("  [body embedded in index]" if embed_body else ""))


if __name__ == "__main__":
    main()
