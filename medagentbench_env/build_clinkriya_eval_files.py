"""Split clinKriya-Fair (benchmark_fair.json) into per-corpus eval files.

Reason: the env+verifier pipeline uses new_refsol (v2-new graders). v1 tasks
from clinKriya-Fair need the canonical v1 refsol.py + a v1-harness adapter
that emits raw-HTTP agent messages — out of scope until Phase 4.

For Phase 1 we separate the two corpuses so we can run eval on the v2-new
subset (255 tasks) using the already-correct grader path. v1 subset (273
tasks) is written too for future reuse once the adapter lands.

Outputs:
  medagentbench_env/data/benchmark_fair_v1.json         (273 tasks)
  medagentbench_env/data/benchmark_fair_v2new.json      (255 tasks)
  medagentbench_env/data/benchmark_fair_v2new.index.json (aligned labels)
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "medagentbench_env/data"

tasks = json.loads((DATA / "benchmark_fair.json").read_text())
index = json.loads((DATA / "benchmark_fair.index.json").read_text())

assert len(tasks) == len(index), f"{len(tasks)} vs {len(index)}"

v1_tasks, v2new_tasks = [], []
v1_index, v2new_index = [], []
for t, idx in zip(tasks, index):
    if idx["corpus"] == "v1":
        v1_tasks.append(t)
        v1_index.append(idx)
    elif idx["corpus"] in ("v2-new", "v2_new"):
        v2new_tasks.append(t)
        v2new_index.append(idx)
    else:
        raise ValueError(f"Unknown corpus: {idx['corpus']}")

(DATA / "benchmark_fair_v1.json").write_text(json.dumps(v1_tasks, indent=2))
(DATA / "benchmark_fair_v1.index.json").write_text(json.dumps(v1_index, indent=2))
(DATA / "benchmark_fair_v2new.json").write_text(json.dumps(v2new_tasks, indent=2))
(DATA / "benchmark_fair_v2new.index.json").write_text(json.dumps(v2new_index, indent=2))

# Combined file with corpus-prefixed IDs for unified eval. Verifier strips
# the prefix and routes grading to the correct refsol module.
def _prefix(t, prefix):
    t2 = dict(t)
    t2["id"] = f"{prefix}_{t['id']}"
    return t2

combined = [_prefix(t, "v1") for t in v1_tasks] + [_prefix(t, "v2new") for t in v2new_tasks]
combined_index = [{**i, "prefixed_id": f"v1_{i['task_id']}"} for i in v1_index] + \
                 [{**i, "prefixed_id": f"v2new_{i['task_id']}"} for i in v2new_index]
(DATA / "benchmark_fair_combined.json").write_text(json.dumps(combined, indent=2))
(DATA / "benchmark_fair_combined.index.json").write_text(json.dumps(combined_index, indent=2))
print(f"combined: {len(combined)} tasks (prefixed IDs)")

print(f"v1     : {len(v1_tasks):>3} tasks")
print(f"v2-new : {len(v2new_tasks):>3} tasks")
print()
print(f"v1 by task_type:     {dict(Counter(i['task_type'] for i in v1_index))}")
print(f"v1 by action_label:  {dict(Counter(i['action_label'] for i in v1_index))}")
print()
print(f"v2new by task_type:    {dict(Counter(i['task_type'] for i in v2new_index))}")
print(f"v2new by action_label: {dict(Counter(i['action_label'] for i in v2new_index))}")
