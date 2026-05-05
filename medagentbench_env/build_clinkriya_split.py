"""Stratified 80/20 train/test split of clinKriya-Fair (554 tasks).

Stratification: (corpus × task_type × action_label). The intent is to
preserve the six-column reporting (Axis 1 task-type, Axis 2 action label,
corpus) on both sides so train-set shaping and test-set evaluation
coincide on the same distribution strata.

Outputs (medagentbench_env/data/):
  clinkriya_train.json           training tasks (prefixed IDs)
  clinkriya_train.index.json     corresponding labels
  clinkriya_test.json            held-out tasks (prefixed IDs)
  clinkriya_test.index.json      corresponding labels
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "medagentbench_env/data"

SEED = 42
TEST_FRAC = 0.20

tasks = json.loads((DATA / "benchmark_fair.json").read_text())
index = json.loads((DATA / "benchmark_fair.index.json").read_text())
assert len(tasks) == len(index)

# Stratify
rng = random.Random(SEED)
strata: dict = defaultdict(list)
for t, idx in zip(tasks, index):
    key = (idx["corpus"], idx["task_type"], idx["action_label"])
    strata[key].append((t, idx))

train_tasks, test_tasks = [], []
train_index, test_index = [], []
for key in sorted(strata):
    items = list(strata[key])
    rng.shuffle(items)
    n_test = max(1, round(len(items) * TEST_FRAC)) if len(items) >= 5 else max(1, len(items) // 5)
    # Guard: don't leave strata with zero train
    n_test = min(n_test, len(items) - 1) if len(items) > 1 else 0
    test_items, train_items = items[:n_test], items[n_test:]
    for t, idx in train_items:
        # Prefix id with corpus (same as benchmark_fair_combined.json)
        corpus_prefix = "v1" if idx["corpus"] == "v1" else "v2new"
        t2 = dict(t); t2["id"] = f"{corpus_prefix}_{t['id']}"
        train_tasks.append(t2)
        train_index.append({**idx, "prefixed_id": t2["id"]})
    for t, idx in test_items:
        corpus_prefix = "v1" if idx["corpus"] == "v1" else "v2new"
        t2 = dict(t); t2["id"] = f"{corpus_prefix}_{t['id']}"
        test_tasks.append(t2)
        test_index.append({**idx, "prefixed_id": t2["id"]})

# Shuffle within train/test to avoid position-based artifacts
train_combined = list(zip(train_tasks, train_index))
test_combined = list(zip(test_tasks, test_index))
rng.shuffle(train_combined)
rng.shuffle(test_combined)
train_tasks, train_index = [list(t) for t in zip(*train_combined)] if train_combined else ([], [])
test_tasks, test_index = [list(t) for t in zip(*test_combined)] if test_combined else ([], [])

(DATA / "clinkriya_train.json").write_text(json.dumps(train_tasks, indent=2))
(DATA / "clinkriya_train.index.json").write_text(json.dumps(train_index, indent=2))
(DATA / "clinkriya_test.json").write_text(json.dumps(test_tasks, indent=2))
(DATA / "clinkriya_test.index.json").write_text(json.dumps(test_index, indent=2))

# Sanity
print(f"TRAIN: {len(train_tasks)} tasks")
print(f"TEST:  {len(test_tasks)} tasks")
print()
print(f"TRAIN by (corpus, action_label):")
for k, v in sorted(Counter((i["corpus"], i["action_label"]) for i in train_index).items()):
    print(f"  {k[0]:<10} {k[1]:<10} n={v}")
print(f"\nTEST by (corpus, action_label):")
for k, v in sorted(Counter((i["corpus"], i["action_label"]) for i in test_index).items()):
    print(f"  {k[0]:<10} {k[1]:<10} n={v}")
print()
print("Per-type train/test counts:")
all_strata = set(strata.keys())
for k in sorted(all_strata):
    n_tr = sum(1 for i in train_index if (i["corpus"], i["task_type"], i["action_label"]) == k)
    n_te = sum(1 for i in test_index if (i["corpus"], i["task_type"], i["action_label"]) == k)
    print(f"  {k[0]:<8} {k[1]:<8} {k[2]:<10} train={n_tr} test={n_te}")
