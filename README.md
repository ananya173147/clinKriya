# clinKriya

Reinforcement learning framework for training LLM-based clinical agents on structured EHR tasks. Uses GRPO (Group Relative Policy Optimization) via TRL to train models that interact with a mock FHIR server through tool calls.

Built on top of [MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench) v1 and v2.

---

## Quick Start

```bash
pip install -e "medagentbench_env[train]"

# Run GRPO training (Qwen3-8B, SFT warmstart)
CUBLAS_DIR="$(pwd)/venv/lib/python3.12/site-packages/nvidia/cublas/lib"
nohup env LD_PRELOAD="$CUBLAS_DIR/libcublas.so.12:$CUBLAS_DIR/libcublasLt.so.12" \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  venv/bin/python -m medagentbench_env.train \
  --model Qwen/Qwen3-8B \
  --sft-adapter <path-to-sft-adapter> \
  --lora-rank 64 \
  --per-device-batch-size 4 \
  --num-generations 4 \
  --tasks-file medagentbench_env/data/clinkriya_train.json \
  --output-dir training/output_v31

# Eval a trained adapter on the held-out test set
venv/bin/python -m medagentbench_env.eval_checkpoints \
  --model Qwen/Qwen3-8B \
  --adapter training/output_v31 \
  --tasks-file medagentbench_env/data/clinkriya_test.json \
  --num-rollouts-per-task 4 \
  --output-dir training/eval/v31

# SFT warmstart
venv/bin/python -m medagentbench_env.train_sft \
  --model Qwen/Qwen3-8B \
  --lora-rank 64 \
  --output-dir training/sft_r64
```

---

## Dataset — clinKriya-Fair

554 tasks split into train/test with no overlap:

| Split | File | Tasks |
|-------|------|-------|
| Train | `data/clinkriya_train.json` | 439 |
| Test  | `data/clinkriya_test.json`  | 115 |

Tasks come from MedAgentBench v1 (`v1_` prefix) and v2-new (`v2new_` prefix), covering 20 task types across both corpora. The train set excludes `v1_task3` during RL training (always-action HR average — no decision signal).

---

## Task Types (Scenarios)

Each task gives the agent a patient MRN and a clinical instruction. The agent must look up relevant chart data via FHIR tool calls, decide whether action is needed, and either place orders or report findings.

### v1 Corpus (MedAgentBench v1 patients)

| Task | Clinical Scenario | Decision |
|------|-------------------|----------|
| **v1_task1** | Patient lookup by name + DOB | Return MRN or "not found" |
| **v1_task2** | Patient demographics | Return patient age |
| **v1_task3** | Vital sign filing | Record a blood pressure reading (always-action, excluded from RL) |
| **v1_task4** | Magnesium level check | Return most recent Mg lab value within 24 hours |
| **v1_task5** | Magnesium replacement | If Mg is low, order IV magnesium replacement per protocol |
| **v1_task6** | CBG (capillary blood glucose) average | Calculate mean CBG over last 24 hours |
| **v1_task7** | CBG spot check | Return most recent CBG value |
| **v1_task8** | Orthopedic surgery referral | Order STAT referral with structured SBAR note |
| **v1_task9** | Potassium replacement | If K⁺ is low, order potassium replacement + follow-up serum K lab |
| **v1_task10** | HbA1c reorder | If last A1c is >1 year old, place a STAT HbA1c order |

### v2-new Corpus (MedAgentBench v2 patients)

| Task | Clinical Scenario | Decision |
|------|-------------------|----------|
| **v2new_task1** | CT Abdomen recency | If last CT is >12 months old, order new CT Abdomen with IV contrast |
| **v2new_task2** | DVT prophylaxis deduplication | Ensure exactly one active heparin order; discontinue duplicates |
| **v2new_task3** | Heart rate averages | Calculate mean HR over 6h and 12h windows |
| **v2new_task4** | Urinary catheter removal | If catheter has been in place >48h with no removal order, create one |
| **v2new_task5** | Oncology CT + IR referral | If malignant kidney neoplasm and CT >3 months old, order CT + IR referral |
| **v2new_task6** | Thyroid protocol | Based on TSH/FT4 levels, order levothyroxine and/or thyroid labs |
| **v2new_task7** | QTc safety protocol | If QTc is prolonged, discontinue QT-prolonging meds and order follow-up ECG |
| **v2new_task8** | Opioid safety — naloxone pairing | Ensure every active opioid order has a matching naloxone prescription |
| **v2new_task9** | Influenza vaccine recency | If last flu vaccine was >365 days ago, order a new one |
| **v2new_task10** | COVID booster recency | If last COVID vaccine was >12 months ago, order a booster |

---

## Tools

The agent has access to 9 FHIR tools exposed as function calls:

| Tool | Type | Description |
|------|------|-------------|
| `GET /Condition` | Read | Retrieve patient problem list (diagnoses, conditions) |
| `GET /Observation` | Read | Retrieve lab results and vital signs |
| `POST /Observation` | Write | File a vital sign or flowsheet value |
| `GET /MedicationRequest` | Read | Retrieve active and historical medication orders |
| `POST /MedicationRequest` | Write | Create a new medication order |
| `GET /Procedure` | Read | Retrieve completed procedures and surgical history |
| `POST /ServiceRequest` | Write | Create a lab order, imaging request, or referral |
| `GET /Patient` | Read | Search for a patient by name, DOB, or MRN |
| `finish(value)` | Control | Declare task complete and return the answer |

All read tools support filtering by `patient`, `code`, `date`, `status`, and `_count`. The environment runs fully offline against a mock FHIR cache (`data/fhir_cache.json.gz`) with no live server required.

---

## Reward / Verifier

Scoring is handled by `verifier.py`, the single source of truth for both training and evaluation. Rewards are in `[-1.0, 2.0]`.

### Terminal reward (dominant signal)
The primary grader (`new_refsol` / `refsol`) checks the agent's final answer and all POSTed FHIR resources against the reference solution. A pass awards **+1.0**. This is the main learning signal.

### Dense shape rewards (intermediate guidance)

| Signal | Weight | Condition |
|--------|--------|-----------|
| `terminal` | **+1.00** | Reference solution grader passes |
| `get_credit` | +0.05 | Agent queried chart AND placed an order AND called finish |
| `action_a` | +0.10 | Correct primary POST: right resource type + task-specific code match |
| `action_b` | +0.10 | Correct secondary POST (tasks requiring two orders: task5/6/7/v1_task9) |
| `partial_action` | +0.05 | Right resource type + patient reference, but wrong/missing code |

### Penalties

| Signal | Weight | Condition |
|--------|--------|-----------|
| `spurious_post` | -0.05 | Posted a resource unrelated to the task |
| `skip_finish_penalty` | -0.20 | Called `finish()` with no prior GET or POST |
| `invalid_fhir` | -0.10 (per call) | Malformed or rejected FHIR request |
| `redundant_lookup` | -0.05 (per call) | Repeated identical GET URL, capped at -0.10 |
| `offtarget_lookup` | -0.05 (per call) | GET on wrong resource type for the task, capped at -0.10 |

### Per-task grader rules (action_a / action_b full credit)

| Task | Primary action check | Secondary action check |
|------|---------------------|----------------------|
| v1_task1 | `ServiceRequest` with CPT `74177` (CT Abdomen) | — |
| v1_task2 | `MedicationRequest` with anticoagulant (heparin/enoxaparin) | — |
| v1_task4 | `ServiceRequest` with catheter removal code `NUR1373` | — |
| v1_task5 (CT+IR) | `ServiceRequest` with CPT `74177` | `ServiceRequest` with IR referral code `CON417` |
| v1_task6 (thyroid) | `MedicationRequest` levothyroxine 25mcg, active order, authoredOn | `ServiceRequest` TSH or FT4 lab, active order, authoredOn |
| v1_task7 (QTc) | `MedicationRequest` QT-prolonging med discontinued | `ServiceRequest` ECG (`445118002`) |
| v1_task8 (ortho) | `ServiceRequest` orthopedic referral (`306181000000106`), STAT, authoredOn | — |
| v1_task9 (flu) | `ServiceRequest` or `MedicationRequest` flu vaccine (CPT `90686`), active, authoredOn | — |
| v1_task10 (HbA1c) | `ServiceRequest` HbA1c (`4548-4`), STAT, active, authoredOn | — |
| v1_task5 (Mg) | `MedicationRequest` magnesium (NDC `0338-1715-40`), active, authoredOn, dosage | — |
| v1_task9 (K) | `MedicationRequest` potassium (NDC `40032-917-01`), active, authoredOn, dosage | `ServiceRequest` serum K lab (`2823-3`), active, authoredOn |
| v2new_task10 (COVID) | `ServiceRequest` or `MedicationRequest` COVID vaccine, active, authoredOn | — |

The strict reference solution grader (from MedAgentBench) is the primary pass/fail check and imposes additional format constraints (exact coding structures, note text, etc.) beyond what the intermediate reward captures.

---

## Architecture

```
train.py  →  fhir_env.py (MedAgentTrainEnv)  →  verifier.py (reward)
```

- **`train.py`** — GRPO training entrypoint. Configures GRPOTrainer with LoRA, runs rollout batches.
- **`fhir_env.py`** — Training-time FHIR environment. Exposes tools as methods, manages episode state, uses MockFHIR for offline rollouts.
- **`verifier.py`** — Decoupled reward logic. `evaluate(history, task_spec, ...)` is the single scoring function used by both training and eval.
- **`train_sft.py`** — SFT warmstart trainer (LoRA, assistant-only loss, Qwen3 chat template).
- **`eval_checkpoints.py`** — Held-out evaluation harness for LoRA adapters.
- **`t2_baseline_v2.py`** — Frontier model baseline via OpenRouter (GPT-4o, Claude, etc.).
- **`server/`** — FastAPI server + WebSocket env for interactive demo and baseline eval.

## Repository Layout

```
medagentbench_env/
  train.py                    # GRPO training
  train_sft.py                # SFT warmstart
  fhir_env.py                 # Training environment
  verifier.py                 # Reward / grader
  eval_checkpoints.py         # Eval harness
  t2_baseline_v2.py           # Frontier baseline
  export.py                   # Reward curve plots
  models.py                   # Pydantic types
  client.py                   # WebSocket client
  build_clinkriya_split.py    # Builds train/test split
  build_benchmark_subsets.py  # Builds clinKriya-Fair benchmark
  build_oracle_demos.py       # Oracle SFT demo generator
  build_sft_dataset.py        # Rejection-sampling SFT extractor
  silent_finish_ceiling.py    # Measures silent-finish ceiling per task
  create_holdout.py           # Legacy 80/20 split (pre-clinKriya-Fair)
  server/
    app.py                    # FastAPI + UI
    fhir_cache.py             # MockFHIR (offline FHIR cache)
    reward.py                 # Server-side reward wrapper
  data/
    clinkriya_train.json      # 439-task train split
    clinkriya_test.json       # 115-task test split
    benchmark_fair.json       # 554-task clinKriya-Fair benchmark
    fhir_cache.json.gz        # Compressed offline FHIR cache (28 MB)
    new_system.txt            # System prompt
    funcs_v1.json             # Tool JSON schemas
    silent_finish_ceiling.json  # Per-task silent-finish labels
medagentbenchv2/              # MedAgentBench v2 reference (read-only)
```
