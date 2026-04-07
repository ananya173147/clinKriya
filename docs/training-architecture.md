# Training Architecture: TRL vs Custom GRPO

This document explains when to use TRL-based training vs a custom GRPO loop for MedAgentBench-style tool-augmented clinical tasks.

## Context

Our training tasks are not plain next-token prediction. They involve:

- Multi-turn tool usage (FHIR GET/POST calls)
- Large structured tool schemas
- Potentially large tool responses
- Episode-level success/failure rewards

These properties can cause context growth, prompt/env desynchronization, and reward collapse if not handled carefully.

## Two Training Paths

### 1) TRL-Based Path

Example: `medagentbench_env/train.py` using `GRPOTrainer`.

**Strengths**

- Faster to set up and iterate.
- Reuses battle-tested trainer internals.
- Less code to maintain.

**Risks**

- Easier to accidentally desync prompt and environment episode state.
- Less explicit control over rollout-to-loss truncation semantics.
- Can hide failure modes (e.g., long prompts, no tool calls, zero rewards) behind trainer abstraction.

### 2) Custom GRPO Loop

Example: `rl_training/train.py` + `rl_training/env.py` + `rl_training/policy.py`.

**Strengths**

- Full control over rollout collection and task binding.
- Full control over context management and truncation behavior.
- Easier debugging of reward path and tool-call behavior.

**Costs**

- More code and maintenance.
- More responsibility for correctness and regression testing.

## Long-Prompt Handling: What Actually Works

In the custom path, long-context stability comes from layered safeguards:

1. **History cap in env**
   - Keep system prompt + last `N` messages only.
2. **Tool response truncation**
   - Clip very large tool outputs before appending to conversation.
3. **Tokenizer/model window truncation**
   - Left-truncate prompt when near model max context.
   - During loss computation, preserve response tokens and trim older prompt first.
4. **Episode cap**
   - Max steps per episode prevents unbounded context growth.

These safeguards are independent and complementary.

## Can TRL Include These Techniques?

Mostly yes, but with caveats:

- **Yes**: prompt/completion caps (e.g., max prompt length, max completion length).
- **Yes**: env-side controls (history cap, tool-output truncation) if implemented in your environment code.
- **Partial**: exact truncation semantics in loss path; may require deeper customization depending on TRL internals/version.
- **Partial**: strict prompt↔env deterministic binding; must be explicitly engineered.

Summary: TRL can implement most guardrails, but custom gives stronger guarantees and observability.

## Decision Matrix

- **Use TRL-first** when:
  - You want rapid experimentation.
  - Task complexity is moderate.
  - You can tolerate some abstraction and less fine-grained control.

- **Use custom-first** when:
  - Tool traces are long or highly variable.
  - Rewards are sparse and debugging is critical.
  - You need deterministic episode/task binding and auditable behavior.

- **Use hybrid (recommended for this project)** when:
  - You want TRL productivity but keep custom context-safety controls in env/policy boundaries.

## Recommended Project Policy (Hybrid)

Keep TRL orchestration where practical, but always enforce:

1. Deterministic task binding between sampled prompt and env episode.
2. Message-history cap in environment observations.
3. Tool-response truncation/summarization before history append.
4. Explicit max prompt and max completion bounds.
5. Max steps per episode.
6. Logging metrics for:
   - `tools/call_frequency`
   - `completions/clipped_ratio`
   - reward mean/std
   - terminal success rate

If these metrics degrade (e.g., tool calls near zero, clipped ratio near 1.0, reward std near 0), switch the run to the custom loop for diagnosis.

## Failure Signatures and Interpretation

- **`tools/call_frequency ~ 0`**  
  Model is not using tools; likely formatting/prompting mismatch.

- **`completions/clipped_ratio ~ 1` and terminated length ~ 0**  
  Completion budget is exhausted before useful action.

- **Very large token sequence warnings**  
  Prompt/history growth exceeded context safeguards.

- **`reward_std = 0` for long spans**  
  No meaningful learning signal; likely repeated failure policy.

## Operational Guidance

For production-quality runs:

- Prefer the custom loop (`rl_training`) when stability and traceability matter most.
- Keep TRL path for fast prototyping and ablation experiments.
- Promote changes from TRL prototype to custom pipeline once behavior is validated.

