"""Build an SFT dataset from v9 GRPO rollouts that passed the terminal grader.

Rejection-sampling SFT / expert iteration:
  - Filter completions with reward >= terminal_threshold
  - Group by task_id, keep top-K per task to avoid dominance
  - Emit chat-format JSONL consumable by transformers/TRL SFTTrainer

Outputs:
  training/sft_v1/train.jsonl   chat-formatted training samples
  training/sft_v1/stats.json    per-task coverage + length distribution
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def task_type_of(task_id: str) -> str:
    parts = task_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else task_id


def load_all_completions(comp_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(comp_dir.glob("completions_*.parquet")):
        frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_task_id_per_episode(stats_path: Path):
    """Return list of task_ids matching the completions row order (best effort).

    GRPO logs one task_stats record per env; with n_prompts * n_gens = n_comps.
    The order in task_stats.jsonl matches the prompt order. Within a step we
    have: all generations of prompt 1, then prompt 2, etc. Completions parquet
    is emitted in the same order.  So the Nth row in a given step's completions
    maps to task_stats[step * prompts_per_step + prompt_idx] where
    prompt_idx = row_idx // n_gens.
    """
    rows = [json.loads(l) for l in stats_path.read_text().splitlines() if l.strip()]
    return rows


def build_sft_samples(
    comp_dir: Path,
    stats_path: Path,
    tasks_file: Path,
    terminal_threshold: float = 1.0,
    max_per_task_id: int = 8,
    dedupe: bool = True,
):
    df = load_all_completions(comp_dir)
    if df.empty:
        print(f"No completions found in {comp_dir}")
        return []

    # Pair each completion row with its task_stats entry.
    # In TRL GRPO, each step has num_prompts envs, each produces num_gens completions.
    # task_stats.jsonl records ONCE per env (not per generation).
    stats_rows = load_task_id_per_episode(stats_path)
    tasks_by_id = {t["id"]: t for t in json.loads(tasks_file.read_text())}

    # Reconstruct mapping: for each step, determine how many stats rows and how
    # many completion rows were logged, and infer n_gens = n_comps / n_envs.
    df_by_step = {step: sub.reset_index(drop=True) for step, sub in df.groupby("step")}
    stats_by_step = defaultdict(list)
    for r in stats_rows:
        stats_by_step[r["step"]].append(r)

    demos = []
    for step, sub in df_by_step.items():
        stats = stats_by_step.get(step, [])
        if not stats:
            continue
        n_comps = len(sub)
        n_envs = len(stats)
        if n_comps % n_envs != 0:
            # mismatch; skip this step
            continue
        n_gens = n_comps // n_envs
        for i, row in sub.iterrows():
            env_idx = i // n_gens
            meta = stats[env_idx]
            # Use stats["reward"] (raw env reward, pre-bonus) as the filter, NOT
            # completion["reward_func"] (bonus-inflated). Only raw reward >= ~1.0
            # indicates an actual terminal grader pass.
            stats_reward = float(meta["reward"])
            if stats_reward < terminal_threshold:
                continue
            reward = stats_reward
            comp = row["completion"]
            if isinstance(comp, list):
                comp = "\n".join(str(c) for c in comp)
            prompt = row["prompt"]
            if isinstance(prompt, list):
                prompt = "\n".join(str(c) for c in prompt)
            task_id = meta["task_id"]
            task_type = meta["task_type"]
            task_spec = tasks_by_id.get(task_id, {})
            demos.append({
                "step": int(step),
                "task_id": task_id,
                "task_type": task_type,
                "workflow_branch": task_spec.get("workflow_branch", ""),
                "reward": reward,
                "tool_calls": meta.get("tool_calls", 0),
                "prompt": prompt,
                "completion": comp,
            })

    # Dedup identical completions
    if dedupe:
        seen = set()
        unique = []
        for d in demos:
            key = (d["task_id"], d["completion"][:500])
            if key in seen:
                continue
            seen.add(key)
            unique.append(d)
        demos = unique

    # Cap per task_id to avoid dominance
    by_id = defaultdict(list)
    for d in demos:
        by_id[d["task_id"]].append(d)
    capped = []
    for tid, items in by_id.items():
        items.sort(key=lambda x: (-x["reward"], -x["tool_calls"]))
        capped.extend(items[:max_per_task_id])
    return capped


def to_sft_messages(d: dict, system_prompt: str):
    """Convert a demo into a chat-format sample.

    We reconstruct:
      system (from new_system.txt)
      user   (the task instruction + context from prompt)
      assistant (the full completion = tool_calls + env responses + finish)
    """
    # The 'prompt' field from TRL contains the formatted chat prompt fed to the
    # model. For SFT we want:
    #   {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", "content": completion}]}
    # Use the raw completion as the assistant response.
    instruction = d["prompt"]
    # Strip leading Qwen chat template artifacts (keep the user/instruction content)
    if "<|im_start|>user\n" in instruction and "<|im_end|>" in instruction:
        try:
            user_block = instruction.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        except Exception:
            user_block = instruction
    else:
        user_block = instruction

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_block.strip()},
            {"role": "assistant", "content": d["completion"].strip()},
        ],
        "task_id": d["task_id"],
        "task_type": d["task_type"],
        "reward": d["reward"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp-dir", default=str(ROOT / "training/output_v9/completions"))
    ap.add_argument("--stats", default=str(ROOT / "training/output_v9_checkpoints/task_stats.jsonl"))
    ap.add_argument("--tasks", default=str(ROOT / "medagentbench_env/data/train_tasks.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "training/sft_v1"))
    ap.add_argument("--terminal-threshold", type=float, default=1.0)
    ap.add_argument("--max-per-task-id", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demos = build_sft_samples(
        Path(args.comp_dir),
        Path(args.stats),
        Path(args.tasks),
        terminal_threshold=args.terminal_threshold,
        max_per_task_id=args.max_per_task_id,
    )
    print(f"Collected {len(demos)} demos (reward >= {args.terminal_threshold})")

    # Stats
    by_type = Counter(d["task_type"] for d in demos)
    by_id = Counter(d["task_id"] for d in demos)
    by_branch = Counter((d["task_type"], d["workflow_branch"]) for d in demos)
    print("\nBy task_type:")
    for k, v in by_type.most_common():
        print(f"  {k:<12} {v}")
    print(f"\nBy branch (top 20):")
    for k, v in by_branch.most_common(20):
        print(f"  {k[0]:<12} {k[1]:<30} {v}")

    # Write SFT jsonl
    system_prompt = (ROOT / "medagentbench_env/data/new_system.txt").read_text().strip()
    out_path = out_dir / "train.jsonl"
    with out_path.open("w") as f:
        for d in demos:
            f.write(json.dumps(to_sft_messages(d, system_prompt)) + "\n")
    print(f"\nWrote {out_path}")

    stats = {
        "n_demos": len(demos),
        "by_task_type": dict(by_type),
        "by_task_id": dict(by_id),
        "by_branch": {f"{k[0]}/{k[1]}": v for k, v in by_branch.items()},
        "terminal_threshold": args.terminal_threshold,
        "max_per_task_id": args.max_per_task_id,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
