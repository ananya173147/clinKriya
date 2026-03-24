#!/usr/bin/env python3
"""
GRPO Training Script for MedAgentBench — multi-task clinical decision-making.

Trains a Qwen2.5-7B-Instruct model using Group Relative Policy Optimization
(GRPO) to improve its performance across all RL-worthy MedAgentBench tasks:
task1, task2, task4–task10, and v2_task5/9/10 (Mg/K+/A1c from test_data_v2).
Each task requires the agent to read a patient's FHIR chart and decide whether
to take clinical action (order labs, stop medications, place referrals, etc.).

Usage:
    # Local model training (all task types, default config):
    python -m rl_training.train

    # API-based rollout collection (for faster iteration with vLLM):
    python -m rl_training.train \
        --api-url http://localhost:8001/v1 \
        --api-key EMPTY

    # Restrict to specific task types:
    python -m rl_training.train \
        --task-types task7 task5 task10

    # Custom configuration:
    python -m rl_training.train \
        --epochs 100 \
        --group-size 4 \
        --lr 5e-7 \
        --output-dir ./my_rl_output

    # Load an existing train/test split instead of generating a new one:
    python -m rl_training.train \
        --split-file rl_training/train_test_split.json

Pipeline:
    1. Load task data from new_patient_tasks.json (+ test_data_v2.json for v2 types)
    2. Create or load a stratified 80/20 train/test split per task type
    3. Initialise policy (Qwen2.5-7B-Instruct + LoRA) and MedAgentEnv
    4. For each epoch:
       a. Sample a batch of tasks from the training pool
       b. For each task, collect K rollouts (GRPO group)
       c. Compute group-relative advantages: A_k = (r_k - mean) / std
       d. Update policy with clipped PPO-style gradient (one step per group)
    5. Periodically evaluate on held-out test tasks and save best checkpoint
    6. Final evaluation over the full test set; save final checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from .config import RLConfig, ModelConfig, TrainConfig, PathConfig, FHIRConfig
from .env import MedAgentEnv, make_env
from .policy import PolicyModel
from .rollout import (
    Trajectory,
    RolloutGroup,
    collect_rollout,
    collect_group_rollouts,
)
from .fhir_reset import verify_fhir_server


# ─── Rollout Serialization ───────────────────────────────────────────────────


def _serialize_tool_call(tc) -> dict:
    """Convert a tool call (ParsedToolCall or dict) to a JSON-safe dict."""
    if hasattr(tc, "name"):  # ParsedToolCall
        return {"name": tc.name, "arguments": tc.arguments, "call_id": getattr(tc, "call_id", "")}
    if isinstance(tc, dict):
        func = tc.get("function", {})
        args = func.get("arguments", tc.get("arguments", "{}"))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                pass
        return {
            "name": func.get("name", tc.get("name", "")),
            "arguments": args,
            "call_id": tc.get("id", tc.get("call_id", "")),
        }
    return {"name": str(tc), "arguments": {}, "call_id": ""}


def _serialize_turn(turn) -> dict:
    """Serialize a TurnRecord to a JSON-safe dict."""
    # Parse tool_calls from assistant_response
    tool_calls_raw = turn.assistant_response.get("tool_calls", [])
    tool_calls = [_serialize_tool_call(tc) for tc in tool_calls_raw]

    return {
        "step": turn.step,
        "generated_text": turn.generated_text,
        "content": turn.assistant_response.get("content"),
        "tool_calls": tool_calls,
        "step_reward": round(turn.step_reward, 4),
    }


def _serialize_trajectory(traj, rollout_idx: int, advantage: float | None = None) -> dict:
    """Serialize a Trajectory to a JSON-safe dict."""
    ep_rewards = traj.episode_rewards
    if ep_rewards is not None and hasattr(ep_rewards, "to_dict"):
        ep_rewards = ep_rewards.to_dict()

    return {
        "task_id": traj.task_id,
        "rollout_idx": rollout_idx,
        "total_reward": round(traj.total_reward, 4),
        "terminal_pass": traj.terminal_pass,
        "episode_rewards": ep_rewards,
        "num_turns": traj.num_turns,
        "duration_s": round(traj.duration_seconds, 2),
        "advantage": round(advantage, 4) if advantage is not None else None,
        "turns": [_serialize_turn(t) for t in traj.turns],
    }


def _serialize_group(group, group_idx: int) -> dict:
    """Serialize a RolloutGroup to a JSON-safe dict."""
    advantages = group.compute_advantages()
    return {
        "group_idx": group_idx,
        "task_id": group.task_id,
        "mean_reward": round(group.mean_reward, 4),
        "std_reward": round(group.std_reward, 4),
        "rollouts": [
            _serialize_trajectory(traj, k, adv)
            for k, (traj, adv) in enumerate(zip(group.trajectories, advantages))
        ],
    }


def save_epoch_rollouts(
    output_dir: Path,
    epoch: int,
    groups: list,
) -> Path:
    """Save all rollout groups for one epoch to a JSON file."""
    rollouts_dir = output_dir / "rollouts"
    rollouts_dir.mkdir(exist_ok=True)

    data = {
        "epoch": epoch,
        "num_groups": len(groups),
        "groups": [_serialize_group(g, i) for i, g in enumerate(groups)],
    }

    out_path = rollouts_dir / f"epoch_{epoch}.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return out_path


# ─── Train/Test Split ────────────────────────────────────────────────────────


def load_train_test_split(split_file: str) -> tuple[list[str], list[str]]:
    """
    Load train/test split from a JSON file.

    The JSON file should have the format:
        {"train": ["task7_1", ...], "test": ["task7_2", ...]}

    Args:
        split_file: Path to the split JSON file

    Returns:
        (train_ids, test_ids)
    """
    with open(split_file, "r") as f:
        split_data = json.load(f)

    train_ids = sorted(split_data["train"])
    test_ids = sorted(split_data["test"])

    print(f"  Loaded split from: {split_file}")
    return train_ids, test_ids


def create_train_test_split(
    tasks_path: str,
    rl_task_types: tuple | list = ("task1", "task2", "task4", "task5", "task6",
                                   "task7", "task8", "task9", "task10"),
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Create a deterministic stratified train/test split across all RL task types.

    For each task type, 80% of instances go to train and 20% to test,
    ensuring every task type is represented in both sets.

    Args:
        tasks_path: Path to the tasks JSON file (new_patient_tasks.json or test_data_v2.json)
        rl_task_types: Task type prefixes to include
        train_ratio: Fraction of each task type allocated to training
        seed: Random seed for reproducibility

    Returns:
        (train_ids, test_ids)
    """
    with open(tasks_path, "r") as f:
        all_tasks = json.load(f)

    rng = random.Random(seed)
    train_ids: list[str] = []
    test_ids: list[str] = []

    for task_type in sorted(rl_task_types):
        ids = sorted([
            t["id"] for t in all_tasks
            if t["id"].startswith(f"{task_type}_")
        ])
        if not ids:
            continue
        rng.shuffle(ids)
        split_idx = max(1, int(len(ids) * train_ratio))
        train_ids.extend(ids[:split_idx])
        test_ids.extend(ids[split_idx:])

    return sorted(train_ids), sorted(test_ids)


# ─── GRPO Update ────────────────────────────────────────────────────────────


def grpo_update(
    policy: PolicyModel,
    optimizer: torch.optim.Optimizer,
    groups: list[RolloutGroup],
    config: RLConfig,
) -> dict:
    """
    Perform one GRPO policy gradient update using collected rollout groups.

    For each group (K rollouts of the same task):
      1. Compute advantages: A_k = (r_k - mean(r)) / (std(r) + eps)
      2. For each trajectory k:
         - Re-compute log probs of the generated text under current policy
         - Compute clipped PPO loss weighted by advantage
      3. Aggregate losses and update

    Returns:
        Training statistics dict
    """
    policy.train_mode()

    total_loss = 0.0
    num_trajectories = 0
    total_advantage = 0.0
    total_reward = 0.0
    grad_norms = []

    # ── One optimizer step per group (proper GRPO) ────────────────────────────
    # Advantages are computed within each group; gradients accumulate across all
    # trajectories in the group before a single optimizer.step().
    # Previously: zero_grad() was called inside the trajectory loop, making K=1
    # independent updates instead of one group-relative update.
    for group in groups:
        advantages = group.compute_advantages(eps=config.train.advantage_eps)

        optimizer.zero_grad()    # reset once per group, not per trajectory
        group_loss = torch.tensor(0.0)
        group_trajectories = 0

        for traj, advantage in zip(group.trajectories, advantages):
            if not traj.turns:
                continue

            traj_loss = torch.tensor(0.0, requires_grad=True)
            num_turns_with_loss = 0

            for turn in traj.turns:
                if not turn.generated_text or not turn.log_probs:
                    continue

                try:
                    turn_loss = policy.compute_loss(
                        messages=turn.messages_before,
                        response_text=turn.generated_text,
                        advantage=advantage,
                        old_log_probs=turn.log_probs,
                        clip_range=config.train.clip_range,
                        kl_coef=config.train.kl_coef,
                        tools=None,
                    )
                    traj_loss = traj_loss + turn_loss
                    num_turns_with_loss += 1
                except Exception as e:
                    print(f"  [WARN] Loss computation failed for turn: {e}")
                    continue

            if num_turns_with_loss > 0:
                traj_loss = traj_loss / num_turns_with_loss
                # Accumulate gradient for this trajectory
                traj_loss.backward()
                group_loss = group_loss + traj_loss.detach()
                group_trajectories += 1
                total_advantage += advantage
                total_reward += traj.total_reward

        # Single optimizer step for the whole group
        if group_trajectories > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.get_trainable_parameters(),
                config.train.max_grad_norm,
            )
            grad_norms.append(grad_norm.item())
            optimizer.step()

            total_loss += (group_loss / group_trajectories).item()
            num_trajectories += group_trajectories

    policy.eval_mode()

    stats = {
        "loss": total_loss / max(len(groups), 1),
        "num_trajectories": num_trajectories,
        "mean_advantage": total_advantage / max(num_trajectories, 1),
        "mean_reward": total_reward / max(num_trajectories, 1),
        "mean_grad_norm": (
            sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        ),
    }
    return stats


# ─── Evaluation ──────────────────────────────────────────────────────────────


def evaluate(
    env: MedAgentEnv,
    policy: PolicyModel,
    test_ids: list[str],
    num_episodes: int = 10,
    verbose: bool = False,
) -> tuple[dict, list[Trajectory]]:
    """
    Evaluate the policy on held-out test tasks.

    Returns:
        (eval_stats_dict, list_of_trajectories)
    """
    policy.eval_mode()

    # Randomly sample test IDs so every task type gets coverage over time
    if len(test_ids) > num_episodes:
        eval_ids = random.sample(test_ids, num_episodes)
    else:
        eval_ids = list(test_ids)

    results = []
    trajectories: list[Trajectory] = []
    for task_id in eval_ids:
        try:
            traj = collect_rollout(
                env=env,
                policy=policy,
                task_id=task_id,
                verbose=verbose,
            )
            results.append({
                "task_id": task_id,
                "total_reward": traj.total_reward,
                "terminal_pass": traj.terminal_pass,
                "num_turns": traj.num_turns,
                "episode_rewards": traj.episode_rewards,
            })
            trajectories.append(traj)
        except Exception as e:
            print(f"  [EVAL ERROR] {task_id}: {e}")
            results.append({
                "task_id": task_id,
                "total_reward": 0.0,
                "terminal_pass": False,
                "num_turns": 0,
                "episode_rewards": None,
            })

    n = len(results)
    n_pass = sum(r["terminal_pass"] for r in results)
    mean_reward = sum(r["total_reward"] for r in results) / n if n else 0.0

    stats = {
        "num_tasks": n,
        "num_pass": n_pass,
        "pass_rate": n_pass / n if n else 0.0,
        "mean_reward": mean_reward,
        "per_task": results,
    }
    return stats, trajectories


def save_eval_rollouts(
    output_dir: Path,
    label: str,
    trajectories: list[Trajectory],
) -> Path:
    """Save evaluation/inference rollouts to a JSON file.

    Args:
        output_dir: Run output directory.
        label: A label string used in the filename, e.g. "eval_epoch_5" or "final_eval".
        trajectories: List of Trajectory objects from evaluation.

    Returns:
        Path to the saved file.
    """
    rollouts_dir = output_dir / "rollouts"
    rollouts_dir.mkdir(exist_ok=True)

    data = {
        "label": label,
        "num_trajectories": len(trajectories),
        "trajectories": [
            _serialize_trajectory(traj, idx)
            for idx, traj in enumerate(trajectories)
        ],
    }

    out_path = rollouts_dir / f"{label}.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return out_path


# ─── Main Training Loop ─────────────────────────────────────────────────────


def train(config: RLConfig, split_file: str | None = None):
    """Main GRPO training loop."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.paths.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    task_types_str = ", ".join(sorted(config.train.rl_task_types))
    print("═" * 65)
    print("  GRPO Training: MedAgentBench – Multi-Task Clinical RL")
    print("═" * 65)
    print(f"  Model         : {config.model.model_name}")
    print(f"  LoRA          : {config.model.use_lora}")
    print(f"  FHIR server   : {config.fhir.api_base}")
    print(f"  Epochs        : {config.train.num_epochs}")
    print(f"  Group size (K): {config.train.group_size}")
    print(f"  Episodes/epoch: {config.train.episodes_per_epoch}")
    print(f"  Learning rate : {config.train.learning_rate}")
    print(f"  Task types    : {task_types_str}")
    print(f"  Output        : {output_dir}")
    print("═" * 65)

    # ── Verify FHIR server ────────────────────────────────────────────
    if not verify_fhir_server(config.fhir.api_base):
        raise RuntimeError(
            f"FHIR server not reachable at {config.fhir.api_base}\n"
            "Start it with: docker run -p 8080:8080 jyxsu6/medagentbench:latest"
        )
    print("✓ FHIR server OK\n")

    # ── Create / load train/test split ──────────────────────────────
    if split_file and Path(split_file).is_file():
        train_ids, test_ids = load_train_test_split(split_file)
        print(f"  (loaded from {split_file})")
    else:
        train_ids, test_ids = create_train_test_split(
            config.paths.tasks_path,
            rl_task_types=config.train.rl_task_types,
            train_ratio=config.train.train_ratio,
            seed=config.train.seed,
        )
        print("  (generated new split)")
    print(f"Train tasks ({len(train_ids)}): {train_ids}")
    print(f"Test tasks  ({len(test_ids)}): {test_ids}\n")

    # Save split (always save a copy in the run directory)
    with open(output_dir / "train_test_split.json", "w") as f:
        json.dump({"train": train_ids, "test": test_ids}, f, indent=2)

    # ── Initialise environment ────────────────────────────────────────
    env = make_env(
        config=config,
        task_ids=train_ids,
        task_types=list(config.train.rl_task_types),
        auto_reset_fhir=True,
        verbose=False,
    )
    eval_env = make_env(
        config=config,
        task_ids=test_ids,
        task_types=list(config.train.rl_task_types),
        auto_reset_fhir=True,
        verbose=False,
    )
    print("✓ Environment created\n")

    # ── Initialise policy ─────────────────────────────────────────────
    policy = PolicyModel(config.model)
    print("✓ Policy model loaded\n")

    # ── Optimiser ─────────────────────────────────────────────────────
    if not config.model.api_url:
        optimizer = torch.optim.AdamW(
            policy.get_trainable_parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        # Learning rate scheduler with warmup
        total_steps = config.train.num_epochs * config.train.episodes_per_epoch
        warmup_steps = int(total_steps * config.train.warmup_ratio)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (
                min(1.0, step / max(warmup_steps, 1))
                if step < warmup_steps
                else max(0.1, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))
            ),
        )
    else:
        optimizer = None
        scheduler = None

    # ── Training log ──────────────────────────────────────────────────
    training_log = []

    # ── Training loop ─────────────────────────────────────────────────
    best_eval_reward = -float("inf")

    for epoch in range(1, config.train.num_epochs + 1):
        epoch_start = time.time()
        print(f"\n{'='*65}")
        print(f"  Epoch {epoch}/{config.train.num_epochs}")
        print(f"{'='*65}")

        # Sample tasks for this epoch
        epoch_task_ids = random.sample(
            train_ids,
            min(config.train.episodes_per_epoch, len(train_ids)),
        )

        # ── Collect rollouts ──────────────────────────────────────────
        print("\n  Collecting rollouts…")
        groups: list[RolloutGroup] = []

        for i, task_id in enumerate(epoch_task_ids):
            print(f"\n  [{i+1}/{len(epoch_task_ids)}] Task: {task_id}")

            group = collect_group_rollouts(
                env=env,
                policy=policy,
                task_id=task_id,
                group_size=config.train.group_size,
                verbose=False,
            )
            groups.append(group)

            # Print group summary
            rewards = group.rewards
            passes = sum(t.terminal_pass for t in group.trajectories)
            print(
                f"    Rewards: {[f'{r:.2f}' for r in rewards]}  "
                f"Passes: {passes}/{config.train.group_size}  "
                f"Mean: {group.mean_reward:.3f}"
            )

        # ── Compute epoch statistics ──────────────────────────────────
        all_rewards = [t.total_reward for g in groups for t in g.trajectories]
        all_passes = [t.terminal_pass for g in groups for t in g.trajectories]
        epoch_mean_reward = sum(all_rewards) / len(all_rewards)
        epoch_pass_rate = sum(all_passes) / len(all_passes)

        # ── Policy update ─────────────────────────────────────────────
        if optimizer is not None:
            print("\n  Updating policy…")
            update_stats = grpo_update(policy, optimizer, groups, config)
            if scheduler is not None:
                scheduler.step()

            print(
                f"    Loss: {update_stats['loss']:.4f}  "
                f"Grad norm: {update_stats['mean_grad_norm']:.4f}  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            update_stats = {"loss": 0.0, "mean_grad_norm": 0.0}
            print("\n  [API mode] Skipping policy update.")

        epoch_duration = time.time() - epoch_start

        # ── Epoch summary ─────────────────────────────────────────────
        epoch_log = {
            "epoch": epoch,
            "mean_reward": round(epoch_mean_reward, 4),
            "pass_rate": round(epoch_pass_rate, 4),
            "loss": round(update_stats["loss"], 6),
            "grad_norm": round(update_stats["mean_grad_norm"], 4),
            "duration_s": round(epoch_duration, 1),
        }
        training_log.append(epoch_log)

        print(
            f"\n  ── Epoch {epoch} Summary ──\n"
            f"    Mean reward : {epoch_mean_reward:.3f}\n"
            f"    Pass rate   : {epoch_pass_rate*100:.1f}%\n"
            f"    Duration    : {epoch_duration:.1f}s"
        )

        # ── Save rollouts ────────────────────────────────────────────
        rollout_path = save_epoch_rollouts(output_dir, epoch, groups)
        print(f"    Rollouts saved → {rollout_path}")

        # ── Evaluation ────────────────────────────────────────────────
        if epoch % config.train.eval_every == 0:
            print(f"\n  ── Evaluation (epoch {epoch}) ──")
            eval_results, eval_trajs = evaluate(
                env=eval_env,
                policy=policy,
                test_ids=test_ids,
                num_episodes=config.train.eval_episodes,
                verbose=False,
            )

            print(
                f"    Eval pass rate : {eval_results['pass_rate']*100:.1f}%  "
                f"({eval_results['num_pass']}/{eval_results['num_tasks']})\n"
                f"    Eval mean reward: {eval_results['mean_reward']:.3f}"
            )

            epoch_log["eval_pass_rate"] = round(eval_results["pass_rate"], 4)
            epoch_log["eval_mean_reward"] = round(eval_results["mean_reward"], 4)

            # Save best model
            if eval_results["mean_reward"] > best_eval_reward:
                best_eval_reward = eval_results["mean_reward"]
                if optimizer is not None:
                    best_path = str(checkpoints_dir / "best")
                    policy.save_checkpoint(best_path)
                    print(f"    ★ New best model saved (reward={best_eval_reward:.3f})")

            # Save eval results
            eval_path = output_dir / f"eval_epoch_{epoch}.json"
            with open(eval_path, "w") as f:
                json.dump(eval_results, f, indent=2)

            # Save eval rollouts
            eval_rollout_path = save_eval_rollouts(
                output_dir, f"eval_epoch_{epoch}", eval_trajs
            )
            print(f"    Eval rollouts saved → {eval_rollout_path}")

        # ── Checkpoint ────────────────────────────────────────────────
        if epoch % config.train.save_every == 0 and optimizer is not None:
            ckpt_path = str(checkpoints_dir / f"epoch_{epoch}")
            policy.save_checkpoint(ckpt_path)

        # ── Save training log ─────────────────────────────────────────
        with open(output_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  Training Complete!")
    print("═" * 65)
    print(f"  Total epochs: {config.train.num_epochs}")
    print(f"  Best eval reward: {best_eval_reward:.3f}")
    print(f"  Output directory: {output_dir}")

    # Final evaluation
    print("\n  ── Final Evaluation ──")
    final_eval, final_trajs = evaluate(
        env=eval_env,
        policy=policy,
        test_ids=test_ids,
        num_episodes=len(test_ids),
        verbose=True,
    )
    print(
        f"\n  Final pass rate: {final_eval['pass_rate']*100:.1f}%  "
        f"({final_eval['num_pass']}/{final_eval['num_tasks']})\n"
        f"  Final mean reward: {final_eval['mean_reward']:.3f}"
    )

    with open(output_dir / "final_eval.json", "w") as f:
        json.dump(final_eval, f, indent=2)

    # Save final eval rollouts
    final_rollout_path = save_eval_rollouts(output_dir, "final_eval", final_trajs)
    print(f"  Final eval rollouts saved → {final_rollout_path}")

    # Save final model
    if optimizer is not None:
        final_path = str(checkpoints_dir / "final")
        policy.save_checkpoint(final_path)

    print(f"\n  All results saved → {output_dir}")
    return training_log


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO training for MedAgentBench — multi-task clinical RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name or path",
    )
    p.add_argument(
        "--api-url", default=None,
        help="OpenAI-compatible API URL for rollout collection (None = local model)",
    )
    p.add_argument(
        "--api-key", default=None,
        help="API key (use 'EMPTY' for local vLLM)",
    )
    p.add_argument("--no-lora", action="store_true", help="Disable LoRA")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")

    # Data split
    p.add_argument(
        "--split-file", default=None,
        help="Path to a train_test_split.json file. If provided, the train/test "
             "split is loaded from this file instead of being generated. "
             "Format: {\"train\": [...], \"test\": [...]}",
    )

    # Training
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p.add_argument(
        "--episodes-per-epoch", type=int, default=8,
        help="Tasks per epoch",
    )
    p.add_argument("--group-size", type=int, default=4, help="GRPO group size K")
    p.add_argument("--max-steps", type=int, default=12, help="Max steps per episode")
    p.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    p.add_argument("--kl-coef", type=float, default=0.05, help="KL penalty coefficient")
    p.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    p.add_argument(
        "--task-types", nargs="+",
        default=["task1", "task2", "task4", "task5", "task6",
                 "task7", "task8", "task9", "task10"],
        help="RL task types to include in training",
    )
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="Fraction of each task type allocated to training")
    p.add_argument("--train-size", type=int, default=20, help="Legacy: training set size (task7-only mode)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Evaluation
    p.add_argument("--eval-every", type=int, default=5, help="Evaluate every N epochs")
    p.add_argument("--eval-episodes", type=int, default=10, help="Episodes per evaluation")

    # Output
    p.add_argument("--output-dir", default=None, help="Output directory")
    p.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")

    # FHIR
    p.add_argument(
        "--fhir-url", default="http://localhost:8080/fhir/",
        help="FHIR server base URL",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Build config from CLI args
    config = RLConfig(
        paths=PathConfig(
            output_dir=args.output_dir or str(
                Path(__file__).resolve().parent.parent / "rl_output"
            ),
        ),
        fhir=FHIRConfig(api_base=args.fhir_url),
        model=ModelConfig(
            model_name=args.model,
            api_url=args.api_url,
            api_key=args.api_key,
            use_lora=not args.no_lora,
            lora_r=args.lora_r,
        ),
        train=TrainConfig(
            num_epochs=args.epochs,
            episodes_per_epoch=args.episodes_per_epoch,
            group_size=args.group_size,
            max_steps_per_episode=args.max_steps,
            learning_rate=args.lr,
            kl_coef=args.kl_coef,
            clip_range=args.clip_range,
            rl_task_types=tuple(args.task_types),
            train_ratio=args.train_ratio,
            train_size=args.train_size,
            seed=args.seed,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            save_every=args.save_every,
        ),
    )

    # Set seeds
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.train.seed)

    train(config, split_file=args.split_file)


if __name__ == "__main__":
    main()
