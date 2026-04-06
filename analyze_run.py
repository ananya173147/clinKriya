#!/usr/bin/env python3
"""Parse a GRPO training log and produce reward curves + summary."""

import re
import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


_METRIC_RE = re.compile(r"\{'loss': '([^']+)'.*?'reward': '([^']+)'.*?'reward_std': '([^']+)'.*?'frac_reward_zero_std': '([^']+)'.*?'kl': '([^']+)'.*?'entropy': '([^']+)'.*?'step_time': '([^']+)'.*?'epoch': '([^']+)'\}")
# Also try dict repr (no quotes around values)
_METRIC_RE2 = re.compile(r"\{'loss': (-?[\d.e+-]+).*?'reward': ([\d.e+-]+).*?'reward_std': ([\d.e+-]+).*?'frac_reward_zero_std': ([\d.e+-]+).*?'kl': ([\d.e+-]+).*?'entropy': ([\d.e+-]+).*?'step_time': ([\d.e+-]+).*?'epoch': ([\d.e+-]+)\}")
_STEP_RE = re.compile(r"'step': ['\"]?(\d+)['\"]?")
_GRAD_RE = re.compile(r"'grad_norm': ['\"]?([\d.e+-]+)['\"]?")
_TOOLS_RE = re.compile(r"'tools/call_frequency': ['\"]?([\d.e+-]+)['\"]?")
_LR_RE = re.compile(r"'learning_rate': ['\"]?([\d.e+-]+)['\"]?")
_CLIPPED_RE = re.compile(r"'completions/clipped_ratio': ['\"]?([\d.e+-]+)['\"]?")
_MEAN_LEN_RE = re.compile(r"'completions/mean_length': ['\"]?([\d.e+-]+)['\"]?")


_PBAR_RE = re.compile(r"\|\s*(\d+)/\d+\s*\[")


def parse_log(log_path: str) -> list[dict]:
    records = []
    last_pbar_step = None
    with open(log_path) as f:
        for line in f:
            pbar_m = _PBAR_RE.search(line)
            if pbar_m:
                last_pbar_step = int(pbar_m.group(1))

            if "'reward':" not in line:
                continue

            step_m = _STEP_RE.search(line)
            if step_m:
                step = int(step_m.group(1))
            elif last_pbar_step is not None:
                step = last_pbar_step
            else:
                continue

            def grab(pat, line=line):
                m = pat.search(line)
                return float(m.group(1)) if m else None

            records.append({
                "step": step,
                "epoch": grab(re.compile(r"'epoch': ['\"]?([\d.e+-]+)['\"]?")),
                "reward": grab(re.compile(r"'reward': ['\"]?([\d.e+-]+)['\"]?")),
                "reward_std": grab(re.compile(r"'reward_std': ['\"]?([\d.e+-]+)['\"]?")),
                "loss": grab(re.compile(r"'loss': ['\"]?(-?[\d.e+-]+)['\"]?")),
                "grad_norm": grab(_GRAD_RE),
                "kl": grab(re.compile(r"'kl': ['\"]?([\d.e+-]+)['\"]?")),
                "entropy": grab(re.compile(r"'entropy': ['\"]?([\d.e+-]+)['\"]?")),
                "frac_zero_std": grab(re.compile(r"'frac_reward_zero_std': ['\"]?([\d.e+-]+)['\"]?")),
                "step_time": grab(re.compile(r"'step_time': ['\"]?([\d.e+-]+)['\"]?")),
                "tools_freq": grab(_TOOLS_RE),
                "lr": grab(_LR_RE),
                "clipped_ratio": grab(_CLIPPED_RE),
                "mean_length": grab(_MEAN_LEN_RE),
            })
    # deduplicate by step, keep last
    seen = {}
    for r in records:
        seen[r["step"]] = r
    return sorted(seen.values(), key=lambda r: r["step"])


def smooth(vals, window=5):
    if len(vals) < window:
        return vals
    kernel = np.ones(window) / window
    return np.convolve(vals, kernel, mode="same")


def plot(records: list[dict], out_path: str, log_path: str):
    steps = [r["step"] for r in records]
    rewards = [r["reward"] for r in records]
    reward_stds = [r["reward_std"] for r in records]
    losses = [r["loss"] for r in records]
    grad_norms = [r["grad_norm"] for r in records]
    kls = [r["kl"] for r in records]
    entropies = [r["entropy"] for r in records]
    frac_zero = [r["frac_zero_std"] for r in records]
    step_times = [r["step_time"] for r in records]
    tools = [r["tools_freq"] for r in records]
    mean_lens = [r["mean_length"] for r in records]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"GRPO Training Run — {Path(log_path).stem}\nSteps {steps[0]}–{steps[-1]}  |  Epoch {records[0]['epoch']:.2f}–{records[-1]['epoch']:.2f}", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    def ax(row, col): return fig.add_subplot(gs[row, col])

    # 1. Reward mean + smoothed
    a = ax(0, 0)
    a.fill_between(steps, [r - s for r, s in zip(rewards, reward_stds)],
                   [r + s for r, s in zip(rewards, reward_stds)], alpha=0.2, color="steelblue", label="±std")
    a.plot(steps, rewards, alpha=0.4, color="steelblue", lw=1)
    a.plot(steps, smooth(rewards), color="steelblue", lw=2, label="reward (smoothed)")
    a.set_title("Reward (mean ± std)")
    a.set_xlabel("step"); a.set_ylabel("reward"); a.legend(fontsize=8)
    a.axhline(0.5, color="green", lw=0.8, linestyle="--", alpha=0.5)

    # 2. Reward std
    a = ax(0, 1)
    a.plot(steps, reward_stds, alpha=0.4, color="orange", lw=1)
    a.plot(steps, smooth(reward_stds), color="orange", lw=2)
    a.set_title("Reward Std (gradient signal)")
    a.set_xlabel("step"); a.set_ylabel("reward_std")
    a.axhline(0.1, color="red", lw=0.8, linestyle="--", alpha=0.5, label="min healthy")
    a.legend(fontsize=8)

    # 3. Frac zero std
    a = ax(0, 2)
    a.plot(steps, frac_zero, color="red", lw=1.5)
    a.set_title("Frac Reward Zero Std\n(dead batches — want low)")
    a.set_xlabel("step"); a.set_ylabel("fraction"); a.set_ylim(-0.05, 1.05)

    # 4. Loss
    a = ax(1, 0)
    a.plot(steps, losses, alpha=0.4, color="purple", lw=1)
    a.plot(steps, smooth(losses), color="purple", lw=2)
    a.set_title("Loss")
    a.set_xlabel("step"); a.set_ylabel("loss")
    a.axhline(0, color="black", lw=0.5, linestyle="--")

    # 5. Grad norm
    a = ax(1, 1)
    a.plot(steps, grad_norms, alpha=0.4, color="brown", lw=1)
    a.plot(steps, smooth(grad_norms), color="brown", lw=2)
    a.set_title("Gradient Norm")
    a.set_xlabel("step"); a.set_ylabel("grad_norm")

    # 6. KL divergence
    a = ax(1, 2)
    a.plot(steps, kls, alpha=0.4, color="teal", lw=1)
    a.plot(steps, smooth(kls), color="teal", lw=2)
    a.set_title("KL Divergence (from ref)")
    a.set_xlabel("step"); a.set_ylabel("kl")

    # 7. Entropy
    a = ax(2, 0)
    a.plot(steps, entropies, alpha=0.4, color="green", lw=1)
    a.plot(steps, smooth(entropies), color="green", lw=2)
    a.set_title("Entropy (exploration)")
    a.set_xlabel("step"); a.set_ylabel("entropy")

    # 8. Tool call frequency
    a = ax(2, 1)
    a.plot(steps, tools, alpha=0.4, color="navy", lw=1)
    a.plot(steps, smooth(tools), color="navy", lw=2)
    a.axhline(6, color="red", lw=0.8, linestyle="--", alpha=0.6, label="max_steps=6")
    a.set_title("Tool Call Frequency")
    a.set_xlabel("step"); a.set_ylabel("calls/episode"); a.legend(fontsize=8)

    # 9. Step time + mean completion length
    a = ax(2, 2)
    a2 = a.twinx()
    a.plot(steps, step_times, alpha=0.4, color="gray", lw=1)
    a.plot(steps, smooth(step_times), color="gray", lw=2, label="step time (s)")
    a2.plot(steps, mean_lens, color="salmon", lw=1.5, linestyle="--", label="mean len (tokens)")
    a.set_title("Step Time & Completion Length")
    a.set_xlabel("step"); a.set_ylabel("seconds", color="gray")
    a2.set_ylabel("tokens", color="salmon")
    lines1, lab1 = a.get_legend_handles_labels()
    lines2, lab2 = a2.get_legend_handles_labels()
    a.legend(lines1 + lines2, lab1 + lab2, fontsize=7)

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved plot → {out_path}")


def summary(records: list[dict], log_path: str):
    steps = [r["step"] for r in records]
    rewards = [r["reward"] for r in records]
    reward_stds = [r["reward_std"] for r in records]
    step_times = [r["step_time"] for r in records]
    frac_zero = [r["frac_zero_std"] for r in records]
    tools = [r["tools_freq"] for r in records]

    # Split into thirds for trend analysis
    n = len(records)
    thirds = [records[:n//3], records[n//3:2*n//3], records[2*n//3:]]
    third_names = ["early", "mid", "late"]

    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY — {Path(log_path).stem}")
    print(f"{'='*60}")
    print(f"Steps logged : {steps[0]} → {steps[-1]}  ({len(steps)} steps)")
    print(f"Epochs       : {records[0]['epoch']:.2f} → {records[-1]['epoch']:.2f}")
    print(f"")
    print(f"{'--- Reward ---':^40}")
    print(f"  Mean (all)  : {np.mean(rewards):.4f}")
    print(f"  Peak        : {max(rewards):.4f}  @ step {steps[rewards.index(max(rewards))]}")
    print(f"  Final 10    : {np.mean(rewards[-10:]):.4f}")
    nonempty = [t for t in thirds if t]
    trend_vals = " → ".join(f"{np.mean([r['reward'] for r in t]):.3f}" for t in nonempty)
    trend_labels = " / ".join(third_names[:len(nonempty)])
    print(f"  Trend       : {trend_vals}  ({trend_labels})")
    print(f"")
    print(f"{'--- Gradient Signal ---':^40}")
    print(f"  Reward std mean   : {np.mean(reward_stds):.4f}")
    print(f"  Frac zero-std     : {np.mean(frac_zero):.2%}  (dead batches)")
    print(f"")
    print(f"{'--- Efficiency ---':^40}")
    print(f"  Avg step time     : {np.mean(step_times):.1f}s")
    print(f"  Median step time  : {np.median(step_times):.1f}s")
    print(f"  Slow steps (>300s): {sum(1 for t in step_times if t > 300)}")
    print(f"  Avg tool calls    : {np.mean(tools):.2f}  (max_steps=6)")
    print(f"  Over-step rate    : {sum(1 for t in tools if t > 6) / len(tools):.1%}  (hitting step cap)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="Path to training log file")
    parser.add_argument("--out", default=None, help="Output PNG path (default: docs/<log_stem>_curves.png)")
    args = parser.parse_args()

    records = parse_log(args.log)
    if not records:
        print("No metric lines found in log.")
        sys.exit(1)

    out = args.out or str(Path(args.log).parent / "docs" / (Path(args.log).stem + "_curves.png"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    summary(records, args.log)
    plot(records, out, args.log)


if __name__ == "__main__":
    main()
