#!/usr/bin/env python3
"""
Full training run analysis: steps 1-404, all phases combined.

Sources:
  - Steps   1-150: output_v3/checkpoint-150/trainer_state.json  (60-task dataset)
  - Steps 151-404: output_v3_resume{3-6}.log                    (360-task dataset)

Outputs:
  docs/full_run_curves.png      — 9-panel training metrics
  docs/full_run_task_summary.png — per-task-type reward + episode breakdown
"""

import re
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BASE = Path(__file__).parent

# ── regex helpers ──────────────────────────────────────────────────────────────
def _grab(pattern, line):
    m = re.search(pattern, line)
    return float(m.group(1)) if m else None

_PBAR_RE   = re.compile(r"\|\s*(\d+)/\d+\s*\[")
_METRIC_KEYS = ["reward", "reward_std", "loss", "grad_norm", "kl",
                "entropy", "frac_reward_zero_std", "step_time",
                "tools/call_frequency", "completions/mean_length",
                "completions/clipped_ratio", "learning_rate", "epoch"]

def _parse_metric_line(line):
    row = {}
    for k in _METRIC_KEYS:
        pat = rf"'{re.escape(k)}': ['\"]?(-?[\d.e+-]+)['\"]?"
        row[k] = _grab(pat, line)
    return row if row["reward"] is not None else None


# ── Phase 1: trainer_state.json (steps 1-150) ──────────────────────────────────
def load_phase1():
    path = BASE / "output_v3/checkpoint-150/trainer_state.json"
    state = json.loads(path.read_text())
    records = []
    for e in state.get("log_history", []):
        if "reward" not in e:
            continue
        records.append({
            "step":                        e["step"],
            "epoch":                       e.get("epoch"),
            "reward":                      e.get("reward"),
            "reward_std":                  e.get("reward_std"),
            "loss":                        e.get("loss"),
            "grad_norm":                   e.get("grad_norm"),
            "kl":                          e.get("kl"),
            "entropy":                     e.get("entropy"),
            "frac_reward_zero_std":        e.get("frac_reward_zero_std"),
            "step_time":                   e.get("step_time"),
            "tools/call_frequency":        e.get("tools/call_frequency"),
            "completions/mean_length":     e.get("completions/mean_length"),
            "completions/clipped_ratio":   e.get("completions/clipped_ratio"),
            "learning_rate":               e.get("learning_rate"),
            "phase":                       1,   # 60-task phase
        })
    return records


# ── Phase 2: resume logs (steps 151-404) ──────────────────────────────────────
def load_phase2():
    logs = [
        "output_v3_resume3.log",
        "output_v3_resume4.log",
        "output_v3_resume5.log",
        "output_v3_resume6.log",
    ]
    seen = {}
    for lname in logs:
        path = BASE / lname
        if not path.exists():
            continue
        last_pbar = None
        with open(path) as f:
            for line in f:
                pm = _PBAR_RE.search(line)
                if pm:
                    last_pbar = int(pm.group(1))
                if "'reward':" not in line:
                    continue
                row = _parse_metric_line(line)
                if not row:
                    continue
                # Determine step
                sm = re.search(r"'step': ['\"]?(\d+)['\"]?", line)
                step = int(sm.group(1)) if sm else last_pbar
                if step is None:
                    continue
                row["step"] = step
                row["phase"] = 2   # 360-task phase
                seen[step] = row   # last log wins (most recent resume takes precedence)
    return list(seen.values())


# ── Episode trace parsing (for per-task-type breakdown) ────────────────────────
_TRACE_RE = re.compile(
    r"EPISODE TRACE\s+task=(\S+)\s+steps=(\d+)\s+reward=(-?[\d.e+-]+)"
)

def load_episodes():
    """Parse all EPISODE TRACE lines from all resume logs."""
    episodes = []
    for lname in ["output_v3_resume3.log","output_v3_resume4.log",
                  "output_v3_resume5.log","output_v3_resume6.log"]:
        path = BASE / lname
        if not path.exists():
            continue
        current_step = None
        last_pbar = None
        with open(path) as f:
            for line in f:
                pm = _PBAR_RE.search(line)
                if pm:
                    last_pbar = int(pm.group(1))
                m = _TRACE_RE.search(line)
                if m:
                    task_id, steps, reward = m.group(1), int(m.group(2)), float(m.group(3))
                    task_type = re.sub(r"_\d+$", "", task_id)          # e.g. task9_25 → task9
                    task_type = re.sub(r"^v2_", "", task_type)          # v2_task9 → task9
                    episodes.append({
                        "task_id":   task_id,
                        "task_type": task_type,
                        "steps":     steps,
                        "reward":    reward,
                        "global_step": last_pbar,
                    })
    # Deduplicate: same task_id + global_step → keep last
    seen = {}
    for ep in episodes:
        seen[(ep["task_id"], ep["global_step"])] = ep
    return list(seen.values())


# ── Merge + sort ───────────────────────────────────────────────────────────────
def build_records():
    p1 = load_phase1()
    p2 = load_phase2()
    merged = {}
    for r in p1 + p2:
        merged[r["step"]] = r
    return sorted(merged.values(), key=lambda r: r["step"])


# ── Smoothing ──────────────────────────────────────────────────────────────────
def smooth(vals, w=8):
    vals = np.array(vals, dtype=float)
    if len(vals) < w:
        return vals
    return np.convolve(vals, np.ones(w)/w, mode="same")


# ── Plot 1: training metrics ───────────────────────────────────────────────────
def plot_metrics(records, out_path):
    steps   = [r["step"] for r in records]
    rewards = [r["reward"] for r in records]
    rstds   = [r["reward_std"] or 0 for r in records]
    losses  = [r["loss"] or 0 for r in records]
    gnorms  = [r["grad_norm"] or 0 for r in records]
    kls     = [r["kl"] or 0 for r in records]
    ents    = [r["entropy"] or 0 for r in records]
    fzero   = [r["frac_reward_zero_std"] or 0 for r in records]
    stimes  = [r["step_time"] or 0 for r in records]
    tools   = [r["tools/call_frequency"] or 0 for r in records]
    mlens   = [r["completions/mean_length"] or 0 for r in records]
    phases  = [r["phase"] for r in records]

    # Phase boundary
    p2_start = next((s for s, p in zip(steps, phases) if p == 2), None)

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        "GRPO Training — Full Run  (Steps 1–404 | ~2.2 Data Passes)\n"
        "Phase 1: steps 1–150, 60-task dataset  |  Phase 2: steps 151–404, 360-task dataset",
        fontsize=12, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

    def ax(r, c): return fig.add_subplot(gs[r, c])

    def add_phase_line(a):
        if p2_start:
            a.axvline(p2_start, color="black", lw=1, linestyle=":", alpha=0.6, label="→ 360-task")

    # 1. Reward
    a = ax(0, 0)
    a.fill_between(steps,
                   [r-s for r,s in zip(rewards, rstds)],
                   [r+s for r,s in zip(rewards, rstds)],
                   alpha=0.15, color="steelblue")
    a.plot(steps, rewards, alpha=0.3, color="steelblue", lw=1)
    a.plot(steps, smooth(rewards), color="steelblue", lw=2, label="reward")
    a.axhline(0.5, color="green", lw=0.8, ls="--", alpha=0.5, label="0.5 target")
    add_phase_line(a)
    a.set_title("Reward (mean ± std)"); a.set_xlabel("step"); a.set_ylabel("reward")
    a.legend(fontsize=7)

    # 2. Reward std
    a = ax(0, 1)
    a.plot(steps, rstds, alpha=0.3, color="orange", lw=1)
    a.plot(steps, smooth(rstds), color="orange", lw=2)
    a.axhline(0.1, color="red", lw=0.8, ls="--", alpha=0.5, label="min healthy")
    add_phase_line(a)
    a.set_title("Reward Std (gradient signal)"); a.set_xlabel("step"); a.legend(fontsize=7)

    # 3. Frac zero std
    a = ax(0, 2)
    a.plot(steps, fzero, color="red", lw=1.5)
    add_phase_line(a)
    a.set_ylim(-0.05, 1.05)
    a.set_title("Frac Zero Reward Std\n(dead batches — want ~0)"); a.set_xlabel("step")

    # 4. Loss
    a = ax(1, 0)
    a.plot(steps, losses, alpha=0.3, color="purple", lw=1)
    a.plot(steps, smooth(losses), color="purple", lw=2)
    a.axhline(0, color="black", lw=0.5, ls="--")
    add_phase_line(a)
    a.set_title("Loss"); a.set_xlabel("step")

    # 5. Grad norm
    a = ax(1, 1)
    a.plot(steps, gnorms, alpha=0.3, color="brown", lw=1)
    a.plot(steps, smooth(gnorms), color="brown", lw=2)
    add_phase_line(a)
    a.set_title("Gradient Norm"); a.set_xlabel("step")

    # 6. KL divergence
    a = ax(1, 2)
    a.plot(steps, kls, alpha=0.3, color="teal", lw=1)
    a.plot(steps, smooth(kls), color="teal", lw=2)
    add_phase_line(a)
    a.set_title("KL Divergence (from ref)"); a.set_xlabel("step")

    # 7. Entropy
    a = ax(2, 0)
    a.plot(steps, ents, alpha=0.3, color="green", lw=1)
    a.plot(steps, smooth(ents), color="green", lw=2)
    add_phase_line(a)
    a.set_title("Entropy (exploration)"); a.set_xlabel("step")

    # 8. Tool calls
    a = ax(2, 1)
    a.plot(steps, tools, alpha=0.3, color="navy", lw=1)
    a.plot(steps, smooth(tools), color="navy", lw=2)
    a.axhline(6, color="red", lw=0.8, ls="--", alpha=0.6, label="cap=6")
    add_phase_line(a)
    a.set_title("Tool Call Frequency"); a.set_xlabel("step"); a.legend(fontsize=7)

    # 9. Step time + completion length
    a = ax(2, 2)
    a2 = a.twinx()
    a.plot(steps, stimes, alpha=0.3, color="gray", lw=1)
    a.plot(steps, smooth(stimes), color="gray", lw=2, label="step time (s)")
    a2.plot(steps, mlens, color="salmon", lw=1.5, ls="--", label="mean len")
    add_phase_line(a)
    a.set_title("Step Time & Completion Length"); a.set_xlabel("step")
    a.set_ylabel("seconds", color="gray"); a2.set_ylabel("tokens", color="salmon")
    lines = a.get_legend_handles_labels()[0] + a2.get_legend_handles_labels()[0]
    labs  = a.get_legend_handles_labels()[1] + a2.get_legend_handles_labels()[1]
    a.legend(lines, labs, fontsize=7)

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ── Plot 2: per-task-type breakdown ────────────────────────────────────────────
TASK_LABELS = {
    "task1":  "T1: CT Abdomen\n(overdue → order scan)",
    "task2":  "T2: DVT prophylaxis\n(check/create/deduplicate)",
    "task3":  "T3: Heart rate avg\n(compute 6h & 12h mean)",
    "task4":  "T4: Urinary catheter\n(>48h → removal order)",
    "task5":  "T5: Renal CT + referral\n(multi-step: dx → order×2)",
    "task6":  "T6: TSH/free T4\n(thyroid protocol)",
    "task7":  "T7: QTc prolonged\n(discontinue meds + ECG)",
    "task8":  "T8: Opioid + naloxone\n(pair check → order)",
    "task9":  "T9: Flu vaccine\n(>365d → order vaccine)",
    "task10": "T10: COVID booster\n(>12mo → order booster)",
}

def plot_task_summary(episodes, records, out_path):
    # Per-task-type reward distribution
    by_type = defaultdict(list)
    for ep in episodes:
        by_type[ep["task_type"]].append(ep["reward"])

    task_types = sorted(by_type.keys(), key=lambda t: int(re.sub(r"\D","",t) or 0))
    means  = [np.mean(by_type[t]) for t in task_types]
    stds   = [np.std(by_type[t]) for t in task_types]
    counts = [len(by_type[t]) for t in task_types]
    labels = [TASK_LABELS.get(t, t) for t in task_types]

    # Reward over training, sliding window per task type (first 5 types only for readability)
    # group episodes by global_step bucket (every 20 steps)
    def bucket_rewards(ttype, bucket=20):
        eps = [(e["global_step"], e["reward"]) for e in episodes
               if e["task_type"] == ttype and e["global_step"] is not None]
        if not eps:
            return [], []
        eps.sort()
        buckets = defaultdict(list)
        for gs, r in eps:
            buckets[(gs // bucket) * bucket].append(r)
        xs = sorted(buckets)
        ys = [np.mean(buckets[x]) for x in xs]
        return xs, ys

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "Per-Task-Type Analysis  (Phase 2: 360-task run, steps 151–404)\n"
        "10 task types × 30 instances each = 300 tasks (+ 60 new-patient tasks)",
        fontsize=12, fontweight="bold"
    )
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.4)

    # Panel 1: mean reward by task type (bar chart)
    a = fig.add_subplot(gs[0, :2])
    colors = plt.cm.RdYlGn(np.array(means) * 0.8 + 0.1)
    bars = a.bar(range(len(task_types)), means, yerr=stds, capsize=4,
                 color=colors, edgecolor="gray", linewidth=0.5)
    a.set_xticks(range(len(task_types)))
    a.set_xticklabels([t.replace("task","T") for t in task_types], fontsize=9)
    a.set_ylabel("Mean Reward"); a.set_ylim(0, 1.1)
    a.axhline(0.5, color="green", lw=1, ls="--", alpha=0.7, label="0.5")
    a.set_title("Mean Reward by Task Type (± std)"); a.legend(fontsize=8)
    for i, (m, n) in enumerate(zip(means, counts)):
        a.text(i, m + stds[i] + 0.02, f"n={n}", ha="center", fontsize=7, color="gray")

    # Panel 2: reward distribution violin/box
    a = fig.add_subplot(gs[0, 2:])
    data = [by_type[t] for t in task_types]
    bp = a.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    a.set_xticks(range(1, len(task_types)+1))
    a.set_xticklabels([t.replace("task","T") for t in task_types], fontsize=9)
    a.axhline(0.5, color="green", lw=1, ls="--", alpha=0.5)
    a.set_title("Reward Distribution by Task Type"); a.set_ylabel("reward")

    # Panels 3-12: reward trend per task type over training
    colors_trend = plt.cm.tab10(np.linspace(0, 1, len(task_types)))
    for i, tt in enumerate(task_types):
        row = 1 + i // 4
        col = i % 4
        a = fig.add_subplot(gs[row, col])
        xs, ys = bucket_rewards(tt, bucket=25)
        if xs:
            a.plot(xs, ys, color=colors_trend[i], lw=1.5, marker="o", markersize=3)
            # trend line
            if len(xs) >= 3:
                z = np.polyfit(xs, ys, 1)
                xf = np.linspace(min(xs), max(xs), 50)
                a.plot(xf, np.polyval(z, xf), color="black", lw=1, ls="--", alpha=0.5)
        a.axhline(0.5, color="green", lw=0.6, ls="--", alpha=0.4)
        a.set_ylim(-0.1, 1.15)
        a.set_title(TASK_LABELS.get(tt, tt), fontsize=8)
        a.set_xlabel("step", fontsize=7); a.set_ylabel("reward", fontsize=7)
        a.tick_params(labelsize=7)

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ── Text summary ───────────────────────────────────────────────────────────────
def print_summary(records, episodes):
    steps   = [r["step"] for r in records]
    rewards = [r["reward"] for r in records]
    p1 = [r for r in records if r["phase"] == 1]
    p2 = [r for r in records if r["phase"] == 2]

    print("\n" + "="*65)
    print("FULL RUN SUMMARY  (Steps 1–404)")
    print("="*65)
    print(f"  Total steps logged : {len(records)}")
    print(f"  Phase 1 (60 tasks) : steps 1–{p1[-1]['step'] if p1 else '?'}  "
          f"reward {np.mean([r['reward'] for r in p1]):.3f}")
    print(f"  Phase 2 (360 tasks): steps {p2[0]['step'] if p2 else '?'}–{p2[-1]['step'] if p2 else '?'}  "
          f"reward {np.mean([r['reward'] for r in p2]):.3f}")
    print(f"")
    n = len(p2)
    thirds = [p2[:n//3], p2[n//3:2*n//3], p2[2*n//3:]]
    trend = " → ".join(f"{np.mean([r['reward'] for r in t]):.3f}" for t in thirds if t)
    print(f"  Phase 2 trend (early/mid/late): {trend}")
    print(f"  Phase 2 peak: {max(r['reward'] for r in p2):.4f} @ step "
          f"{max(p2, key=lambda r: r['reward'])['step']}")
    print(f"  Final 10-step avg : {np.mean([r['reward'] for r in records[-10:]]):.4f}")
    print()

    # Per-task summary
    by_type = defaultdict(list)
    for ep in episodes:
        by_type[ep["task_type"]].append(ep["reward"])

    print(f"  {'Task':<10} {'N':>5} {'Mean':>7} {'Std':>7} {'Pass(>0.5)':>10}  Description")
    print(f"  {'-'*8:<10} {'-'*4:>5} {'-'*6:>7} {'-'*6:>7} {'-'*9:>10}  -----------")
    task_types = sorted(by_type.keys(), key=lambda t: int(re.sub(r"\D","",t) or 0))
    for tt in task_types:
        vals = by_type[tt]
        ppass = sum(1 for v in vals if v > 0.5) / len(vals)
        short = TASK_LABELS.get(tt, tt).replace("\n", " ")
        print(f"  {tt:<10} {len(vals):>5} {np.mean(vals):>7.3f} {np.std(vals):>7.3f} {ppass:>9.1%}  {short}")
    print("="*65 + "\n")

    # Episode stats
    total_eps = len(episodes)
    zero_eps  = sum(1 for e in episodes if e["reward"] == 0)
    neg_eps   = sum(1 for e in episodes if e["reward"] < 0)
    full_eps  = sum(1 for e in episodes if e["reward"] >= 1.0)
    cap_eps   = sum(1 for e in episodes if e["steps"] >= 6)
    print(f"  Episode breakdown ({total_eps} total):")
    print(f"    Reward = 0       : {zero_eps:>5} ({zero_eps/total_eps:.1%})")
    print(f"    Reward < 0       : {neg_eps:>5} ({neg_eps/total_eps:.1%})")
    print(f"    Reward ≥ 1.0     : {full_eps:>5} ({full_eps/total_eps:.1%})")
    print(f"    Hit step cap (6) : {cap_eps:>5} ({cap_eps/total_eps:.1%})")
    print("="*65 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    out_dir = BASE / "docs"
    out_dir.mkdir(exist_ok=True)

    print("Loading training records...")
    records = build_records()
    print(f"  {len(records)} steps loaded (steps {records[0]['step']}–{records[-1]['step']})")

    print("Loading episode traces...")
    episodes = load_episodes()
    print(f"  {len(episodes)} unique episodes loaded")

    print_summary(records, episodes)

    print("Plotting training metrics...")
    plot_metrics(records, str(out_dir / "full_run_curves.png"))

    print("Plotting task-type breakdown...")
    plot_task_summary(episodes, records, str(out_dir / "full_run_task_summary.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
