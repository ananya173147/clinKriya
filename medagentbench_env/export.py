"""
MedAgentBench export utilities.

Post-training helpers that write reward curves and readable completion
debug files. No dependency on the training loop or environment.

Public API
----------
export_reward_graph(output_dir, log_history) -> None
export_completions_debug(output_dir) -> None
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def export_reward_graph(output_dir: str, log_history: List[Dict[str, Any]]) -> None:
    """Export reward metrics CSV and a reward-curve PNG.

    Parameters
    ----------
    output_dir : str
        Directory to write reward_metrics.csv and reward_curve.png.
    log_history : list of dict
        trainer.state.log_history from a HuggingFace Trainer.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[int, float, float]] = []
    for item in log_history:
        if "rewards/reward_func/mean" not in item and "reward" not in item:
            continue
        step = item.get("step")
        if not isinstance(step, (int, float)):
            continue
        reward_mean = item.get("rewards/reward_func/mean", item.get("reward", 0.0))
        reward_std = item.get("rewards/reward_func/std", item.get("reward_std", 0.0))
        try:
            rows.append((int(step), float(reward_mean), float(reward_std)))
        except Exception:
            continue

    if not rows:
        print("No reward history found; skipped reward graph export.")
        return

    rows.sort(key=lambda x: x[0])
    csv_path = out_dir / "reward_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "reward_mean", "reward_std"])
        writer.writerows(rows)
    print(f"Saved reward metrics CSV: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipped reward_curve.png")
        return

    steps = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [max(0.0, r[2]) for r in rows]
    lows = [m - s for m, s in zip(means, stds)]
    highs = [m + s for m, s in zip(means, stds)]

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, means, linewidth=2.0, label="reward mean")
    plt.fill_between(steps, lows, highs, alpha=0.2, label="reward std")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    png_path = out_dir / "reward_curve.png"
    plt.savefig(png_path, dpi=140)
    plt.close()
    print(f"Saved reward graph: {png_path}")


def export_completions_debug(output_dir: str) -> None:
    """Write readable CSV/JSONL from rollout parquet logs under output_dir/completions/.

    Parameters
    ----------
    output_dir : str
        Training output directory; completions parquets expected under
        ``output_dir/completions/completions_*.parquet``.
    """
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except Exception:
        print("pandas/pyarrow unavailable; skipped completion debug export.")
        return

    completions_dir = Path(output_dir) / "completions"
    parquet_files = sorted(completions_dir.glob("completions_*.parquet"))
    if not parquet_files:
        print(f"No completion parquets found under {completions_dir}; skipped debug export.")
        return

    debug_dir = Path(output_dir) / "debug_readable"
    debug_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for file_path in parquet_files:
        table = pq.read_table(file_path)
        cols = table.column_names
        col_data = {c: table[c].to_pylist() for c in cols}
        n_rows = len(next(iter(col_data.values()))) if col_data else 0
        for idx in range(n_rows):
            raw = {c: col_data[c][idx] for c in cols}
            completion = str(raw.get("completion") or "")
            prompt = str(raw.get("prompt") or "")
            tool_names = re.findall(r'"name"\s*:\s*"([^"]+)"', completion)
            rows.append(
                {
                    "source_file": file_path.name,
                    "row_idx": idx,
                    "step": raw.get("step"),
                    "reward_func": raw.get("reward_func"),
                    "advantage": raw.get("advantage"),
                    "tool_calls": completion.count("<tool_call>"),
                    "has_finish": bool(re.search(r'"name"\s*:\s*"finish"', completion)),
                    "tool_names": "|".join(tool_names[:12]),
                    "completion_chars": len(completion),
                    "prompt_chars": len(prompt),
                    "completion_preview": completion[:500].replace("\n", " "),
                }
            )

    if not rows:
        print("Completion parquet files had no rows; skipped debug export.")
        return

    df = pd.DataFrame(rows)
    csv_path = debug_dir / "completions_readable.csv"
    jsonl_path = debug_dir / "completions_readable.jsonl"
    summary_path = debug_dir / "summary.json"

    df.to_csv(csv_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "num_files": len(parquet_files),
        "num_rows": int(len(df)),
        "finish_rate": float(df["has_finish"].mean()),
        "mean_tool_calls": float(df["tool_calls"].mean()),
        "mean_completion_chars": float(df["completion_chars"].mean()),
        "files": [p.name for p in parquet_files],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved completion debug CSV: {csv_path}")
    print(f"Saved completion debug JSONL: {jsonl_path}")
    print(f"Saved completion debug summary: {summary_path}")
