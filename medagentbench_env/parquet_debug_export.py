#!/usr/bin/env python3
"""
Convert TRL rollout completion parquet files into readable debug artifacts.

Outputs:
  - completions_readable.csv
  - completions_readable.jsonl
  - summary.json

Example:
  python medagentbench_env/parquet_debug_export.py \
    --input-dir /workspace/Healthcare/clinKriya/output_rollout_e3/completions \
    --output-dir /workspace/Healthcare/clinKriya/output_rollout_e3/debug_readable
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pyarrow.parquet as pq


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _collect_rows(parquet_files: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for file_path in parquet_files:
        table = pq.read_table(file_path)
        cols = table.column_names
        col_data = {c: table[c].to_pylist() for c in cols}
        n_rows = len(next(iter(col_data.values()))) if col_data else 0

        for idx in range(n_rows):
            raw = {c: col_data[c][idx] for c in cols}
            completion = _safe_text(raw.get("completion"))
            prompt = _safe_text(raw.get("prompt"))

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
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export readable rollout debug tables from parquet files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing completions_*.parquet")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write csv/jsonl/summary")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(args.input_dir.glob("completions_*.parquet"))
    rows = _collect_rows(parquet_files)

    csv_path = args.output_dir / "completions_readable.csv"
    jsonl_path = args.output_dir / "completions_readable.jsonl"
    summary_path = args.output_dir / "summary.json"

    if not rows:
        summary_path.write_text(
            json.dumps(
                {
                    "num_files": len(parquet_files),
                    "num_rows": 0,
                    "message": "No parquet rows found.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"No rows found. Wrote {summary_path}")
        return

    df = pd.DataFrame(rows)
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

    print(f"Wrote {csv_path}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
