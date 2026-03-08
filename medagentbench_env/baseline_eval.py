#!/usr/bin/env python3
"""
Baseline evaluation: run a model via OpenRouter against all MedAgentBench tasks.

Usage:
    python baseline_eval.py                       # all 90 tasks, default model
    python baseline_eval.py --num-tasks 2         # quick smoke test
    python baseline_eval.py --model qwen/qwen3-8b # different model
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Ensure the parent package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medagentbench_env.models import ActionType, MedAgentBenchAction
from medagentbench_env.server.medagentbench_env_environment import MedAgentBenchEnvironment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "qwen/qwen3-8b"
DEFAULT_OUTPUT = str(Path(__file__).resolve().parent / "data" / "baseline_results.json")


# ---------------------------------------------------------------------------
# OpenRouter API (via openai client, matching run_openrouter_benchmark.py)
# ---------------------------------------------------------------------------


def make_client(api_key: str) -> OpenAI:
    """Create an OpenAI client pointed at OpenRouter."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def call_openrouter(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    max_retries: int = 3,
) -> str:
    """Send a chat completion request to OpenRouter and return the reply text."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  API error ({e}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise

    return ""


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------


def parse_action(raw_text: str) -> MedAgentBenchAction:
    """Parse model output into a MedAgentBenchAction.

    Recognises three patterns:
      GET <url>
      POST <url>\n<json body>
      FINISH([...])
    Falls back to FINISH with empty answer on parse failure.
    """
    text = raw_text.strip()

    # --- FINISH ---
    finish_match = re.search(r"FINISH\((.+)\)", text, re.DOTALL)
    if finish_match:
        inner = finish_match.group(1).strip()
        try:
            answer = json.loads(inner)
            if not isinstance(answer, list):
                answer = [answer]
        except json.JSONDecodeError:
            answer = [inner]
        return MedAgentBenchAction(
            action_type=ActionType.FINISH,
            answer=answer,
            raw_response=raw_text,
        )

    # --- GET ---
    for line in text.splitlines():
        line_stripped = line.strip()
        if line_stripped.upper().startswith("GET "):
            url = line_stripped[4:].strip()
            return MedAgentBenchAction(
                action_type=ActionType.GET,
                url=url,
                raw_response=raw_text,
            )

    # --- POST ---
    for i, line in enumerate(text.splitlines()):
        line_stripped = line.strip()
        if line_stripped.upper().startswith("POST "):
            url = line_stripped[5:].strip()
            # Remaining lines form the JSON body
            body_lines = text.splitlines()[i + 1 :]
            body_text = "\n".join(body_lines).strip()
            body = None
            if body_text:
                try:
                    body = json.loads(body_text)
                except json.JSONDecodeError:
                    body = None
            return MedAgentBenchAction(
                action_type=ActionType.POST,
                url=url,
                body=body,
                raw_response=raw_text,
            )

    # --- Fallback: unparseable → FINISH with empty answer ---
    return MedAgentBenchAction(
        action_type=ActionType.FINISH,
        answer=[],
        raw_response=raw_text,
    )


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------


def run_task(
    env: MedAgentBenchEnvironment,
    task_index: int,
    model: str,
    client: OpenAI,
    max_retries: int,
) -> Dict[str, Any]:
    """Run one task and return its result dict (with trace)."""
    obs = env.reset(task_index=task_index)
    system_prompt = obs.response_text
    task_id = obs.task_id
    task_type = task_id.split("_")[0]

    # Conversation for OpenRouter (role: user/assistant)
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": system_prompt},
    ]
    # Full trace for output
    trace: List[Dict[str, str]] = [
        {"role": "user", "content": system_prompt},
    ]

    reward = 0.0
    task_status = "running"
    steps = 0

    while not obs.done:
        # Call model
        try:
            reply = call_openrouter(client, messages, model, max_retries)
        except Exception as e:
            print(f"  API error on task {task_id}: {e}")
            reply = "FINISH([])"

        messages.append({"role": "assistant", "content": reply})
        trace.append({"role": "assistant", "content": reply})

        # Parse action
        action = parse_action(reply)
        steps += 1

        # Step environment
        obs = env.step(action)

        env_response = obs.response_text
        messages.append({"role": "user", "content": env_response})
        trace.append({"role": "user", "content": env_response})

        if obs.done:
            reward = obs.reward
            task_status = obs.task_status.value

    return {
        "task_id": task_id,
        "task_type": task_type,
        "reward": round(reward, 4),
        "task_status": task_status,
        "steps": steps,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Baseline eval on MedAgentBench")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model ID")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to run (default: all 90)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max API retries per call",
    )
    args = parser.parse_args()

    # Load API key
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set. Add it to ../.env or environment.")
        sys.exit(1)

    # Create OpenRouter client
    client = make_client(api_key)

    # Create environment (uses mock FHIR cache automatically)
    env = MedAgentBenchEnvironment()
    total_tasks = len(env._tasks)
    num_tasks = args.num_tasks if args.num_tasks is not None else total_tasks

    print(f"Model:  {args.model}")
    print(f"Tasks:  {num_tasks} / {total_tasks}")
    print(f"Output: {args.output}")
    print()

    results: List[Dict[str, Any]] = []

    for i in range(num_tasks):
        task_idx = i % total_tasks
        print(f"[{i + 1}/{num_tasks}] Running task index {task_idx}...", end=" ", flush=True)
        try:
            result = run_task(env, task_idx, args.model, client, args.max_retries)
        except Exception as e:
            print(f"CRASH: {e}")
            result = {
                "task_id": f"task_idx_{task_idx}",
                "task_type": "unknown",
                "reward": 0.0,
                "task_status": "error",
                "steps": 0,
                "trace": [],
                "error": str(e),
            }
        results.append(result)
        print(
            f"{result['task_id']}  reward={result['reward']:.4f}  "
            f"status={result['task_status']}  steps={result['steps']}"
        )

    # --- Build summary ---
    avg_reward = sum(r["reward"] for r in results) / len(results) if results else 0.0
    by_type: Dict[str, Dict[str, Any]] = {}
    for r in results:
        tt = r["task_type"]
        if tt not in by_type:
            by_type[tt] = {"count": 0, "total_reward": 0.0}
        by_type[tt]["count"] += 1
        by_type[tt]["total_reward"] += r["reward"]

    by_type_summary = {
        tt: {"count": v["count"], "avg_reward": round(v["total_reward"] / v["count"], 4)}
        for tt, v in sorted(by_type.items())
    }

    output = {
        "model": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_tasks": len(results),
            "avg_reward": round(avg_reward, 4),
            "by_type": by_type_summary,
        },
        "results": results,
    }

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Console summary
    print()
    print("=" * 60)
    print(f"Results saved to {out_path}")
    print(f"Average reward: {avg_reward:.4f}")
    print()
    print("By task type:")
    for tt, info in by_type_summary.items():
        print(f"  {tt}: n={info['count']}  avg_reward={info['avg_reward']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
