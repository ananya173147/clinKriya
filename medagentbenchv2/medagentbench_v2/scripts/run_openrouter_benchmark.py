"""
Benchmark runner for closed-source models via OpenRouter.
Runs the stratified_benchmark.json against multiple models and reports pass rates.

Usage:
    cd medagentbenchv2
    python -m scripts.run_openrouter_benchmark --output-dir ./results
"""

import sys
import os
import json
import queue
import threading
import time
import argparse
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[3] / ".env")

sys.path.insert(0, str(Path(__file__).parents[1]))

from openai import OpenAI
from src.evals import MedAgentBench
from src.wrapper import MedAgentBenchWrapper, TaskResult
from src.agent import MedAgentResult
import src.tool.patient_search as patient_search
import src.tool.vitals_search as vitals_search
import src.tool.observation_search as observation_search
import src.tool.medication_request_search as medication_request_search
import src.tool.medication_request_create as medication_request_create
import src.tool.service_request_create as service_request_create
import src.tool.vitals_create as vitals_create
import src.tool.calculator as calculator_create
import src.tool.procedure_search as procedure_search
import src.tool.condition_search as condition_search
import src.tool.finish as finish

# ---------------------------------------------------------------------------
# OpenRouter model IDs
# ---------------------------------------------------------------------------
MODELS = {
    "claude-3.5-sonnet-v2": "anthropic/claude-3.5-sonnet",
    "gpt-5.2":               "openai/gpt-5.2",
    "deepseek-v3":           "deepseek/deepseek-v3.2",
    "gemini-1.5-pro":        "google/gemini-3.1-pro-preview",  # 1.5 unavailable on OpenRouter; using 3.1 Pro
}

FHIR_API_BASE = "http://localhost:8080/fhir/"
TASKS_PATH = str(Path(__file__).parent.parent / "src" / "MedAgentBench" / "data" / "medagentbench" / "stratified_benchmark.json")


# ---------------------------------------------------------------------------
# OpenRouter agent (Chat Completions API — compatible with all providers)
# ---------------------------------------------------------------------------
class OpenRouterAgent:
    def __init__(self, model: str, system_prompt: str, fhir_api_base: str):
        self.model = model
        self.system_prompt = system_prompt
        self.fhir_api_base = fhir_api_base

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

        # Build tools (same as MedAgent)
        from src.tool.base import Tool
        self.tools: list[Tool] = [
            patient_search.create(fhir_api_base),
            vitals_search.create(fhir_api_base),
            observation_search.create(fhir_api_base),
            medication_request_search.create(fhir_api_base),
            medication_request_create.create(fhir_api_base),
            service_request_create.create(fhir_api_base),
            vitals_create.create(fhir_api_base),
            procedure_search.create(fhir_api_base),
            condition_search.create(fhir_api_base),
            calculator_create.create(),
            finish.create(),
        ]
        self.tools_registry = {t.name: t for t in self.tools}

        # Convert Responses-API schema → Chat Completions schema
        self.tool_schemas = [self._to_chat_schema(t.json_schema()) for t in self.tools]

    def _to_chat_schema(self, schema: dict) -> dict:
        """Convert from OpenAI Responses API format to Chat Completions format."""
        inner = {k: v for k, v in schema.items() if k != "type"}
        return {"type": "function", "function": inner}

    def run_iter(self, instruction: str, context: str = None, max_steps: int = 8):
        run_id = str(uuid4())
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._user_message(instruction, context)},
        ]

        for _ in range(max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto",
                temperature=0,
            )

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if msg.content:
                yield {"type": "message", "content": msg.content}

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                call_id = tc.id
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                yield {
                    "type": "tool_call",
                    "name": tool_name,
                    "arguments": args,
                    "call_id": call_id,
                }

                tool_obj = self.tools_registry.get(tool_name)
                if tool_obj is None:
                    result = f"Unknown tool: {tool_name}"
                else:
                    try:
                        tool_inputs = tool_obj.input_schema.model_validate(args)
                        result = tool_obj(tool_inputs)
                    except Exception as e:
                        result = f"Tool error: {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": str(result),
                })

                yield {
                    "type": "tool_output",
                    "output": result,
                    "call_id": call_id,
                }

                if tool_name == "finish":
                    value = args.get("value", [])
                    yield {"type": "finish", "id": run_id, "value": value}
                    return

        yield {"type": "finish", "id": run_id, "value": []}

    def run(self, instruction: str, context: str = None, max_steps: int = 8, verbose: bool = False):
        trace = []
        for item in self.run_iter(instruction=instruction, context=context, max_steps=max_steps):
            trace.append(item)
            if item.get("type") == "finish":
                return MedAgentResult(id=item["id"], value=item["value"], trace=trace)
        return MedAgentResult(id=None, value=[], trace=trace)

    def _user_message(self, instruction: str, context: str = None) -> str:
        msg = f"<instruction>\n{instruction}\n</instruction>\n"
        if context:
            msg += f"<context>\n{context}\n</context>\n"
        return msg


# ---------------------------------------------------------------------------
# Wrapper adapter so OpenRouterAgent works with MedAgentBenchWrapper
# ---------------------------------------------------------------------------
class OpenRouterWrapper(MedAgentBenchWrapper):
    def __init__(self, agent: OpenRouterAgent):
        # Skip parent __init__ — just set what we need
        self.agent = agent
        from urllib.parse import urljoin
        base = agent.fhir_api_base
        self.api_mapping = {
            "patient_search":                ("GET",  urljoin(base, "Patient")),
            "fhir_vitals_create":             ("POST", urljoin(base, "Observation")),
            "fhir_medication_request_create": ("POST", urljoin(base, "MedicationRequest")),
            "fhir_service_request_create":    ("POST", urljoin(base, "ServiceRequest")),
            "fhir_observation_search":        ("GET",  urljoin(base, "Observation")),
            "fhir_medication_request_search": ("GET",  urljoin(base, "MedicationRequest")),
            "fhir_vitals_search":             ("GET",  urljoin(base, "Observation")),
            "fhir_procedure_search":          ("GET",  urljoin(base, "Procedure")),
            "fhir_condition_search":          ("GET",  urljoin(base, "Condition")),
        }

    def _run(self, task: dict, max_steps: int = 8, verbose: bool = False):
        return self.agent.run(
            instruction=task["instruction"],
            context=task.get("context", ""),
            max_steps=max_steps,
            verbose=verbose,
        )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def eval_worker(medagentbench, task_queue, wrapper, output_dir):
    while True:
        task_id = task_queue.get()
        if task_id is None:
            task_queue.task_done()
            break

        task_path = os.path.join(output_dir, f"{task_id}.jsonl")
        if os.path.exists(task_path):
            print(f"[SKIP] {task_id} already done")
            task_queue.task_done()
            continue

        print(f"[RUN ] {task_id}")
        try:
            task = medagentbench.get_task_by_id(task_id)
            task_result, trace = wrapper.run(task, max_steps=8, verbose=False)
            with open(task_path, "w") as f:
                json.dump({"result": task_result.result}, f)
                f.write("\n")
                for step in trace:
                    json.dump(step, f)
                    f.write("\n")
            print(f"[DONE] {task_id}")
        except Exception as e:
            print(f"[ERR ] {task_id}: {e}")
            time.sleep(2)
            task_queue.put(task_id)
        finally:
            task_queue.task_done()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_results(medagentbench, results_dir, wrapper):
    from collections import defaultdict
    type_pass = defaultdict(int)
    type_total = defaultdict(int)
    failed = []

    for fname in os.listdir(results_dir):
        if not fname.endswith(".jsonl"):
            continue
        task_id = fname[:-6]
        task_type = task_id.split("_")[0]

        with open(os.path.join(results_dir, fname)) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        if not lines:
            continue

        from src.agent import MedAgentResult
        agent_result = MedAgentResult(
            id=task_id,
            value=json.loads(lines[0]["result"]),
            trace=lines[1:],
        )
        task_result = wrapper._to_task_result(agent_result)
        success = medagentbench.evaluate_task(task_id, task_result)

        type_total[task_type] += 1
        if success:
            type_pass[task_type] += 1
        else:
            failed.append(task_id)

    total = sum(type_total.values())
    passed = sum(type_pass.values())
    overall = passed / total * 100 if total else 0

    return {
        "overall": overall,
        "passed": passed,
        "total": total,
        "by_task": {t: {"pass": type_pass[t], "total": type_total[t],
                        "pct": type_pass[t]/type_total[t]*100}
                    for t in sorted(type_total)},
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(Path(__file__).parent.parent / "benchmark_results"), help="Root output dir")
    p.add_argument("--tasks-path", default=TASKS_PATH)
    p.add_argument("--num-workers", type=int, default=5)
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                   help="Which models to run (default: all)")
    p.add_argument("--score-only", action="store_true",
                   help="Skip running, just re-score existing results")
    return p.parse_args()


def main():
    args = parse_args()

    prompt_path = Path(__file__).parent.parent / "src" / "prompts" / "new_system.txt"
    with open(prompt_path) as f:
        system_prompt = f.read()

    os.makedirs(args.output_dir, exist_ok=True)

    medagentbench = MedAgentBench(tasks_path=args.tasks_path, api_base=FHIR_API_BASE)
    all_task_ids = [t["id"] for t in medagentbench.get_tasks()]

    print(f"Benchmark: {len(all_task_ids)} tasks")
    print(f"Models:    {args.models}\n")

    all_scores = {}

    for model_key in args.models:
        model_id = MODELS.get(model_key, model_key)
        model_dir = os.path.join(args.output_dir, model_key, "tasks")
        os.makedirs(model_dir, exist_ok=True)

        agent = OpenRouterAgent(
            model=model_id,
            system_prompt=system_prompt,
            fhir_api_base=FHIR_API_BASE,
        )
        wrapper = OpenRouterWrapper(agent)

        if not args.score_only:
            # Filter already-completed
            pending = [tid for tid in all_task_ids
                       if not os.path.exists(os.path.join(model_dir, f"{tid}.jsonl"))]
            print(f"[{model_key}] Running {len(pending)}/{len(all_task_ids)} tasks...")

            task_queue = queue.Queue()
            for tid in pending:
                task_queue.put(tid)
            for _ in range(args.num_workers):
                task_queue.put(None)

            workers = [
                threading.Thread(target=eval_worker,
                                 args=(medagentbench, task_queue, wrapper, model_dir))
                for _ in range(args.num_workers)
            ]
            for w in workers:
                w.start()
            task_queue.join()
            for w in workers:
                w.join()

        # Score
        print(f"[{model_key}] Scoring...")
        scores = score_results(medagentbench, model_dir, wrapper)
        all_scores[model_key] = scores

        print(f"  Overall: {scores['overall']:.1f}%  ({scores['passed']}/{scores['total']})")
        for ttype, s in scores["by_task"].items():
            print(f"    {ttype}: {s['pct']:.0f}%  ({s['pass']}/{s['total']})")
        print()

    # Final comparison table
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    header = f"{'Task':<12}" + "".join(f"{m:<18}" for m in args.models)
    print(header)
    print("-"*70)

    task_types = sorted({t for scores in all_scores.values() for t in scores["by_task"]})
    for ttype in task_types:
        row = f"{ttype:<12}"
        for m in args.models:
            s = all_scores.get(m, {}).get("by_task", {}).get(ttype, {})
            pct = f"{s['pct']:.0f}%" if s else "n/a"
            row += f"{pct:<18}"
        print(row)

    print("-"*70)
    overall_row = f"{'OVERALL':<12}"
    for m in args.models:
        s = all_scores.get(m, {})
        pct = f"{s.get('overall', 0):.1f}%"
        overall_row += f"{pct:<18}"
    print(overall_row)

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
