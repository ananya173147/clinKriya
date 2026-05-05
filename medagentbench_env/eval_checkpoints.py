"""Held-out evaluation harness for clinKriya checkpoints.

Runs a model (base + optional LoRA adapters merged) on the held-out test
set (23 tasks in data/test_tasks.json) and reports per-task + overall metrics.

Usage:
    # Evaluate base Qwen3-1.7B
    python -m medagentbench_env.eval_checkpoints \
        --model Qwen/Qwen3-1.7B \
        --output-dir training/eval/base

    # Evaluate SFT-only (LoRA adapter)
    python -m medagentbench_env.eval_checkpoints \
        --model Qwen/Qwen3-1.7B \
        --adapter training/sft_ckpt \
        --output-dir training/eval/sft_only

    # Evaluate v9 (pure GRPO)
    python -m medagentbench_env.eval_checkpoints \
        --model Qwen/Qwen3-1.7B \
        --adapter training/output_v9 \
        --output-dir training/eval/v9

    # Evaluate v11 (SFT→GRPO) — SFT merged into base then v11 adapter on top
    python -m medagentbench_env.eval_checkpoints \
        --model Qwen/Qwen3-1.7B \
        --pre-merge-adapter training/sft_ckpt \
        --adapter training/output_v11 \
        --output-dir training/eval/v11

Multiple --adapter or --pre-merge-adapter can be given; they're applied in
order (pre-merge first, then adapter).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_call(text: str):
    """Return (name, arguments_dict) of the LAST tool_call block, or None."""
    matches = list(TOOL_CALL_RE.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(1)
    # Find balanced JSON
    depth = 0
    end = -1
    for i, c in enumerate(raw):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end <= 0:
        return None
    try:
        obj = json.loads(raw[:end])
        name = obj.get("name")
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        return (name, args)
    except Exception:
        return None


def extract_tool_schemas(env):
    """Extract OpenAI-style tool schemas from env methods (matches TRL behavior)."""
    import inspect
    tool_methods = [
        "fhir_patient_search", "fhir_observation_search", "fhir_vitals_search",
        "fhir_condition_search", "fhir_procedure_search", "fhir_medication_request_search",
        "fhir_service_request_create", "fhir_medication_request_create", "fhir_vitals_create",
        "calculator", "finish",
    ]
    schemas = []
    for name in tool_methods:
        method = getattr(env, name, None)
        if method is None:
            continue
        sig = inspect.signature(method)
        doc = inspect.getdoc(method) or ""
        properties = {}
        required = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            properties[pname] = {"type": "string"}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": doc.split("\n")[0] if doc else name,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return schemas


def run_episode(model, tokenizer, env, task: dict, system_prompt: str,
                tool_schemas=None,
                max_turns: int = 8, max_new_tokens: int = 2000,
                temperature: float = 0.0, disable_thinking: bool = True):
    """Drive the model through a single episode using multi-turn tool calls.

    Returns dict with reward, steps, tool_calls, terminal_pass.
    """
    env.reset(task_id=task["id"])

    instruction = (
        task.get("instruction", "")
        + "\n\n"
        + (task.get("context", "") or "")
        + "\n\nProceed with the provided task."
    ).strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]

    kwargs = {}
    if disable_thinking:
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    tool_call_count = 0

    for turn in range(max_turns):
        # Build prompt (pass tool schemas so chat template inserts tool descriptions
        # matching what TRL did during training)
        tmpl_kwargs = dict(kwargs)
        if tool_schemas:
            tmpl_kwargs["tools"] = tool_schemas
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **tmpl_kwargs,
            )
        except TypeError:
            # chat_template_kwargs not supported in this tokenizer version
            tmpl_kwargs.pop("chat_template_kwargs", None)
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **tmpl_kwargs,
            )
        # Force no-think — only makes sense for Qwen3's full template (which
        # has thinking support). A SFT model trained with a simple template
        # sees `<think>` tags as garbage and generates nothing.
        has_thinking_template = "enable_thinking" in (tokenizer.chat_template or "")
        if disable_thinking and has_thinking_template and not prompt_text.rstrip().endswith("</think>"):
            prompt_text = prompt_text + "<think>\n\n</think>\n\n"
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            out = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-5),
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = out[0, prompt_ids.shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

        messages.append({"role": "assistant", "content": generated})

        tc = parse_tool_call(generated)
        if tc is None:
            # No recognizable tool call — model stopped. Treat as episode end.
            break

        tool_call_count += 1
        name, args = tc

        if name == "finish":
            value = args.get("value", [])
            try:
                env.finish(value)
            except Exception:
                pass
            break

        method = getattr(env, name, None)
        if method is None:
            response = f"Unknown tool: {name}"
        else:
            try:
                response = method(**args)
            except Exception as e:
                response = f"Tool error: {e}"

        # Feed the env response back as the next user turn
        messages.append({"role": "user", "content": f"<tool_response>\n{response}\n</tool_response>"})

        if getattr(env, "done", False):
            break

    # Force-evaluate partial episodes
    if not getattr(env, "done", False):
        try:
            env.reward = env._evaluate()
        except Exception:
            pass

    return {
        "task_id": task["id"],
        "task_type": task.get("id", "").rsplit("_", 1)[0],
        "workflow_branch": task.get("workflow_branch", ""),
        "reward": float(getattr(env, "reward", 0.0)),
        "tool_calls": tool_call_count,
        "step_count": int(getattr(env, "_step_count", 0)),
        "terminal_pass": float(getattr(env, "reward", 0.0)) >= 1.0,
        "messages": messages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base HF model name or path")
    ap.add_argument("--pre-merge-adapter", action="append", default=[],
                    help="LoRA adapter(s) to merge into the base model before --adapter. "
                         "Repeat for multiple. Applied in order given.")
    ap.add_argument("--adapter", action="append", default=[],
                    help="LoRA adapter(s) applied after pre-merges. Repeat for multiple.")
    ap.add_argument("--tasks", default=str(ROOT / "medagentbench_env/data/test_tasks.json"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-rollouts-per-task", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temp; 0 = greedy. Default 0.7 for diversity.")
    ap.add_argument("--disable-qwen-thinking", action="store_true", default=True)
    ap.add_argument("--enable-qwen-thinking", action="store_true", default=False,
                    help="Override --disable-qwen-thinking; allow Qwen3 to emit <think>...</think> blocks")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    # Load tokenizer from base model to avoid adapter-dir tokenizer_config bugs
    # (e.g. SFT adapters saving extra_special_tokens as a list instead of dict).
    # Chat template is copied from the adapter dir if it contains one.
    tok_src = args.model
    print(f"Loading tokenizer from: {tok_src}")
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    # Copy chat_template from adapter if present (contains {% generation %} markers)
    _adapter_dir = args.adapter[0] if args.adapter else (args.pre_merge_adapter[0] if args.pre_merge_adapter else None)
    if _adapter_dir:
        _ct_path = Path(_adapter_dir) / "chat_template.jinja"
        if _ct_path.exists():
            tokenizer.chat_template = _ct_path.read_text()
            print(f"  chat_template loaded from {_ct_path}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = True

    # Apply pre-merge adapters (merged into base permanently)
    for ad in args.pre_merge_adapter:
        from peft import PeftModel
        print(f"  merging pre-merge adapter: {ad}")
        model = PeftModel.from_pretrained(model, ad)
        model = model.merge_and_unload()

    # Apply regular adapters (kept as PEFT wrapper for inference)
    if args.adapter:
        from peft import PeftModel
        for i, ad in enumerate(args.adapter):
            print(f"  loading adapter: {ad}")
            if i == 0:
                model = PeftModel.from_pretrained(model, ad)
            else:
                model.load_adapter(ad, adapter_name=f"ad_{i}")
        model = model.merge_and_unload()

    model.eval()

    # Bootstrap env — point it at the eval tasks file so _resolve_task_from_reset_kwargs
    # can find every task by id (otherwise benchmark_fair's task1/task3 etc.
    # would miss the default _RL_TASK_TYPES filter).
    import sys
    sys.path.insert(0, str(ROOT))
    import importlib
    env_mod = importlib.import_module("medagentbench_env.fhir_env")
    env_mod._TASKS_FILE = Path(args.tasks).resolve()
    env_mod._SELECTED_TASK_TYPES = {
        t["id"].rsplit("_", 1)[0] if t["id"].rsplit("_", 1)[-1].isdigit() else t["id"]
        for t in json.loads(Path(args.tasks).read_text())
    }
    print(f"Env task registry: {args.tasks}  types={sorted(env_mod._SELECTED_TASK_TYPES)}")
    env = env_mod.MedAgentTrainEnv()
    tool_schemas = extract_tool_schemas(env)
    print(f"Extracted {len(tool_schemas)} tool schemas")

    system_prompt = (ROOT / "medagentbench_env/data/new_system.txt").read_text().strip()

    tasks = json.loads(Path(args.tasks).read_text())
    print(f"Loaded {len(tasks)} tasks from {args.tasks}")
    print(f"Running {args.num_rollouts_per_task} rollouts per task × {len(tasks)} tasks = "
          f"{args.num_rollouts_per_task * len(tasks)} total rollouts")

    results = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        for roll in range(args.num_rollouts_per_task):
            r = run_episode(
                model, tokenizer, env, task, system_prompt,
                tool_schemas=tool_schemas,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                disable_thinking=args.disable_qwen_thinking and not args.enable_qwen_thinking,
            )
            r["rollout"] = roll
            results.append(r)
        done = (i + 1) * args.num_rollouts_per_task
        total = len(tasks) * args.num_rollouts_per_task
        elapsed = time.time() - t0
        eta = elapsed / done * (total - done)
        print(f"  [{done}/{total}] task={task['id']} eta={eta/60:.1f}min "
              f"recent_rewards={[round(x['reward'],2) for x in results[-args.num_rollouts_per_task:]]}")

    # Write results
    (out_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results))

    # Aggregate stats
    by_type = defaultdict(lambda: {"n": 0, "r_sum": 0.0, "pass": 0, "tc": 0})
    for r in results:
        b = by_type[r["task_type"]]
        b["n"] += 1
        b["r_sum"] += r["reward"]
        b["pass"] += int(r["terminal_pass"])
        b["tc"] += r["tool_calls"]

    print(f"\n{'='*60}")
    print(f"Eval complete: {len(results)} rollouts")
    print(f"{'='*60}")
    print(f"{'task_type':<12} {'n':>3} {'mean_r':>7} {'pass%':>6} {'tc/ep':>6}")
    for k in sorted(by_type):
        b = by_type[k]
        n = b["n"]
        print(f"{k:<12} {n:>3} {b['r_sum']/n:>7.3f} {b['pass']*100/n:>5.1f}% {b['tc']/n:>6.2f}")
    overall_r = sum(r["reward"] for r in results) / len(results)
    overall_pass = sum(1 for r in results if r["terminal_pass"]) / len(results)
    print(f"\nOVERALL: mean_r={overall_r:.3f}  pass={overall_pass*100:.1f}%  "
          f"n={len(results)}")

    summary = {
        "model": args.model,
        "pre_merge_adapters": args.pre_merge_adapter,
        "adapters": args.adapter,
        "n_rollouts": len(results),
        "overall_mean_reward": overall_r,
        "overall_pass_rate": overall_pass,
        "by_task_type": {
            k: {"n": b["n"], "mean_r": b["r_sum"]/b["n"],
                "pass_rate": b["pass"]/b["n"], "tc_per_ep": b["tc"]/b["n"]}
            for k, b in by_type.items()
        },
        "config": {
            "num_rollouts_per_task": args.num_rollouts_per_task,
            "temperature": args.temperature,
            "max_turns": args.max_turns,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir}/results.jsonl and {out_dir}/summary.json")


if __name__ == "__main__":
    main()
