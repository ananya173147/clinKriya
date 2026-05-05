#!/usr/bin/env python3
"""
MedAgentBench RL Training Script.

Uses TRL's GRPOTrainer with named FHIR tool calls matching the benchmark
evaluation format so the model trains and evaluates on the same interface.

Usage:
    python train.py

    # Or on Northflank with OUTPUT_DIR set:
    python train.py --output-dir /output
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

# Lazy imports: only needed when actually training
try:
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback
except ImportError:
    GRPOConfig = None
    GRPOTrainer = None
    TrainerCallback = object

# Environment, dataset builder, and all shared state live in fhir_env.
import medagentbench_env.fhir_env as _fhir_env
from medagentbench_env.fhir_env import (
    MedAgentTrainEnv,
    build_dataset,
    _RL_TASK_TYPES,
)

# Post-training export helpers.
from medagentbench_env.export import export_reward_graph, export_completions_debug


# ---------------------------------------------------------------------------
# Metrics logger — writes every step to extra_metrics.jsonl for live monitor
# ---------------------------------------------------------------------------

class MetricsLogger(TrainerCallback):
    """Writes every step's metrics to {output_dir}_checkpoints/extra_metrics.jsonl.

    Compatible with the shared VM training monitor at port 7861.
    """
    def __init__(self, output_dir: str, total_steps: int = 0):
        ckpt_dir = f"{output_dir}_checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        self.path = os.path.join(ckpt_dir, "extra_metrics.jsonl")
        self.total_steps = total_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        global _CURRENT_STEP
        _CURRENT_STEP = state.global_step
        record = {"step": state.global_step, "total_steps": self.total_steps, **logs}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Expected tool calls per task type
# (GET lookups + POST actions + finish call; range covers no-action vs action branch)
# ---------------------------------------------------------------------------

EXPECTED_TOOL_CALLS: dict = {
    "task1":     (2, 3),   # 1 GET Procedure + finish [+ 1 POST ServiceRequest]
    "task2":     (2, 3),   # 1 GET MedicationRequest + finish [+ 1 POST]
    "task4":     (2, 3),   # 1 GET Procedure + finish [+ 1 POST]
    "task5":     (3, 5),   # 2 GETs (Condition+Procedure) + finish [+ 2 POSTs]
    "task6":     (2, 4),   # 1 GET Observation + finish [+ 2 POSTs: med+lab]
    "task7":     (3, 5),   # 2 GETs (Obs+Med) + finish [+ 2 POSTs: stop+ECG]
    "task8":     (2, 3),   # 1 GET MedicationRequest + finish [+ 1 POST Naloxone]
    "task9":     (2, 3),   # 1 GET Procedure + finish [+ 1 POST flu vax order]
    "task10":    (2, 3),   # 1 GET Procedure + finish [+ 1 POST covid vax order]
    "v2_task5":  (2, 3),   # 1 GET Observation (Mg) + finish [+ 1 POST MedRequest]
    "v2_task9":  (2, 4),   # 1 GET Observation (K) + finish [+ 2 POSTs: med+lab]
    "v2_task10": (2, 3),   # 1 GET Observation (A1c) + finish [+ 1 POST ServiceRequest]
}


# ---------------------------------------------------------------------------
# Per-episode task stats logger
# ---------------------------------------------------------------------------

_TASK_STATS_PATH: str = ""  # set by main() after output_dir is known


def _log_task_stats(env, step: int) -> None:
    """Append one record per episode to task_stats.jsonl."""
    if not _TASK_STATS_PATH:
        return
    task = getattr(env, "_task", None) or {}
    task_id = task.get("id", "")
    parts = task_id.rsplit("_", 1)
    task_type = parts[0] if len(parts) == 2 and parts[1].isdigit() else task_id
    exp = EXPECTED_TOOL_CALLS.get(task_type, (1, 3))
    reward = float(getattr(env, "reward", 0.0))
    # terminal_pass = grader passed (reward includes the +1.0 terminal weight).
    terminal_pass = reward >= 0.95
    agent_answer = getattr(env, "_agent_answer", None)
    history = getattr(env, "_history", []) or []
    # Full structured trace: every history item with role + first 600 chars of content.
    # Skip the system prompt (history[0]) — too long and identical across rollouts.
    trace_items = []
    for it in history[1:]:
        role = getattr(it, "role", "?")
        content = getattr(it, "content", "") or ""
        if len(content) > 600:
            content = content[:600] + " …[truncated]"
        trace_items.append({"role": role, "content": content})
    # Also keep a compact one-liner for at-a-glance dashboards.
    summary_items = []
    for it in history[-6:]:
        role = getattr(it, "role", "?")
        content = (getattr(it, "content", "") or "")[:100].replace("\n", " ")
        summary_items.append(f"[{role[:1]}] {content}")
    trace_summary = " | ".join(summary_items)
    record = {
        "step":          step,
        "task_id":       task_id,
        "task_type":     task_type,
        "tool_calls":    getattr(env, "_step_count", 0),
        "reward":        reward,
        "terminal_pass": terminal_pass,
        "agent_answer":  agent_answer,
        "trace_summary": trace_summary,
        "trace":         trace_items,
        "instruction":   (task.get("instruction", "") or "")[:300],
        "exp_min":       exp[0],
        "exp_max":       exp[1],
    }
    with open(_TASK_STATS_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

# Set by main() — small bonus added when model uses ≥1 tool, capped at 1.0.
# Counteracts the "skip tools and guess" collapse seen after epoch 1.
_TOOL_USE_BONUS: float = 0.0
_CURRENT_STEP: int = 0  # updated by MetricsLogger.on_log


def reward_func(prompts, completions, environments=None, **kwargs):
    """Return shaped rewards for a GRPO batch.

    GRPO calls this with len(completions) = num_prompts * num_generations.
    Each env executes once per prompt, so rewards are tiled across generations.
    """
    num_completions = len(completions)

    if environments is None:
        environments = kwargs.get("environments")

    if environments is not None:
        envs = environments
    else:
        # Unsloth fallback: registry has one env per prompt.
        n_prompts = len(MedAgentTrainEnv._registry)
        envs = MedAgentTrainEnv._registry[:n_prompts]
        del MedAgentTrainEnv._registry[:n_prompts]

    n_prompts = len(envs)
    if n_prompts == 0:
        return [0.0] * num_completions

    num_generations = num_completions // n_prompts
    rewards = []
    for env in envs:
        # Evaluate partial episodes (model stopped without calling finish or
        # hitting max_steps) so GET_CREDIT and action rewards are still visible.
        if not env.done:
            env.reward = env._evaluate()
            env._print_trace()
        r = float(env.reward)
        # Graduated tool-use bonus: +bonus for ≥1 tool call, +50% more for ≥2 calls.
        # Incentivizes multi-step reasoning (GET → POST → finish) without inflating
        # perfect episodes.
        if _TOOL_USE_BONUS > 0.0 and env._step_count > 0 and r < 1.0:
            bonus = _TOOL_USE_BONUS if env._step_count < 2 else min(1.5 * _TOOL_USE_BONUS, 0.25)
            r = min(1.0, r + bonus)
        rewards.extend([r] * num_generations)
        _log_task_stats(env, _CURRENT_STEP)

    if len(rewards) < num_completions:
        rewards.extend([0.0] * (num_completions - len(rewards)))
    return rewards[:num_completions]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train on MedAgentBench with GRPO")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-1.7B",
        help="Model name or path",
    )
    parser.add_argument(
        "--disable-qwen-thinking",
        action="store_true",
        help="Force chat_template_kwargs enable_thinking=False",
    )
    parser.add_argument(
        "--enable-qwen-thinking",
        action="store_true",
        help="Force chat_template_kwargs enable_thinking=True (overrides Qwen3 auto-disable)",
    )
    parser.add_argument(
        "--save-total-limit", type=int, default=2,
        help="Max checkpoints to keep (default: 2; use higher for learning-curve analysis)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(_fhir_env._DATA_DIR),
        help="Path to directory containing new_patient_tasks.json",
    )
    parser.add_argument(
        "--tasks-file", type=str, default=None,
        help="Override tasks JSON file (e.g. data/train_tasks.json for holdout split)",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=None,
        help="Number of tasks to use (default: all tasks from selected categories)",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=sorted(_RL_TASK_TYPES),
        help="Task categories to include, e.g. task1 task2 v2_task5",
    )
    parser.add_argument(
        "--max-completion-length", type=int, default=8000,
        help="Max tokens per generation.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("OUTPUT_DIR", "./output"),
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--num-train-epochs", type=int, default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per-device-batch-size", type=int, default=8,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-prompt-length", type=int, default=8192,
        help="Max prompt tokens passed to TRL before generation/loss truncation",
    )
    parser.add_argument(
        "--max-history-messages", type=int, default=64,
        help="Max in-episode history messages kept (includes initial system item)",
    )
    parser.add_argument(
        "--max-tool-response-chars", type=int, default=4000,
        help="Max chars kept from tool responses before truncation",
    )
    parser.add_argument(
        "--max-tool-response-entries", type=int, default=24,
        help="Max FHIR Bundle entries returned to the model per GET response",
    )
    parser.add_argument(
        "--max-steps", type=int, default=6,
        help="Max tool actions per episode before forced evaluation",
    )
    parser.add_argument(
        "--num-generations", type=int, default=8,
        help="GRPO num_generations (must divide per-device-batch-size)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce VRAM (recommended)",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push the final model to HuggingFace Hub after training",
    )
    parser.add_argument(
        "--hub-model-id", type=str, default=None,
        help="HuggingFace repo to push to, e.g. 'username/medagent-qwen3'",
    )
    parser.add_argument(
        "--hub-token", type=str,
        default=os.environ.get("HF_TOKEN") or _fhir_env._DEFAULT_HF_TOKEN,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--sft-adapter", type=str, default=None,
        help="Path to a LoRA SFT adapter to merge into the base model before GRPO."
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="LoRA rank for GRPO adapter (default: 16)",
    )
    parser.add_argument(
        "--resume-from-checkpoint", type=str, default=None,
        help="Path to checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--beta", type=float, default=0.05,
        help="KL penalty coefficient (default: 0.05)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.8,
        help="Sampling temperature for generation (default: 1.8)",
    )
    parser.add_argument(
        "--tool-use-bonus", type=float, default=0.1,
        help="Reward bonus for using ≥1 tool (prevents skip-tools collapse, default: 0.1)",
    )
    parser.add_argument(
        "--sft-mix-alpha", type=float, default=0.0,
        help="Initial weight of BC loss on SFT demos added to GRPO loss. 0 disables SFT-mix.",
    )
    parser.add_argument(
        "--sft-mix-alpha-min", type=float, default=0.1,
        help="Final BC weight after decay (default: 0.1).",
    )
    parser.add_argument(
        "--sft-mix-decay-steps", type=int, default=50,
        help="Optimizer steps over which alpha decays from alpha-init to alpha-min.",
    )
    parser.add_argument(
        "--sft-mix-data", type=str, default=None,
        help="Path to SFT demos JSONL (chat-format). Required if --sft-mix-alpha > 0.",
    )
    parser.add_argument(
        "--sft-mix-bsz", type=int, default=1,
        help="BC batch size per optimizer step.",
    )
    parser.add_argument(
        "--use-vllm", action="store_true",
        help="Offload generation to vLLM (large speedup for ≥14B models).",
    )
    parser.add_argument(
        "--vllm-mode", type=str, default="colocate", choices=["colocate", "server"],
        help="colocate=same process+GPU; server=external trl vllm-serve process.",
    )
    parser.add_argument(
        "--vllm-gpu-mem-util", type=float, default=0.30,
        help="Fraction of GPU memory vLLM may use in colocate mode (default 0.30).",
    )
    parser.add_argument(
        "--vllm-server-port", type=int, default=8000,
        help="Port of vLLM server in server mode (default 8000).",
    )
    parser.add_argument(
        "--sft-mix-max-len", type=int, default=4096,
        help="Truncate SFT demos longer than this many tokens.",
    )
    args = parser.parse_args()

    global _TOOL_USE_BONUS, _TASK_STATS_PATH
    _TOOL_USE_BONUS = args.tool_use_bonus
    ckpt_dir = f"{args.output_dir}_checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    _TASK_STATS_PATH = os.path.join(ckpt_dir, "task_stats.jsonl")

    # ── Configure fhir_env module state ────────────────────────────────────
    if args.tasks_file is not None:
        _fhir_env._TASKS_FILE = Path(args.tasks_file)
    _fhir_env._SELECTED_TASK_TYPES = set(args.task_types)
    _fhir_env._MAX_PROMPT_LENGTH = max(512, int(args.max_prompt_length))
    _fhir_env._MAX_HISTORY_MESSAGES = max(8, int(args.max_history_messages))
    _fhir_env._MAX_TOOL_RESPONSE_CHARS = max(512, int(args.max_tool_response_chars))
    _fhir_env._MAX_TOOL_RESPONSE_ENTRIES = max(4, int(args.max_tool_response_entries))
    _fhir_env._MAX_STEPS = max(2, int(args.max_steps))
    print(
        f"Safeguards: max_prompt_length={_fhir_env._MAX_PROMPT_LENGTH}, "
        f"max_history_messages={_fhir_env._MAX_HISTORY_MESSAGES}, "
        f"max_tool_response_chars={_fhir_env._MAX_TOOL_RESPONSE_CHARS}, "
        f"max_tool_response_entries={_fhir_env._MAX_TOOL_RESPONSE_ENTRIES}, "
        f"max_steps={_fhir_env._MAX_STEPS}"
    )

    # Reset task caches so _get_tasks() picks up the selected task types.
    _fhir_env._TASKS = []
    _fhir_env._TASKS_BY_ID = {}
    _fhir_env._TASKS_BY_INSTRUCTION = {}
    _fhir_env._TASK_INDEX = 0

    # Pre-load shared resources.
    _fhir_env._get_mock_fhir()
    print(f"Loaded FHIR cache from {_fhir_env._CACHE_PATH}")

    dataset = build_dataset(Path(args.data_dir), args.num_tasks)
    print(f"Training dataset: {len(dataset)} tasks")
    if len(dataset) == 0:
        raise RuntimeError(
            "No tasks selected. Check --task-types and --num-tasks settings."
        )

    effective_batch_size = max(1, min(args.per_device_batch_size, len(dataset)))
    effective_grad_accum = max(
        1, min(args.gradient_accumulation_steps, len(dataset))
    )
    if effective_batch_size != args.per_device_batch_size:
        print(
            f"Adjusted per-device batch size from {args.per_device_batch_size} "
            f"to {effective_batch_size} for small dataset."
        )
    if effective_grad_accum != args.gradient_accumulation_steps:
        print(
            f"Adjusted gradient accumulation from {args.gradient_accumulation_steps} "
            f"to {effective_grad_accum} for small dataset."
        )

    # TRL GRPO: each optimizer step consumes (batch_size // num_generations) * grad_accum
    # unique prompts. One "data epoch" = ceil(dataset / prompts_per_step) optimizer steps.
    # Previous formula (dataset * epochs) ignored batch size, giving ~8x too many steps.
    num_gen = max(1, int(args.num_generations))
    prompts_per_step = max(1, (effective_batch_size // num_gen) * effective_grad_accum)
    steps_per_epoch = math.ceil(len(dataset) / prompts_per_step)
    total_train_steps = max(1, steps_per_epoch * args.num_train_epochs)
    print(
        f"Training: {len(dataset)} tasks, {prompts_per_step} prompts/step, "
        f"{steps_per_epoch} steps/epoch × {args.num_train_epochs} epochs "
        f"= {total_train_steps} total steps"
    )

    # ── Load model ──────────────────────────────────────────────────────────
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False

    # Optionally merge SFT LoRA adapter into base model before adding fresh GRPO LoRA
    if args.sft_adapter:
        from peft import PeftModel
        print(f"Loading SFT adapter from {args.sft_adapter} and merging into base model...")
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()
        print("SFT adapter merged.")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    if args.per_device_batch_size % int(args.num_generations) != 0:
        raise ValueError(
            f"--per-device-batch-size ({args.per_device_batch_size}) must be divisible by "
            f"--num-generations ({args.num_generations}) for GRPO."
        )

    # ── GRPO config ─────────────────────────────────────────────────────────
    _grpo_kwargs: Dict[str, Any] = dict(
        output_dir=args.output_dir,
        max_steps=total_train_steps,
        num_train_epochs=args.num_train_epochs,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=effective_batch_size,
        gradient_accumulation_steps=effective_grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=10,
        log_completions=True,
        num_completions_to_print=2,
        logging_steps=1,
        save_steps=20,
        save_total_limit=args.save_total_limit,
        fp16=False,
        bf16=True,
        num_generations=int(args.num_generations),
        beta=args.beta,
        temperature=args.temperature,
        max_tool_calling_iterations=_fhir_env._MAX_STEPS + 1,
    )
    _thinking_off = ("qwen3" in args.model.lower() or args.disable_qwen_thinking) and not args.enable_qwen_thinking
    if _thinking_off:
        _grpo_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        print("chat_template_kwargs: enable_thinking=False", flush=True)
    elif args.enable_qwen_thinking:
        _grpo_kwargs["chat_template_kwargs"] = {"enable_thinking": True}
        print("chat_template_kwargs: enable_thinking=True", flush=True)

    # vLLM integration: offload generation to vLLM (separate or co-located).
    # Big win for ≥14B models since policy forward in HF takes minutes per
    # step. Colocate mode shares GPU memory; tune --vllm-gpu-mem-util.
    if args.use_vllm:
        _grpo_kwargs["use_vllm"] = True
        _grpo_kwargs["vllm_mode"] = args.vllm_mode
        if args.vllm_mode == "colocate":
            _grpo_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_mem_util
            _grpo_kwargs["vllm_max_model_length"] = (
                args.max_prompt_length + args.max_completion_length + 256
            )
        else:  # server
            _grpo_kwargs["vllm_server_port"] = args.vllm_server_port
        print(
            f"vLLM enabled: mode={args.vllm_mode} "
            f"gpu_util={args.vllm_gpu_mem_util if args.vllm_mode=='colocate' else 'n/a'}"
        )

    grpo_config = GRPOConfig(**_grpo_kwargs)

    # ── Optional SFT-mix BC loss (anchors capability to demos) ───────────────
    sft_demos = None
    if args.sft_mix_alpha > 0.0:
        if not args.sft_mix_data:
            raise ValueError("--sft-mix-alpha > 0 requires --sft-mix-data <jsonl>")
        from medagentbench_env.sft_mix import (
            tokenize_sft_demos,
            SFTMixGRPOTrainer,
        )
        sft_demos = tokenize_sft_demos(
            args.model, Path(args.sft_mix_data), args.sft_mix_max_len,
        )
        if not sft_demos:
            raise RuntimeError(f"No usable SFT demos in {args.sft_mix_data}")
        trainer = SFTMixGRPOTrainer(
            model=model,
            reward_funcs=reward_func,
            train_dataset=dataset,
            environment_factory=MedAgentTrainEnv,
            processing_class=tokenizer,
            args=grpo_config,
            callbacks=[MetricsLogger(args.output_dir, total_steps=total_train_steps)],
            sft_demos=sft_demos,
            sft_mix_alpha_init=args.sft_mix_alpha,
            sft_mix_alpha_min=args.sft_mix_alpha_min,
            sft_mix_decay_steps=args.sft_mix_decay_steps,
            sft_mix_bsz=args.sft_mix_bsz,
            pad_token_id=tokenizer.pad_token_id or 151643,
        )
        print(
            f"SFT-mix enabled: alpha {args.sft_mix_alpha} → {args.sft_mix_alpha_min} "
            f"over {args.sft_mix_decay_steps} steps, bsz={args.sft_mix_bsz}, "
            f"{len(sft_demos)} demos"
        )
    else:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_func,
            train_dataset=dataset,
            environment_factory=MedAgentTrainEnv,
            processing_class=tokenizer,
            args=grpo_config,
            callbacks=[MetricsLogger(args.output_dir, total_steps=total_train_steps)],
        )

    # Override logit-computation chunk size to 2 to avoid fp32 logit tensor OOM.
    # With 14B model: 4 items × 4000 tokens × 151936 vocab × fp32 = 9.7 GB → OOM.
    # Chunking to 2 items: 4.86 GB → fits. Per_device_batch_size still controls
    # generation batch and RepeatSampler batch (must be >= num_generations).
    trainer._logit_chunk_size = 2
    # Truncate accumulated tool-call context to max_prompt_length tokens before each re-prefill.
    # TRL's _tool_call_loop re-prefills the full growing history on every turn (no KV reuse).
    # Without this, KV cache grows unboundedly: 8 seq × 20K tokens × 0.16 MB/tok = 26 GB → OOM.
    # With this: KV cache bounded at 8 seq × max_prompt_length × 0.16 MB = 5.4 GB.
    trainer._generation_ctx_limit = args.max_prompt_length

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    export_reward_graph(args.output_dir, trainer.state.log_history)
    export_completions_debug(args.output_dir)
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")

    if args.push_to_hub:
        if not args.hub_model_id:
            model_basename = args.model.split("/")[-1]
            args.hub_model_id = f"medagent-{model_basename}"
            print(f"No --hub-model-id given, using: {args.hub_model_id}")
        print(f"Pushing model to HuggingFace Hub: {args.hub_model_id} ...")
        trainer.push_to_hub(
            repo_id=args.hub_model_id,
            token=args.hub_token,
            private=False,
        )
        print(f"Model pushed to https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
