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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Lazy imports: only needed when actually training
try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    GRPOConfig = None
    GRPOTrainer = None

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
# Reward function
# ---------------------------------------------------------------------------

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
        rewards.extend([float(env.reward)] * num_generations)

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
        "--data-dir", type=str, default=str(_fhir_env._DATA_DIR),
        help="Path to directory containing new_patient_tasks.json",
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
        "--per-device-batch-size", type=int, default=4,
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
        "--num-generations", type=int, default=4,
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
        "--resume-from-checkpoint", type=str, default=None,
        help="Path to checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--beta", type=float, default=0.001,
        help="KL penalty coefficient (default: 0.001)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.5,
        help="Sampling temperature for generation (default: 1.5)",
    )
    args = parser.parse_args()

    # ── Configure fhir_env module state ────────────────────────────────────
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

    total_train_steps = max(1, len(dataset) * args.num_train_epochs)
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

    # ── Load model ──────────────────────────────────────────────────────────
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
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
        save_steps=10,
        save_total_limit=6,
        fp16=True,
        bf16=False,
        num_generations=int(args.num_generations),
        beta=args.beta,
        temperature=args.temperature,
        max_tool_calling_iterations=_fhir_env._MAX_STEPS + 4,
    )
    if "qwen3" in args.model.lower() or args.disable_qwen_thinking:
        _grpo_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        print("chat_template_kwargs: enable_thinking=False", flush=True)

    grpo_config = GRPOConfig(**_grpo_kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        environment_factory=MedAgentTrainEnv,
        processing_class=tokenizer,
        args=grpo_config,
    )

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
