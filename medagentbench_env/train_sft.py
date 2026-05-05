"""LoRA SFT warmup on oracle + rejection-sampled demos, for Qwen3-1.7B.

Usage:
    python -m medagentbench_env.train_sft \
        --model Qwen/Qwen3-1.7B \
        --data training/sft_oracle/train.jsonl \
        --output-dir training/sft_ckpt \
        --num-epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_jsonl(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def clean_rejection_sample(assistant: str) -> str:
    """Strip leading garbage tokens before the first <tool_call> block."""
    idx = assistant.find("<tool_call>")
    if idx < 0:
        return assistant.strip()
    return assistant[idx:].strip()


def build_dataset(paths, clean_rejection: bool = True):
    all_rows = []
    for p in paths:
        rows = load_jsonl(Path(p))
        for r in rows:
            messages = r.get("messages", [])
            if not messages:
                continue
            # Clean assistant content if the sample came from rejection sampling
            if clean_rejection and "sft_oracle" not in str(p):
                if messages[-1]["role"] == "assistant":
                    messages[-1] = dict(messages[-1])
                    messages[-1]["content"] = clean_rejection_sample(messages[-1]["content"])
            all_rows.append({"messages": messages})
    return Dataset.from_list(all_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", nargs="+", required=True,
                    help="One or more SFT jsonl files. First should be oracle.")
    ap.add_argument("--output-dir", default="training/sft_ckpt")
    ap.add_argument("--num-epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--no-rejection-cleaning", action="store_true",
                    help="skip stripping leading garbage from rejection samples")
    ap.add_argument("--save-steps", type=int, default=25,
                    help="checkpoint every N steps (default: 25)")
    ap.add_argument("--resume-from-checkpoint", type=str, default=None)
    ap.add_argument("--pre-merge-adapter", type=str, default=None,
                    help="Path to a LoRA adapter to merge into the base before training (sequential SFT).")
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Patch chat template with {% generation %} markers so TRL's assistant-only
    # loss masking works.  Simplified multi-turn template for Qwen3 (no
    # thinking, no tool schemas — we pass tools via system prompt text).
    tok.chat_template = (
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "{{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}"
        "{%- elif message.role == 'user' %}"
        "{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}"
        "{%- elif message.role == 'assistant' %}"
        "{{- '<|im_start|>assistant\\n' }}"
        "{%- generation %}"
        "{{- message.content + '<|im_end|>' }}"
        "{%- endgeneration %}"
        "{{- '\\n' }}"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\\n' }}"
        "{%- endif %}"
    )

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
    )

    # Optional: merge a pre-existing LoRA adapter into the base before SFT.
    # Useful for sequential fine-tuning (e.g., RFT on top of an SFT checkpoint).
    if getattr(args, "pre_merge_adapter", None):
        from peft import PeftModel
        print(f"Pre-merging adapter: {args.pre_merge_adapter}")
        model = PeftModel.from_pretrained(model, args.pre_merge_adapter)
        model = model.merge_and_unload()
        print("Pre-merge complete; new LoRA will train on top of merged base.")

    dataset = build_dataset(args.data, clean_rejection=not args.no_rejection_cleaning)
    print(f"Loaded {len(dataset)} SFT samples")
    # Show one sample length
    sample_tokens = tok.apply_chat_template(dataset[0]["messages"], tokenize=True)
    print(f"Sample 0 token length: {len(sample_tokens)}")

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=2,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=4,
        max_length=args.max_seq_length,
        report_to="none",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        dataloader_num_workers=0,
        assistant_only_loss=True,  # now works because chat template has {% generation %} markers
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tok,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    print(f"SFT saved to {args.output_dir}")


if __name__ == "__main__":
    main()
