"""
Policy model for RL training: Qwen2.5-7B-Instruct with LoRA.

Supports two modes:
  1. API mode:  Uses an OpenAI-compatible endpoint (vLLM, OpenRouter)
                for fast rollout collection.  No log probs / gradient.
  2. Local mode: Loads the model with HuggingFace transformers + LoRA
                 for training with log prob computation and gradient updates.

The PolicyModel provides:
  - generate(messages, tools) → action dict + raw text + token IDs + log probs
  - compute_log_probs(messages, response_text) → per-token log probs
  - get_trainable_parameters() → parameters for the optimiser
"""

from __future__ import annotations

import json
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import ModelConfig
from .tool_parser import (
    ParsedToolCall,
    parse_chat_completions_tool_calls,
    parse_raw_text_tool_calls,
)


class PolicyModel:
    """
    Wraps Qwen2.5-7B-Instruct for RL training.

    In local mode (default), loads the model with HuggingFace and optionally
    applies LoRA for parameter-efficient fine-tuning.

    In API mode, delegates generation to an OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        config: ModelConfig,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self._api_mode = config.api_url is not None

        if self._api_mode:
            self._init_api_mode()
        else:
            self._init_local_mode()

    # ── Initialisation ───────────────────────────────────────────────────

    def _init_api_mode(self):
        """Initialise for API-based generation (vLLM / OpenRouter)."""
        from openai import OpenAI

        self.client = OpenAI(
            base_url=self.config.api_url,
            api_key=self.config.api_key or "EMPTY",
        )
        # Still load tokenizer for log prob computation if needed
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self.model = None

    def _init_local_mode(self):
        """Load the model locally with optional LoRA."""
        print(f"[Policy] Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )

        if self.config.use_lora:
            self._apply_lora()

        # Enable gradient checkpointing to reduce VRAM during backprop
        self.model.gradient_checkpointing_enable()
        print("[Policy] Gradient checkpointing enabled")

        self.model.eval()
        print(f"[Policy] Model loaded. LoRA={self.config.use_lora}")

    def _apply_lora(self):
        """Apply LoRA adapters to the model."""
        from peft import LoraConfig, get_peft_model, TaskType

        target_modules = [
            m.strip() for m in self.config.lora_target_modules.split(",")
        ]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(
            f"[Policy] LoRA applied: {trainable:,} trainable / "
            f"{total:,} total params ({100*trainable/total:.2f}%)"
        )

    # ── Generation ──────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        return_log_probs: bool = True,
    ) -> dict:
        """
        Generate a response given the conversation context.

        Args:
            messages: OpenAI-format message history
            tools: Tool schemas in Chat Completions format
            temperature: Override temperature (None = use config)
            return_log_probs: Whether to compute per-token log probs

        Returns:
            {
                "action": {"content": str|None, "tool_calls": [ParsedToolCall]},
                "generated_text": str,
                "token_ids": list[int],
                "log_probs": list[float],
            }
        """
        if self._api_mode:
            return self._generate_api(messages, tools, temperature)
        else:
            return self._generate_local(messages, tools, temperature, return_log_probs)

    def _generate_api(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float | None,
    ) -> dict:
        """Generate using an OpenAI-compatible API."""
        temp = temperature if temperature is not None else self.config.temperature

        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": self.config.max_new_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        # Parse tool calls
        tool_calls = parse_chat_completions_tool_calls(msg)

        return {
            "action": {
                "content": msg.content,
                "tool_calls": tool_calls,
            },
            "generated_text": msg.content or "",
            "token_ids": [],   # not available in API mode
            "log_probs": [],   # not available in API mode
        }

    def _generate_local(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float | None,
        return_log_probs: bool = True,
    ) -> dict:
        """Generate using the local HuggingFace model."""
        temp = temperature if temperature is not None else self.config.temperature

        # Format messages using the tokenizer's chat template
        # Qwen2.5-Instruct supports tool calling via its chat template
        try:
            if tools:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tools=self._convert_tools_for_template(tools),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            # Fallback: format without tools in template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Tokenize
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        if hasattr(self.model, "device"):
            model_device = next(self.model.parameters()).device
            input_ids = input_ids.to(model_device)

        prompt_length = input_ids.shape[1]

        # Get model's context window size
        model_max_len = getattr(
            self.model.config if hasattr(self.model, 'config') else None,
            'max_position_embeddings', 32768
        )

        # Truncate prompt from the LEFT if it exceeds context window
        max_prompt_len = model_max_len - self.config.max_new_tokens
        if prompt_length > max_prompt_len:
            print(
                f"[Policy] Truncating prompt from {prompt_length} to {max_prompt_len} tokens "
                f"(context={model_max_len}, reserved for gen={self.config.max_new_tokens})",
                flush=True,
            )
            input_ids = input_ids[:, -max_prompt_len:]
            prompt_length = input_ids.shape[1]

        effective_max_new = self.config.max_new_tokens

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=effective_max_new,
                temperature=max(temp, 0.01),  # avoid 0 for sampling
                do_sample=temp > 0,
                top_p=self.config.top_p if temp > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=return_log_probs,
            )

        # Extract generated tokens
        generated_ids = outputs.sequences[0, prompt_length:].tolist()
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        # Compute log probabilities
        log_probs = []
        if return_log_probs and outputs.scores:
            for i, score in enumerate(outputs.scores):
                if i < len(generated_ids):
                    probs = F.log_softmax(score[0], dim=-1)
                    token_log_prob = probs[generated_ids[i]].item()
                    log_probs.append(token_log_prob)

        # Parse tool calls from generated text
        tool_calls = parse_raw_text_tool_calls(generated_text)

        # Extract content (text before first tool call)
        content = generated_text
        if tool_calls:
            # Remove tool call markup from content
            import re
            content = re.sub(
                r"<\|?tool_call\|?>.*?<\|?/tool_call\|?>",
                "",
                content,
                flags=re.DOTALL,
            ).strip()
            if not content:
                content = None

        return {
            "action": {
                "content": content,
                "tool_calls": tool_calls,
            },
            "generated_text": generated_text,
            "token_ids": generated_ids,
            "log_probs": log_probs,
        }

    @staticmethod
    def _convert_tools_for_template(tools: list[dict]) -> list[dict]:
        """
        Convert Chat Completions tool format to the format expected
        by Qwen's chat template.

        Input (Chat Completions):
          {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

        Output (Qwen template):
          {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

        (They happen to be the same format for Qwen2.5)
        """
        return tools

    # ── Log probability computation ─────────────────────────────────────

    def compute_log_probs_for_response(
        self,
        messages: list[dict],
        response_text: str,
        tools: list[dict] | None = None,
    ) -> tuple[list[int], list[float]]:
        """
        Compute per-token log probabilities for a given response text,
        conditioned on the conversation context.

        This is used during GRPO training to compute policy gradients.

        Args:
            messages: The conversation context (up to assistant turn)
            response_text: The assistant's generated text
            tools: Tool schemas (for chat template formatting)

        Returns:
            (token_ids, log_probs) for the response tokens only
        """
        if self._api_mode:
            raise RuntimeError(
                "compute_log_probs_for_response requires local mode (no API URL)."
            )

        # Format the full conversation including the response
        try:
            if tools:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tools=self._convert_tools_for_template(tools),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        full_text = prompt_text + response_text

        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt")
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt")

        model_device = next(self.model.parameters()).device
        full_ids = full_ids.to(model_device)

        prompt_length = prompt_ids.shape[1]
        response_token_ids = full_ids[0, prompt_length:].tolist()

        # Forward pass
        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits[0]  # (seq_len, vocab_size)

        # Compute log probs for response tokens
        # logits[i] predicts token at position i+1
        log_probs = []
        for i in range(prompt_length - 1, full_ids.shape[1] - 1):
            token_id = full_ids[0, i + 1].item()
            probs = F.log_softmax(logits[i], dim=-1)
            log_probs.append(probs[token_id].item())

        # Only return log probs for response tokens
        response_log_probs = log_probs[prompt_length - 1:]

        return response_token_ids, response_log_probs

    def compute_log_probs_batch(
        self,
        messages_list: list[list[dict]],
        response_texts: list[str],
        tools: list[dict] | None = None,
    ) -> list[tuple[list[int], list[float]]]:
        """
        Batch version of compute_log_probs_for_response.
        Processes each item sequentially (batching is complex with variable lengths).
        """
        results = []
        for messages, response_text in zip(messages_list, response_texts):
            result = self.compute_log_probs_for_response(
                messages, response_text, tools
            )
            results.append(result)
        return results

    # ── Training interface ─────────────────────────────────────────────

    def compute_loss(
        self,
        messages: list[dict],
        response_text: str,
        advantage: float,
        old_log_probs: list[float],
        clip_range: float = 0.2,
        kl_coef: float = 0.05,
        tools: list[dict] | None = None,
    ) -> torch.Tensor:
        """
        Compute the GRPO/PPO clipped policy gradient loss for one trajectory.

        L = -Σ_t min(ratio_t * A, clip(ratio_t, 1-ε, 1+ε) * A) + β * KL

        Args:
            messages: Conversation context for each turn
            response_text: The complete generated text
            advantage: GRPO advantage for this trajectory
            old_log_probs: Log probs from the rollout policy
            clip_range: PPO clip range ε
            kl_coef: KL divergence penalty coefficient β
            tools: Tool schemas

        Returns:
            Scalar loss tensor (requires grad)
        """
        if self._api_mode:
            raise RuntimeError("compute_loss requires local mode.")

        # Format full sequence
        try:
            if tools:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tools=self._convert_tools_for_template(tools),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        full_text = prompt_text + response_text

        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt")
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt")

        model_device = next(self.model.parameters()).device

        prompt_length = prompt_ids.shape[1]
        full_length = full_ids.shape[1]
        response_length = full_length - prompt_length

        # Get model's context window size
        model_max_len = getattr(
            self.model.config if hasattr(self.model, 'config') else None,
            'max_position_embeddings', 32768
        )

        # Truncate from the LEFT if full sequence exceeds context window
        # Keep as much response as possible, truncate prompt from front
        if full_length > model_max_len:
            # Ensure we keep at least the response tokens + some prompt context
            keep_prompt = max(model_max_len - response_length, 256)
            keep_prompt = min(keep_prompt, prompt_length)  # don't exceed actual prompt
            truncated_start = prompt_length - keep_prompt
            full_ids = full_ids[:, truncated_start:]
            prompt_length = keep_prompt
            full_length = full_ids.shape[1]

            # If still too long (very long response), truncate response too
            if full_length > model_max_len:
                full_ids = full_ids[:, :model_max_len]
                full_length = model_max_len

        full_ids = full_ids.to(model_device)

        # Forward pass (WITH gradients)
        outputs = self.model(full_ids)
        logits = outputs.logits[0]

        # Compute new log probs for response tokens
        # The loop starts at prompt_length - 1, so all entries are response token log probs
        response_log_probs = []
        for i in range(prompt_length - 1, full_ids.shape[1] - 1):
            token_id = full_ids[0, i + 1]
            probs = F.log_softmax(logits[i], dim=-1)
            response_log_probs.append(probs[token_id])

        if not response_log_probs:
            print(f"[Policy] WARNING: No response log probs computed. "
                  f"prompt_length={prompt_length}, full_length={full_ids.shape[1]}")
            return torch.tensor(0.0, device=model_device, requires_grad=True)

        # Align lengths
        min_len = min(len(response_log_probs), len(old_log_probs))
        new_lps = torch.stack(response_log_probs[:min_len])
        old_lps = torch.tensor(
            old_log_probs[:min_len],
            device=model_device,
            dtype=self.dtype,
        )

        # PPO-style clipped objective
        ratio = torch.exp(new_lps - old_lps)
        adv = torch.tensor(advantage, device=model_device, dtype=self.dtype)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty (approximate: new_lp - old_lp)
        kl = (new_lps - old_lps).mean()
        kl_loss = kl_coef * kl

        total_loss = policy_loss + kl_loss
        return total_loss

    def get_trainable_parameters(self):
        """Return parameters that require gradients (LoRA params)."""
        if self.model is None:
            raise RuntimeError("No local model loaded.")
        return [p for p in self.model.parameters() if p.requires_grad]

    def train_mode(self):
        """Set model to training mode."""
        if self.model:
            self.model.train()

    def eval_mode(self):
        """Set model to evaluation mode."""
        if self.model:
            self.model.eval()

    def save_checkpoint(self, path: str):
        """Save LoRA adapter weights."""
        if self.model is None:
            raise RuntimeError("No local model to save.")
        if self.config.use_lora:
            self.model.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[Policy] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        """Load LoRA adapter weights."""
        if self.config.use_lora:
            from peft import PeftModel
            # Re-load base model and apply saved adapter
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base_model, path)
            print(f"[Policy] Loaded LoRA checkpoint from {path}")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True,
            )
            print(f"[Policy] Loaded full checkpoint from {path}")


class ReferencePolicyModel:
    """
    Frozen reference policy for KL divergence computation.

    Wraps a PolicyModel in eval mode with no gradients.
    This is the π_ref in the GRPO objective.
    """

    def __init__(self, config: ModelConfig, device: str = "auto"):
        self.policy = PolicyModel(config, device=device)
        self.policy.eval_mode()
        # Freeze all parameters
        if self.policy.model is not None:
            for param in self.policy.model.parameters():
                param.requires_grad = False

    def compute_log_probs(
        self,
        messages: list[dict],
        response_text: str,
        tools: list[dict] | None = None,
    ) -> list[float]:
        """Compute reference policy log probs (no gradient)."""
        _, log_probs = self.policy.compute_log_probs_for_response(
            messages, response_text, tools
        )
        return log_probs
