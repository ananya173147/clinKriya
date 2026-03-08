#!/usr/bin/env python3
"""
MedAgentBench RL Training Script.

Uses TRL's GRPOTrainer with OpenEnv's environment_factory pattern to train
a model on clinical decision-making tasks via FHIR API interaction.

Usage:
    # Start env server first:
    cd medagentbench_env && uvicorn server.app:app --port 8001

    # Run training (single GPU, vLLM colocate):
    python train.py --env-url http://localhost:8001

    # Or on Northflank with ENV_URL set:
    python train.py
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset

from medagentbench_env.client import MedAgentBenchEnv
from medagentbench_env.models import ActionType, MedAgentBenchAction

from trl import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
_DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "medagentbenchv2"
    / "medagentbench_v2"
    / "src"
    / "MedAgentBench"
    / "data"
    / "medagentbench"
)


# ---------------------------------------------------------------------------
# Training environment wrapper (exposed as tools to the model)
# ---------------------------------------------------------------------------

# Module-level env URL, set from CLI args before trainer starts
_ENV_URL = "http://localhost:8001"


class MedAgentTrainEnv:
    """Training wrapper that exposes FHIR operations as callable tools.

    GRPOTrainer's environment_factory creates one instance per rollout.
    The model calls get_fhir/post_fhir/finish as tools during generation.
    """

    def __init__(self):
        self.client = MedAgentBenchEnv(base_url=_ENV_URL)
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        """Start a new clinical task episode.

        Returns the task instruction and available FHIR API functions.
        """
        result = self.client.reset()
        obs = result.observation
        self.reward = 0.0
        self.done = False
        return obs.response_text

    def get_fhir(self, url: str) -> str:
        """Query the FHIR EHR server with a GET request.

        Args:
            url: Full FHIR API URL with query parameters,
                 e.g. 'http://localhost:8080/fhir/Patient?name=Peter&birthdate=1932-12-29'

        Returns:
            JSON response from the FHIR server, or an error message.
        """
        if self.done:
            return "Episode already finished."

        action = MedAgentBenchAction(
            action_type=ActionType.GET,
            url=url,
            raw_response=f"GET {url}",
        )
        result = self.client.step(action)
        obs = result.observation
        self.reward = float(obs.reward or 0.0)
        self.done = obs.done
        return obs.response_text

    def post_fhir(self, url: str, payload: str) -> str:
        """Create or update a FHIR resource with a POST request.

        Args:
            url: FHIR API endpoint URL,
                 e.g. 'http://localhost:8080/fhir/ServiceRequest'
            payload: JSON string containing the FHIR resource to create,
                     e.g. '{"resourceType": "ServiceRequest", "status": "active", ...}'

        Returns:
            Confirmation message or error.
        """
        if self.done:
            return "Episode already finished."

        try:
            body = json.loads(payload)
        except json.JSONDecodeError:
            return "Invalid JSON payload. Please provide valid JSON."

        action = MedAgentBenchAction(
            action_type=ActionType.POST,
            url=url,
            body=body,
            raw_response=f"POST {url}\n{payload}",
        )
        result = self.client.step(action)
        obs = result.observation
        self.reward = float(obs.reward or 0.0)
        self.done = obs.done
        return obs.response_text

    def finish(self, answers: str) -> str:
        """Signal that all tasks are complete and provide final answers.

        Args:
            answers: JSON-formatted list of answers,
                     e.g. '["S6534835"]' or '[]' if no answer needed.

        Returns:
            Completion confirmation.
        """
        if self.done:
            return "Episode already finished."

        try:
            answer_list = json.loads(answers)
            if not isinstance(answer_list, list):
                answer_list = [answer_list]
        except json.JSONDecodeError:
            answer_list = [answers]

        action = MedAgentBenchAction(
            action_type=ActionType.FINISH,
            answer=answer_list,
            raw_response=f"FINISH({answers})",
        )
        result = self.client.step(action)
        obs = result.observation
        self.reward = float(obs.reward or 0.0)
        self.done = True
        return f"Task completed. Reward: {self.reward}"


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def reward_func(completions, environments, **kwargs):
    """Extract binary reward (1.0 = correct, 0.0 = incorrect) from environments."""
    return [env.reward for env in environments]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(data_dir: Path, num_tasks: int | None = None) -> Dataset:
    """Build training dataset from MedAgentBench stratified benchmark."""
    data_file = data_dir / "stratified_benchmark.json"
    with open(data_file) as f:
        tasks = json.load(f)

    if num_tasks is not None:
        tasks = tasks[:num_tasks]

    system_msg = (
        "You are a medical AI assistant that interacts with a FHIR EHR server "
        "to complete clinical tasks. You have access to tools for querying "
        "patient data (get_fhir), creating orders and resources (post_fhir), "
        "and signaling task completion (finish). "
        "Use these tools to fulfill the clinical instruction given to you. "
        "Always call finish when you are done."
    )

    prompts = []
    for task in tasks:
        context_str = f"\nContext: {task['context']}" if task.get("context") else ""
        user_msg = f"Clinical task: {task['instruction']}{context_str}"
        prompts.append([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])

    return Dataset.from_dict({"prompt": prompts})


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
        "--env-url", type=str,
        default=os.environ.get("ENV_URL", "http://localhost:8001"),
        help="URL of the OpenEnv environment server",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(_DATA_DIR),
        help="Path to MedAgentBench data directory",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=None,
        help="Number of tasks to use (default: all 120)",
    )
    parser.add_argument(
        "--max-completion-length", type=int, default=2048,
        help="Max tokens per generation",
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
    args = parser.parse_args()

    # Set module-level env URL for MedAgentTrainEnv instances
    global _ENV_URL
    _ENV_URL = args.env_url

    # Build dataset
    dataset = build_dataset(Path(args.data_dir), args.num_tasks)
    print(f"Training dataset: {len(dataset)} tasks")

    # Configure trainer
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
        num_completions_to_print=2,
        logging_steps=1,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        environment_factory=MedAgentTrainEnv,
        args=grpo_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
