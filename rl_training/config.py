"""
Configuration for RL training pipeline.
All hyperparameters, paths, and constants in one place.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Resolved base paths ─────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # rl_training/
_HEALTHCARE = _HERE.parent                       # Healthcare/
_SRC = _HEALTHCARE / "medagentbench_v2" / "src"  # medagentbench_v2/src/


@dataclass
class PathConfig:
    """All filesystem paths used by the pipeline."""
    healthcare_root: str = str(_HEALTHCARE)
    medagentbench_src: str = str(_SRC)
    tasks_path: str = str(
        _SRC / "MedAgentBench" / "data" / "medagentbench" / "new_patient_tasks.json"
    )
    # test_data_v2 contains the Mg/K+/A1c tasks (different semantics — loaded with
    # "v2_" prefix to avoid ID collision with new_patient_tasks.json)
    tasks_path_v2: str = str(
        _SRC / "MedAgentBench" / "data" / "medagentbench" / "test_data_v2.json"
    )
    system_prompt_path: str = str(_SRC / "prompts" / "system.txt")
    output_dir: str = str(_HEALTHCARE / "rl_output")


@dataclass
class FHIRConfig:
    """FHIR server connection settings."""
    api_base: str = "http://localhost:8080/fhir/"
    cutoff_ts: str = "2025-01-01T00:00:00+00:00"


@dataclass
class ModelConfig:
    """Policy model and LoRA settings."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    # For remote API-based inference (e.g. vLLM, OpenRouter)
    api_url: Optional[str] = None          # e.g. "http://localhost:8001/v1"
    api_key: Optional[str] = None          # e.g. "EMPTY" for local vLLM
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7       # >0 for exploration during training
    temperature_eval: float = 0.0  # greedy during evaluation
    top_p: float = 0.9


@dataclass
class RewardConfig:
    """
    Reward shaping weights — shared across all RL task types.

    Universal components
    ────────────────────
      decision        : correct act/no-act decision (gated on ≥1 GET)
      terminal_pass   : official evaluator passes
      spurious_action : POST when ground-truth says no action needed
      invalid_fhir    : malformed or rejected FHIR call
      redundant_lookup: repeated identical GET call (per occurrence)
      redundant_lookup_cap: max total redundancy penalty per episode

    Per-task required-action step rewards (all tasks)
    ──────────────────────────────────────────────────
      action_a        : primary required action placed correctly
      action_b        : secondary required action (coupled tasks only)
      coupled_missing : one of a required pair placed without the other
    """
    # Universal positive rewards
    decision: float = 0.30      # correct act/no-act decision (gated on ≥1 GET)
    terminal_pass: float = 1.00

    # Universal penalties
    spurious_action: float = -0.30
    invalid_fhir: float = -0.10
    redundant_lookup: float = -0.05
    redundant_lookup_cap: float = -0.20   # total cap per episode

    # Per-task step rewards (generic — apply to all task types)
    action_a: float = 0.25      # primary required action
    action_b: float = 0.25      # secondary required action (coupled tasks)
    coupled_missing: float = -0.20

    # ── Backward-compat aliases for task-7 callers ───────────────────
    @property
    def ecg_order(self) -> float:
        return self.action_b

    @property
    def drug_stop(self) -> float:
        return self.action_a

    # Task-7 clinical constants
    qt_threshold: int = 500
    ecg_snomed_code: str = "445118002"
    qt_prolonging_meds: tuple = (
        "ondansetron", "prochlorperazine", "haloperidol",
        "quetiapine", "olanzapine", "risperidone",
        "ziprasidone", "clozapine", "chlorpromazine",
    )
    non_active_statuses: tuple = (
        "stopped", "cancelled", "completed", "entered-in-error",
    )

    # ── Legacy alias ─────────────────────────────────────────────────────────
    @property
    def threshold_eval(self) -> float:
        return self.decision


@dataclass
class TrainConfig:
    """GRPO training hyperparameters."""
    # Episodes
    num_epochs: int = 50
    episodes_per_epoch: int = 8
    group_size: int = 64          # K rollouts per task for GRPO
    max_steps_per_episode: int = 12

    # Optimiser
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05

    # GRPO specific
    kl_coef: float = 0.05       # KL divergence penalty coefficient
    clip_range: float = 0.2     # PPO-style clip
    advantage_eps: float = 1e-8 # std normalisation epsilon

    # Data — RL-worthy task types (conditional decision required)
    # "v2_task5/9/10" are from test_data_v2.json (Mg/K+/A1c — loaded with v2_ prefix)
    rl_task_types: tuple = (
        "task1", "task2", "task4", "task5", "task6",
        "task7", "task8", "task9", "task10",
        "v2_task5", "v2_task9", "v2_task10",
    )
    train_ratio: float = 0.8     # fraction of each task type used for training
    train_size: int = 20         # legacy: kept for task7-only mode compatibility
    seed: int = 42

    # Evaluation
    eval_every: int = 5          # evaluate every N epochs
    eval_episodes: int = 10      # episodes per evaluation

    # Checkpointing
    save_every: int = 10         # save model every N epochs
    log_wandb: bool = False
    wandb_project: str = "medagentbench-rl"


@dataclass
class RLConfig:
    """Top-level configuration aggregator."""
    paths: PathConfig = None       # type: ignore[assignment]
    fhir: FHIRConfig = None        # type: ignore[assignment]
    model: ModelConfig = None      # type: ignore[assignment]
    reward: RewardConfig = None    # type: ignore[assignment]
    train: TrainConfig = None      # type: ignore[assignment]

    def __post_init__(self):
        if self.paths is None:
            self.paths = PathConfig()
        if self.fhir is None:
            self.fhir = FHIRConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.reward is None:
            self.reward = RewardConfig()
        if self.train is None:
            self.train = TrainConfig()


# ── FHIR URL mapping (tool name → HTTP method, URL) ─────────────────────────

def get_fhir_url_map(api_base: str) -> dict[str, tuple[str, str]]:
    """Build the mapping from tool name to (HTTP method, FHIR endpoint URL)."""
    return {
        "patient_search":                ("GET",  f"{api_base}Patient"),
        "fhir_observation_search":       ("GET",  f"{api_base}Observation"),
        "fhir_vitals_search":            ("GET",  f"{api_base}Observation"),
        "fhir_medication_request_search":("GET",  f"{api_base}MedicationRequest"),
        "fhir_medication_request_create":("POST", f"{api_base}MedicationRequest"),
        "fhir_service_request_create":   ("POST", f"{api_base}ServiceRequest"),
        "fhir_condition_search":         ("GET",  f"{api_base}Condition"),
        "fhir_procedure_search":         ("GET",  f"{api_base}Procedure"),
        "fhir_vitals_create":            ("POST", f"{api_base}Observation"),
    }


# Tools that issue HTTP POSTs to the FHIR server
POST_TOOLS = {
    "fhir_medication_request_create",
    "fhir_service_request_create",
    "fhir_vitals_create",
}
