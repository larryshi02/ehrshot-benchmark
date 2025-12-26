"""
Constants and configuration for RLVR experiments.

This file centralizes all paths, task definitions, and default hyperparameters
to ensure consistency between Phase 1 and Phase 2 experiments.
"""

from typing import Dict, List
from dataclasses import dataclass, field

# =============================================================================
# TASK DEFINITIONS
# =============================================================================

TASKS: List[str] = [
    "acute_mi",
    "hyperlipidemia", 
    "hypertension",
    "pancreatic_cancer",
]

TASK_QUERIES: Dict[str, str] = {
    "acute_mi": "Will the patient develop an acute myocardial infarction in the next year?",
    "hyperlipidemia": "Will the patient develop hyperlipidemia in the next year?",
    "hypertension": "Will the patient develop hypertension in the next year?",
    "pancreatic_cancer": "Will the patient develop pancreatic cancer in the next year?",
}

# =============================================================================
# DATA PATHS
# =============================================================================

@dataclass
class DataPaths:
    """Centralized data path configuration."""
    
    # Base directory for all data
    base_dir: str = "/dev/shm/ehrshot-data"
    
    # SFT training data (contains conversations with GPT-5 traces)
    sft_data_dir: str = f"{base_dir}/data_gpt-5-mini_sft_reasoning_originalEHR"
    
    # Evaluation data (raw patient data with splits)
    eval_data_dir: str = f"{base_dir}/serialized_multi_task_data"
    
    # Training splits available
    train_splits: List[str] = field(default_factory=lambda: ["train_orig", "train_all"])
    
    def get_sft_train_path(self, task: str, split: str = "train_orig") -> str:
        """Get path to SFT training data for a task."""
        return f"{self.sft_data_dir}/{split}/{task}_sft_dataset.json"
    
    def get_sft_val_path(self, task: str) -> str:
        """Get path to SFT validation data for a task."""
        return f"{self.sft_data_dir}/val_small/{task}_sft_dataset.json"
    
    def get_eval_data_path(self, task: str) -> str:
        """Get path to evaluation data (all splits) for a task."""
        return f"{self.eval_data_dir}/{task}_all_splits.json"


DATA_PATHS = DataPaths()

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Model and LoRA configuration - aligned with SFT experiments."""
    
    model_name: str = "Qwen/Qwen3-8B"
    max_seq_length: int = 12288
    
    # LoRA config (must match SFT for checkpoint compatibility)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


MODEL_CONFIG = ModelConfig()

# =============================================================================
# GRPO HYPERPARAMETERS
# =============================================================================

@dataclass 
class GRPOConfig:
    """GRPO training hyperparameters."""
    
    # Generation settings
    num_generations: int = 10           # Samples per prompt
    max_completion_length: int = 2048   # Max response tokens
    temperature: float = 0.7            # Sampling temperature
    top_p: float = 0.9                  # Nucleus sampling
    
    # Training settings
    learning_rate: float = 5e-6         # Lower than SFT (RL is sensitive)
    beta: float = 0.05                  # KL penalty coefficient
    num_train_epochs: int = 1           # Usually 1 for RL
    
    # Batch settings (Phase 1 - single GPU)
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    
    # Batch settings (Phase 2 - multi GPU with DDP)
    per_device_train_batch_size_ddp: int = 2
    gradient_accumulation_steps_ddp: int = 2
    
    # Optimization
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100


GRPO_CONFIG = GRPOConfig()

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

SYSTEM_PROMPT = "You are a medical expert specializing in clinical risk prediction."

USER_PROMPT_TEMPLATE = """Based on the patient's Electronic Healthcare Record below, predict: {task_query}

Analyze the patient's condition and risk factors step by step.

--- Patient EHR ---

{context}

--- End of EHR ---

Provide your response in the following format:
<think>
[Your clinical reasoning]
</think>

Final Answer: [Positive/Negative]

IMPORTANT: YOU HAVE TO FINISH YOUR ANSWER BY SAYING EITHER "Final Answer: Positive" OR "Final Answer: Negative". AND DO NOT SAY ANYTHING ELSE AFTER THAT."""
