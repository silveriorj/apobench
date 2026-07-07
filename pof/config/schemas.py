"""Pydantic v2 configuration schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM backend configuration."""

    backend: str = Field(default="huggingface", description="Backend: 'huggingface' or 'openai'")
    model_name: str = Field(default="Qwen/Qwen2.5-3B-Instruct", description="Model identifier")
    device: str = Field(default="auto", description="Device: 'auto', 'cuda', 'cpu'")
    dtype: str = Field(default="auto", description="Dtype: 'auto', 'float16', 'bfloat16'")
    max_new_tokens: int = Field(default=512, description="Default max new tokens for generation")
    temperature: float = Field(default=0.7, description="Default temperature")
    top_p: float = Field(default=0.95, description="Default top-p")
    batch_size: int = Field(default=8, description="Batch size for parallel evaluation")
    thinking_mode: bool = Field(default=False, description="Enable thinking mode (/think tags)")
    # OpenAI-specific
    api_key: Optional[str] = Field(default=None, description="OpenAI API key (or env var)")
    base_url: Optional[str] = Field(default=None, description="Custom API base URL")


class EvalConfig(BaseModel):
    """Evaluation configuration."""
    
    sample_size: int = Field(default=50, description="Number of samples for evaluation")
    full_eval_size: int = Field(default=100, description="Full evaluation sample size")
    max_new_tokens: int = Field(default=64, description="Max tokens for eval responses (model is instructed to skip CoT)")
    temperature: float = Field(default=0.0, description="Temperature for eval (0 = greedy)")
    batch_size: int = Field(default=8, description="Batch size for evaluation (GPU batching)")
    racing_enabled: bool = Field(default=True, description="Enable Hoeffding racing")
    racing_confidence: float = Field(default=0.05, description="Racing confidence level (alpha)")
    racing_min_samples: int = Field(default=10, description="Minimum samples before racing")


class BudgetConfig(BaseModel):
    """Run-level budget constraints (hard caps)."""

    time_seconds: Optional[int] = Field(default=None, description="Wall-clock time budget in seconds (None = no cap)")
    max_calls: Optional[int] = Field(default=None, description="Maximum total LLM calls")
    max_total_tokens: Optional[int] = Field(default=None, description="Maximum total tokens (input+output)")
    max_input_tokens: Optional[int] = Field(default=None, description="Maximum total input tokens")
    max_output_tokens: Optional[int] = Field(default=None, description="Maximum total output tokens")
    max_generations: Optional[int] = Field(default=None, description="Maximum optimization generations")
    early_stop_patience: int = Field(default=0, description="Stop if no improvement for this many consecutive generations")


class OptimizerConfig(BaseModel):
    """Optimizer-specific configuration."""

    method: str = Field(default="see", description="Optimization method name")
    population_size: int = Field(default=5, description="Population size (K)")
    num_iterations: int = Field(default=3, description="Number of optimization iterations/phases")
    seed_prompt: str = Field(default="", description="Initial seed prompt")
    # Method-specific params stored as dict
    params: Dict[str, Any] = Field(default_factory=dict, description="Method-specific parameters")


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str = Field(default="bbh", description="Dataset name")
    task: str = Field(default="", description="Specific task within dataset")
    split: str = Field(default="train", description="Dataset split")
    num_samples: int = Field(default=100, description="Number of samples to load")
    num_few_shot: int = Field(default=3, description="Number of few-shot examples")


class RunConfig(BaseModel):
    """Top-level run configuration combining all sub-configs."""
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    output_dir: str = Field(default="outputs", description="Output directory for results")
    seed: int = Field(default=42, description="Random seed")
    verbose: bool = Field(default=True, description="Verbose logging")
