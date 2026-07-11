"""Core types for the Prompt Optimization Framework.

PromptRecord is the atomic unit of traceability — every candidate prompt gets one.
It tracks lineage (parents, operator), scoring, and enables cross-run deduplication
via SHA-256 text hashing.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptRecord:
    """Immutable record of a prompt candidate with full lineage.

    Attributes:
        id: Unique identifier (UUID4).
        text: The prompt text (may be stripped for storage efficiency).
        text_hash: SHA-256 of the prompt text for deduplication.
        score: Primary evaluation score.
        scores: Multi-split scoring dict (e.g., {"dev": 0.87, "test": 0.91}).
        parent_ids: IDs of parent records (for lineage tracking).
        operator: The operator/technique that produced this candidate.
        operator_params: Parameters used by the operator.
        generation_created: Generation number when this candidate was born.
        generation_last_active: Last generation this candidate was in the population.
        metadata: Arbitrary metadata dict.
        is_complete: Whether full text is stored (False = text stripped, hash preserved).
        performance_vector: Per-sample binary results for fine-grained analysis.
    """

    text: str
    score: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    parent_ids: List[str] = field(default_factory=list)
    operator: str = "init"
    operator_params: Dict[str, Any] = field(default_factory=dict)
    generation_created: int = 0
    generation_last_active: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_vector: List[int] = field(default_factory=list)
    per_sample_details: List[Dict[str, Any]] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text_hash: str = field(default="", repr=False)
    is_complete: bool = True

    def __post_init__(self):
        if not self.text_hash:
            self.text_hash = self._compute_hash(self.text)

    @staticmethod
    def _compute_hash(text: str) -> str:
        """SHA-256 hash of prompt text for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def strip_text(self) -> None:
        """Remove full text to save storage, preserving hash for identity."""
        self.text = ""
        self.is_complete = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "text_hash": self.text_hash,
            "score": self.score,
            "scores": self.scores,
            "parent_ids": self.parent_ids,
            "operator": self.operator,
            "operator_params": self.operator_params,
            "generation_created": self.generation_created,
            "generation_last_active": self.generation_last_active,
            "metadata": self.metadata,
            "performance_vector": self.performance_vector,
            "per_sample_details": self.per_sample_details,
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptRecord":
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            text=data.get("text", ""),
            text_hash=data.get("text_hash", ""),
            score=data.get("score", 0.0),
            scores=data.get("scores", {}),
            parent_ids=data.get("parent_ids", []),
            operator=data.get("operator", "init"),
            operator_params=data.get("operator_params", {}),
            generation_created=data.get("generation_created", 0),
            generation_last_active=data.get("generation_last_active", 0),
            metadata=data.get("metadata", {}),
            performance_vector=data.get("performance_vector", []),
            per_sample_details=data.get("per_sample_details", []),
            is_complete=data.get("is_complete", True),
        )


@dataclass
class EvalResult:
    """Result of evaluating a prompt on a set of samples."""

    score: float
    num_correct: int = 0
    num_total: int = 0
    performance_vector: List[int] = field(default_factory=list)
    per_sample_details: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Configuration for LLM text generation."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8    # Qwen3 instruct recommendation
    top_k: int = 20       # Qwen3 instruct recommendation
    do_sample: bool = True
    num_return_sequences: int = 1
    stop_sequences: List[str] = field(default_factory=list)
    repetition_penalty: float = 1.0


@dataclass
class LLMUsageStats:
    """Track LLM usage for cost/efficiency analysis."""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_time_seconds: float = 0.0
    generation_calls: int = 0
    evaluation_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_time_per_call(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_time_seconds / self.total_calls

    def merge(self, other: "LLMUsageStats") -> None:
        """Merge another stats object into this one."""
        self.total_calls += other.total_calls
        self.total_input_tokens += other.total_input_tokens
        self.total_output_tokens += other.total_output_tokens
        self.total_time_seconds += other.total_time_seconds
        self.generation_calls += other.generation_calls
        self.evaluation_calls += other.evaluation_calls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "avg_time_per_call": round(self.avg_time_per_call, 4),
            "generation_calls": self.generation_calls,
            "evaluation_calls": self.evaluation_calls,
        }


@dataclass
class OptimizationResult:
    """Result of a complete optimization run."""

    method_name: str
    dataset_name: str
    best_prompt: str
    best_score: float          # score on dev/eval samples during optimization
    test_score: float = 0.0   # score on held-out test samples after optimization
    test_per_sample_details: List[Dict[str, Any]] = field(default_factory=list)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    final_population: List[Dict[str, Any]] = field(default_factory=list)
    llm_usage: Optional[LLMUsageStats] = None
    total_time: float = 0.0
    num_iterations: int = 0
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "dataset_name": self.dataset_name,
            "best_prompt": self.best_prompt,
            "best_score": self.best_score,
            "test_score": self.test_score,
            "test_per_sample_details": self.test_per_sample_details,
            "optimization_history": self.optimization_history,
            "final_population": self.final_population,
            "llm_usage": self.llm_usage.to_dict() if self.llm_usage else None,
            "total_time": round(self.total_time, 2),
            "num_iterations": self.num_iterations,
            "config": self.config,
        }