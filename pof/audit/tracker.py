"""AuditTracker — per-run tracking with JSON/CSV export.

Wraps OptimizationHistory with run-level metadata, timing, and export utilities.
"""
from __future__ import annotations

import csv
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pof.audit.history import OptimizationHistory
from pof.core.types import LLMUsageStats, OptimizationResult, PromptRecord


class AuditTracker:
    """Manages a single optimization run's audit trail.

    Provides:
    - Run-level metadata (ID, method, dataset, timestamps)
    - Timing (start/end/elapsed)
    - LLM usage aggregation
    - JSON + CSV export
    - Integration with OptimizationHistory for lineage
    """

    def __init__(
        self,
        method_name: str,
        dataset_name: str,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None,
    ):
        self.run_id = str(uuid.uuid4())
        self.method_name = method_name
        self.dataset_name = dataset_name
        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")

        self.history = OptimizationHistory()
        self.history.run_id = self.run_id
        self.history.method_name = method_name
        self.history.dataset_name = dataset_name

        self.usage = LLMUsageStats()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.started_at: str = ""
        self.ended_at: str = ""
        self.notes: List[str] = []
        self.test_score: float = 0.0

    def start(self) -> None:
        """Mark the start of the optimization run."""
        self.start_time = time.time()
        self.started_at = datetime.now().isoformat()

    def end(self) -> None:
        """Mark the end of the optimization run."""
        self.end_time = time.time()
        self.ended_at = datetime.now().isoformat()

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def add_note(self, note: str) -> None:
        """Add a timestamped note to the run log."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.notes.append(f"[{ts}] {note}")

    def record_generation(
        self, generation: int, population: List[PromptRecord]
    ) -> None:
        """Delegate generation recording to history."""
        self.history.record_generation(generation, population)

    def add_record(self, record: PromptRecord) -> None:
        """Add a single record to history."""
        self.history.add_record(record)

    def get_summary(self) -> Dict[str, Any]:
        """Get run summary as a dictionary."""
        best = self.history.get_best_record()
        return {
            "run_id": self.run_id,
            "method_name": self.method_name,
            "dataset_name": self.dataset_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "total_generations": len(self.history.generations),
            "total_candidates": len(self.history.records),
            "best_score": best.score if best else 0.0,
            "test_score": self.test_score,
            "best_prompt": best.text if best and best.is_complete else "[stripped]",
            "llm_usage": self.usage.to_dict(),
            "config": self.config,
            "notes": self.notes,
        }

    def to_optimization_result(self) -> OptimizationResult:
        """Convert tracker state to an OptimizationResult."""
        best = self.history.get_best_record()
        return OptimizationResult(
            method_name=self.method_name,
            dataset_name=self.dataset_name,
            best_prompt=best.text if best else "",
            best_score=best.score if best else 0.0,
            optimization_history=[g.to_dict() for g in self.history.generations],
            final_population=[
                {"id": r.id, "prompt": r.text, "score": r.score}
                for r in sorted(
                    self.history.records.values(),
                    key=lambda x: x.score,
                    reverse=True,
                )[:10]
            ],
            all_candidates=[
                {
                    "id": r.id,
                    "operator": r.operator,
                    "score": r.score,
                    "generation_created": r.generation_created,
                    "gate": r.metadata.get("gate"),
                    "parent_ids": r.parent_ids,
                }
                for r in self.history.records.values()
            ],
            llm_usage=self.usage,
            total_time=self.elapsed_seconds,
            num_iterations=len(self.history.generations),
            config=self.config,
        )

    def save_json(self, filename: Optional[str] = None) -> Path:
        """Export full audit trail to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"audit_{self.method_name}_{self.dataset_name}_{self.run_id[:8]}.json"
        path = self.output_dir / filename
        data = {
            "summary": self.get_summary(),
            "history": self.history.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def save_csv(self, filename: Optional[str] = None) -> Path:
        """Export generation-level metrics to CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"metrics_{self.method_name}_{self.dataset_name}_{self.run_id[:8]}.csv"
        path = self.output_dir / filename

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "generation",
                    "best_score",
                    "mean_score",
                    "population_size",
                    "best_record_id",
                    "operator_counts",
                    "timestamp",
                    "test_score",
                ],
            )
            writer.writeheader()
            for gen in self.history.generations:
                row = gen.to_dict()
                row["operator_counts"] = json.dumps(row["operator_counts"])
                row["test_score"] = ""
                writer.writerow(row)
            # Final row with the test score
            if self.test_score:
                writer.writerow({
                    "generation": "final",
                    "best_score": self.history.get_best_record().score if self.history.get_best_record() else "",
                    "mean_score": "",
                    "population_size": "",
                    "best_record_id": "",
                    "operator_counts": "",
                    "timestamp": self.ended_at,
                    "test_score": self.test_score,
                })

        return path