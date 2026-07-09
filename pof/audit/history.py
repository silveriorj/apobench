"""OptimizationHistory — full per-candidate lineage tracking.

Provides storage-efficient partial text storage: top-N candidates per generation
keep full text, the rest are stripped to hash-only for deduplication and audit.

Key features:
- Per-generation snapshots with population state
- Full parent→child lineage graph
- Storage-efficient: configurable complete_store (top-N keep text)
- Export to JSON/CSV for analysis
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pof.core.types import PromptRecord


@dataclass
class GenerationSnapshot:
    """Snapshot of population state at a given generation."""

    generation: int
    best_score: float
    mean_score: float
    population_size: int
    best_record_id: str
    operator_counts: Dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "best_score": self.best_score,
            "mean_score": round(self.mean_score, 4),
            "population_size": self.population_size,
            "best_record_id": self.best_record_id,
            "operator_counts": self.operator_counts,
            "timestamp": self.timestamp,
        }


class OptimizationHistory:
    """Full lineage tracking for an optimization run.

    Args:
        complete_store: Number of top candidates per generation to keep full text.
            None = keep all, 0 = strip all, N = keep top-N.
    """

    def __init__(self, complete_store: Optional[int] = 3):
        self.complete_store = complete_store
        self.records: Dict[str, PromptRecord] = {}
        self.generations: List[GenerationSnapshot] = []
        self.run_id: str = ""
        self.method_name: str = ""
        self.dataset_name: str = ""
        self.started_at: str = datetime.now().isoformat()

    def add_record(self, record: PromptRecord) -> None:
        """Add a prompt record to the history."""
        self.records[record.id] = record

    def get_record(self, record_id: str) -> Optional[PromptRecord]:
        """Retrieve a record by ID."""
        return self.records.get(record_id)

    def get_by_hash(self, text_hash: str) -> Optional[PromptRecord]:
        """Find a record by its text hash (deduplication check)."""
        for record in self.records.values():
            if record.text_hash == text_hash:
                return record
        return None

    def record_generation(
        self,
        generation: int,
        population: List[PromptRecord],
    ) -> None:
        """Record a generation snapshot and manage storage efficiency.

        Args:
            generation: Current generation number.
            population: Current population of candidates.
        """
        if not population:
            return

        # Update generation_last_active for all in population
        for record in population:
            record.generation_last_active = generation
            if record.id not in self.records:
                self.add_record(record)

        # Sort by score descending
        sorted_pop = sorted(population, key=lambda r: r.score, reverse=True)

        # Compute operator counts
        operator_counts: Dict[str, int] = {}
        for r in population:
            op = r.operator
            operator_counts[op] = operator_counts.get(op, 0) + 1

        # Create snapshot
        scores = [r.score for r in population]
        snapshot = GenerationSnapshot(
            generation=generation,
            best_score=sorted_pop[0].score,
            mean_score=sum(scores) / len(scores),
            population_size=len(population),
            best_record_id=sorted_pop[0].id,
            operator_counts=operator_counts,
        )
        self.generations.append(snapshot)
        # NOTE: text stripping happens at export time (to_dict), never on live
        # records — optimizers keep references to these objects, and stripping
        # mid-run would feed empty prompts to crossover/mutation operators.

    def get_lineage(self, record_id: str) -> List[PromptRecord]:
        """Trace full lineage of a record back to its roots."""
        lineage = []
        visited = set()
        queue = [record_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            record = self.records.get(current_id)
            if record:
                lineage.append(record)
                queue.extend(record.parent_ids)

        return lineage

    def get_best_record(self) -> Optional[PromptRecord]:
        """Get the best-scoring record across all generations."""
        if not self.records:
            return None
        return max(self.records.values(), key=lambda r: r.score)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize history to dictionary (storage-efficient).

        Full text is exported only for a representative subset: the overall
        top-`complete_store` records by score, plus the best record of each
        operator (diversity), plus the seed. All other records export
        hash-only. Per-sample evaluation details are exported only for the
        single best record — everything else would just flood the files.
        """
        sorted_recs = sorted(
            self.records.values(), key=lambda r: r.score, reverse=True
        )
        best = sorted_recs[0] if sorted_recs else None

        if self.complete_store is None:
            keep_ids = {r.id for r in sorted_recs}
        else:
            keep_ids = {r.id for r in sorted_recs[: self.complete_store]}
            # Diversity: keep the best example of each operator + the seed
            seen_ops = set()
            for r in sorted_recs:
                if r.operator not in seen_ops:
                    seen_ops.add(r.operator)
                    keep_ids.add(r.id)

        records_out: Dict[str, Any] = {}
        for rid, r in self.records.items():
            d = r.to_dict()
            if rid not in keep_ids:
                d["text"] = ""
                d["is_complete"] = False
            if best is None or rid != best.id:
                d["per_sample_details"] = []
            records_out[rid] = d

        return {
            "run_id": self.run_id,
            "method_name": self.method_name,
            "dataset_name": self.dataset_name,
            "started_at": self.started_at,
            "total_records": len(self.records),
            "total_generations": len(self.generations),
            "generations": [g.to_dict() for g in self.generations],
            "records": records_out,
        }

    def save(self, path: Path) -> None:
        """Save history to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "OptimizationHistory":
        """Load history from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        history = cls()
        history.run_id = data.get("run_id", "")
        history.method_name = data.get("method_name", "")
        history.dataset_name = data.get("dataset_name", "")
        history.started_at = data.get("started_at", "")

        for rid, rdata in data.get("records", {}).items():
            history.records[rid] = PromptRecord.from_dict(rdata)

        for gdata in data.get("generations", []):
            history.generations.append(
                GenerationSnapshot(
                    generation=gdata["generation"],
                    best_score=gdata["best_score"],
                    mean_score=gdata["mean_score"],
                    population_size=gdata["population_size"],
                    best_record_id=gdata["best_record_id"],
                    operator_counts=gdata.get("operator_counts", {}),
                    timestamp=gdata.get("timestamp", ""),
                )
            )

        return history