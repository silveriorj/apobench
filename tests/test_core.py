"""Tests for core types and audit system."""
import json
import tempfile
from pathlib import Path

from pof.core.types import (
    EvalResult,
    GenerationConfig,
    LLMUsageStats,
    OptimizationResult,
    PromptRecord,
)
from pof.audit.history import OptimizationHistory
from pof.audit.tracker import AuditTracker


class TestPromptRecord:
    def test_creation(self):
        record = PromptRecord(text="Test prompt")
        assert record.text == "Test prompt"
        assert record.score == 0.0
        assert record.text_hash != ""
        assert record.is_complete is True
        assert record.id != ""

    def test_hash_consistency(self):
        r1 = PromptRecord(text="Same text")
        r2 = PromptRecord(text="Same text")
        assert r1.text_hash == r2.text_hash
        assert r1.id != r2.id  # Different UUIDs

    def test_hash_uniqueness(self):
        r1 = PromptRecord(text="Text A")
        r2 = PromptRecord(text="Text B")
        assert r1.text_hash != r2.text_hash

    def test_strip_text(self):
        record = PromptRecord(text="Full text here")
        original_hash = record.text_hash
        record.strip_text()
        assert record.text == ""
        assert record.is_complete is False
        assert record.text_hash == original_hash

    def test_serialization(self):
        record = PromptRecord(
            text="Test",
            score=0.85,
            operator="crossover",
            parent_ids=["parent1", "parent2"],
            generation_created=3,
        )
        data = record.to_dict()
        restored = PromptRecord.from_dict(data)
        assert restored.text == record.text
        assert restored.score == record.score
        assert restored.operator == record.operator
        assert restored.parent_ids == record.parent_ids
        assert restored.text_hash == record.text_hash


class TestLLMUsageStats:
    def test_merge(self):
        s1 = LLMUsageStats(total_calls=5, total_input_tokens=100, total_output_tokens=50)
        s2 = LLMUsageStats(total_calls=3, total_input_tokens=60, total_output_tokens=30)
        s1.merge(s2)
        assert s1.total_calls == 8
        assert s1.total_tokens == 240

    def test_avg_time(self):
        stats = LLMUsageStats(total_calls=4, total_time_seconds=8.0)
        assert stats.avg_time_per_call == 2.0


class TestOptimizationHistory:
    def test_add_and_retrieve(self):
        history = OptimizationHistory()
        record = PromptRecord(text="Test", score=0.9)
        history.add_record(record)
        assert history.get_record(record.id) is record

    def test_get_by_hash(self):
        history = OptimizationHistory()
        record = PromptRecord(text="Unique text")
        history.add_record(record)
        found = history.get_by_hash(record.text_hash)
        assert found is record

    def test_record_generation(self):
        history = OptimizationHistory(complete_store=2)
        records = [
            PromptRecord(text=f"Prompt {i}", score=i * 0.1)
            for i in range(5)
        ]
        history.record_generation(0, records)
        assert len(history.generations) == 1
        assert history.generations[0].best_score == 0.4
        # Stripping happens only at export time (to_dict), never on live
        # records -- optimizers keep references to these objects and
        # stripping mid-run would feed empty prompts to mutation operators.
        for r in records:
            assert r.is_complete is True
        # Top 2 by score should keep text in the export; the rest stripped
        # (all 5 share operator="init", so the diversity-keep rule doesn't
        # add anyone beyond the top-2 here).
        sorted_records = sorted(records, key=lambda r: r.score, reverse=True)
        exported = history.to_dict()["records"]
        assert exported[sorted_records[0].id]["is_complete"] is True
        assert exported[sorted_records[1].id]["is_complete"] is True
        assert exported[sorted_records[2].id]["is_complete"] is False
        assert exported[sorted_records[2].id]["text"] == ""

    def test_lineage_tracking(self):
        history = OptimizationHistory()
        parent = PromptRecord(text="Parent", score=0.5)
        child = PromptRecord(text="Child", score=0.7, parent_ids=[parent.id])
        grandchild = PromptRecord(text="Grandchild", score=0.9, parent_ids=[child.id])
        history.add_record(parent)
        history.add_record(child)
        history.add_record(grandchild)

        lineage = history.get_lineage(grandchild.id)
        assert len(lineage) == 3

    def test_save_load(self, tmp_path):
        history = OptimizationHistory()
        history.run_id = "test-run"
        history.method_name = "see"
        record = PromptRecord(text="Test", score=0.8)
        history.add_record(record)
        history.record_generation(0, [record])

        path = tmp_path / "history.json"
        history.save(path)

        loaded = OptimizationHistory.load(path)
        assert loaded.run_id == "test-run"
        assert loaded.method_name == "see"
        assert len(loaded.records) == 1
        assert len(loaded.generations) == 1


class TestAuditTracker:
    def test_lifecycle(self, tmp_path):
        tracker = AuditTracker(
            method_name="see",
            dataset_name="bbh_test",
            output_dir=tmp_path,
        )
        tracker.start()

        record = PromptRecord(text="Test", score=0.75)
        tracker.add_record(record)
        tracker.record_generation(0, [record])

        tracker.end()

        summary = tracker.get_summary()
        assert summary["method_name"] == "see"
        assert summary["best_score"] == 0.75
        assert summary["elapsed_seconds"] >= 0

    def test_export_json(self, tmp_path):
        tracker = AuditTracker(
            method_name="capo",
            dataset_name="test",
            output_dir=tmp_path,
        )
        tracker.start()
        record = PromptRecord(text="Hello", score=0.5)
        tracker.add_record(record)
        tracker.end()

        path = tracker.save_json()
        assert path.exists()
        data = json.loads(path.read_text())
        assert "summary" in data
        assert "history" in data