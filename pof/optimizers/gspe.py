"""GSPE — Grammar-guided Structured Prompt Evolution.

A proposed method from the prompt-optimization-framework that uses:
1. Structured prompt specification with typed fields
2. Grammar-guided generation ensuring valid prompt structures
3. Field-level crossover and mutation
4. Hierarchical evaluation (structure validity → task performance)

Key differentiator from GEPA: GSPE uses a formal grammar specification
to define valid prompt structures, while GEPA uses free-form component
decomposition. GSPE is more constrained but produces more consistent results.

Design reference: docs/gspe_design.md from prompt-optimization-framework.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer

logger = logging.getLogger(__name__)

# Grammar specification for prompt structure
PROMPT_GRAMMAR = {
    "preamble": {
        "description": "Context-setting opening (role, expertise, context)",
        "required": True,
        "max_tokens": 50,
    },
    "task_definition": {
        "description": "Clear statement of what to do",
        "required": True,
        "max_tokens": 100,
    },
    "output_spec": {
        "description": "Expected output format and constraints",
        "required": True,
        "max_tokens": 50,
    },
    "reasoning_guide": {
        "description": "Step-by-step reasoning instructions",
        "required": False,
        "max_tokens": 80,
    },
    "error_prevention": {
        "description": "Common mistakes to avoid",
        "required": False,
        "max_tokens": 60,
    },
}


@register_optimizer("gspe")
class GSPEOptimizer(BaseOptimizer):
    """GSPE — Grammar-guided Structured Prompt Evolution.

    Proposed method: uses formal grammar to constrain prompt structure.
    Needs validation.
    """

    name = "gspe"

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 6,
        num_iterations: int = 4,
        field_mutation_rate: float = 0.3,
        structure_crossover_rate: float = 0.7,
        **kwargs,
    ):
        super().__init__(
            llm=llm,
            dataset=dataset,
            evaluator=evaluator,
            population_size=population_size,
            num_iterations=num_iterations,
            **kwargs,
        )
        self.field_mutation_rate = field_mutation_rate
        self.structure_crossover_rate = structure_crossover_rate
        # Store structured representations: record_id -> {field: text}
        self._structures: Dict[str, Dict[str, str]] = {}

    def _init_population(self) -> List[PromptRecord]:
        """Initialize population with grammar-valid structured prompts."""
        candidates: List[PromptRecord] = []
        train_samples = self.dataset.get_few_shot_examples(n=5)

        # Generate diverse structured prompts
        strategies = [
            "minimal",      # Only required fields
            "full",         # All fields
            "reasoning",    # Emphasis on reasoning_guide
            "constrained",  # Emphasis on error_prevention
            "balanced",     # Equal weight to all
            "task_focused", # Heavy task_definition
        ]

        for strategy in strategies[:self.population_size + 2]:
            structure = self._generate_structure(train_samples, strategy)
            prompt_text = self._render_structure(structure)
            record = self._create_record(
                prompt_text, operator=f"gspe_init_{strategy}",
                metadata={"structure": structure, "strategy": strategy}
            )
            self._structures[record.id] = structure
            candidates.append(record)

        # Add seed prompt if provided (decompose into structure)
        if self.seed_prompt:
            structure = self._decompose_to_structure(self.seed_prompt)
            record = self._create_record(self.seed_prompt, operator="seed")
            self._structures[record.id] = structure
            candidates.append(record)

        self._evaluate_population(candidates)
        return self._select_top_k(candidates)

    def _step(self) -> List[PromptRecord]:
        """Grammar-guided evolution step."""
        logger.info(f"[GSPE Gen {self.generation}] Structured evolution")
        candidates = list(self.population)

        # Structure-aware crossover
        for i in range(0, len(self.population) - 1, 2):
            if random.random() < self.structure_crossover_rate:
                child = self._structure_crossover(
                    self.population[i], self.population[i + 1]
                )
                if child:
                    candidates.append(child)

        # Field-level mutation
        for record in self.population[:self.population_size]:
            if random.random() < self.field_mutation_rate:
                mutated = self._field_mutation(record)
                if mutated:
                    candidates.append(mutated)

        # Grammar-guided refinement of best
        best = self.population[0]
        refined = self._grammar_refine(best)
        if refined:
            candidates.append(refined)

        # Failure-guided field improvement
        failures_improved = self._failure_guided_field_improve(best)
        if failures_improved:
            candidates.append(failures_improved)

        # Evaluate and select
        new_candidates = [c for c in candidates if c.score == 0.0]
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_racing(new_candidates, baseline)
        return self._select_top_k(candidates)

    def _generate_structure(
        self, samples: List[Dict[str, str]], strategy: str
    ) -> Dict[str, str]:
        """Generate a structured prompt following the grammar."""
        examples_text = "\n".join(
            f"Input: {s['input'][:60]}\nOutput: {s['target']}"
            for s in samples[:3]
        )

        # Determine which fields to generate based on strategy
        fields_to_generate = list(PROMPT_GRAMMAR.keys())
        if strategy == "minimal":
            fields_to_generate = [f for f, spec in PROMPT_GRAMMAR.items() if spec["required"]]
        elif strategy == "reasoning":
            fields_to_generate = ["preamble", "task_definition", "reasoning_guide", "output_spec"]
        elif strategy == "constrained":
            fields_to_generate = ["preamble", "task_definition", "error_prevention", "output_spec"]

        field_descriptions = "\n".join(
            f"- {field.upper()}: {PROMPT_GRAMMAR[field]['description']} (max ~{PROMPT_GRAMMAR[field]['max_tokens']} tokens)"
            for field in fields_to_generate
        )

        meta_prompt = (
            f"Generate a structured instruction for a task. "
            f"Fill in EACH of these fields on a separate line:\n"
            f"{field_descriptions}\n\n"
            f"Based on these examples:\n{examples_text}\n\n"
            f"Strategy: {strategy}\n\n"
            f"Fill each field (use the format 'FIELD_NAME: content'):"
        )

        result = self._generate_prompt(meta_prompt, temperature=0.8)
        return self._parse_structure(result, fields_to_generate)

    def _parse_structure(
        self, text: str, expected_fields: List[str]
    ) -> Dict[str, str]:
        """Parse LLM output into structured fields."""
        structure: Dict[str, str] = {}
        current_field = None

        for line in text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check if line starts a new field
            matched = False
            for field in expected_fields:
                field_upper = field.upper()
                if line_stripped.upper().startswith(f"{field_upper}:"):
                    current_field = field
                    content = line_stripped.split(":", 1)[1].strip()
                    structure[field] = content
                    matched = True
                    break

            if not matched and current_field:
                # Continuation of previous field
                structure[current_field] = structure.get(current_field, "") + " " + line_stripped

        # Fallback: if parsing failed, put everything in task_definition
        if not structure:
            structure["task_definition"] = text.strip()

        return structure

    def _render_structure(self, structure: Dict[str, str]) -> str:
        """Render a structure into a coherent prompt text."""
        parts = []
        # Render in grammar order
        for field in PROMPT_GRAMMAR:
            if field in structure and structure[field]:
                parts.append(structure[field])
        return "\n".join(parts) if parts else "Solve the task."

    def _decompose_to_structure(self, prompt: str) -> Dict[str, str]:
        """Decompose a free-form prompt into grammar fields."""
        field_list = "\n".join(
            f"- {field.upper()}: {spec['description']}"
            for field, spec in PROMPT_GRAMMAR.items()
        )

        meta_prompt = (
            f"Decompose this instruction into structured fields:\n"
            f"{field_list}\n\n"
            f"Instruction:\n{prompt}\n\n"
            f"Decomposition (use 'FIELD_NAME: content' format):"
        )
        result = self._generate_prompt(meta_prompt, temperature=0.3)
        return self._parse_structure(result, list(PROMPT_GRAMMAR.keys()))

    def _structure_crossover(
        self, parent_a: PromptRecord, parent_b: PromptRecord
    ) -> Optional[PromptRecord]:
        """Crossover at the field level, respecting grammar constraints."""
        struct_a = self._structures.get(parent_a.id, {})
        struct_b = self._structures.get(parent_b.id, {})

        if not struct_a or not struct_b:
            # Fallback
            text = self._crossover(parent_a.text, parent_b.text)
            if text.strip():
                record = self._create_record(
                    text, operator="gspe_fallback_crossover",
                    parent_ids=[parent_a.id, parent_b.id]
                )
                self._structures[record.id] = self._parse_structure(text, list(PROMPT_GRAMMAR.keys()))
                return record
            return None

        # Field-level crossover: for each field, pick from better-scoring parent
        # with some randomness
        child_structure = {}
        for field in PROMPT_GRAMMAR:
            val_a = struct_a.get(field, "")
            val_b = struct_b.get(field, "")

            if val_a and val_b:
                # Prefer from higher-scoring parent with 70% probability
                if parent_a.score >= parent_b.score:
                    child_structure[field] = val_a if random.random() < 0.7 else val_b
                else:
                    child_structure[field] = val_b if random.random() < 0.7 else val_a
            elif val_a:
                child_structure[field] = val_a
            elif val_b:
                child_structure[field] = val_b

        prompt_text = self._render_structure(child_structure)
        record = self._create_record(
            prompt_text, operator="gspe_crossover",
            parent_ids=[parent_a.id, parent_b.id],
            metadata={"structure": child_structure}
        )
        self._structures[record.id] = child_structure
        return record

    def _field_mutation(self, record: PromptRecord) -> Optional[PromptRecord]:
        """Mutate a single field while preserving grammar validity."""
        structure = self._structures.get(record.id, {})
        if not structure:
            structure = self._decompose_to_structure(record.text)

        # Pick a random field to mutate
        populated_fields = [f for f, v in structure.items() if v]
        if not populated_fields:
            return None

        target_field = random.choice(populated_fields)
        field_spec = PROMPT_GRAMMAR.get(target_field, {})

        meta_prompt = (
            f"Improve this {target_field.replace('_', ' ')} section of an instruction. "
            f"It should be: {field_spec.get('description', 'clear and effective')}. "
            f"Keep it concise (max ~{field_spec.get('max_tokens', 50)} tokens).\n\n"
            f"Current: {structure[target_field]}\n\n"
            f"Improved:"
        )
        new_value = self._generate_prompt(meta_prompt, temperature=0.7)

        if not new_value or not new_value.strip():
            return None

        new_structure = dict(structure)
        new_structure[target_field] = new_value.strip()
        prompt_text = self._render_structure(new_structure)

        new_record = self._create_record(
            prompt_text, operator=f"gspe_mutate_{target_field}",
            parent_ids=[record.id],
            metadata={"mutated_field": target_field, "structure": new_structure}
        )
        self._structures[new_record.id] = new_structure
        return new_record

    def _grammar_refine(self, record: PromptRecord) -> Optional[PromptRecord]:
        """Refine prompt to better conform to grammar while improving quality."""
        structure = self._structures.get(record.id, {})
        if not structure:
            return None

        # Check which required fields are missing
        missing = [
            f for f, spec in PROMPT_GRAMMAR.items()
            if spec["required"] and not structure.get(f)
        ]

        if missing:
            # Generate missing required fields
            meta_prompt = (
                f"This instruction is missing these components: {', '.join(missing)}.\n"
                f"Add them based on the existing instruction.\n\n"
                f"Existing instruction:\n{record.text}\n\n"
                f"Complete instruction with all components:"
            )
            result = self._generate_prompt(meta_prompt, temperature=0.5)
            if result and result.strip():
                new_structure = self._parse_structure(result, list(PROMPT_GRAMMAR.keys()))
                # Merge: keep existing fields, add new ones
                for field in missing:
                    if field in new_structure:
                        structure[field] = new_structure[field]

        # Re-render with complete structure
        prompt_text = self._render_structure(structure)
        if prompt_text == record.text:
            return None

        new_record = self._create_record(
            prompt_text, operator="gspe_grammar_refine",
            parent_ids=[record.id],
            metadata={"structure": structure}
        )
        self._structures[new_record.id] = structure
        return new_record

    def _failure_guided_field_improve(self, record: PromptRecord) -> Optional[PromptRecord]:
        """Analyze failures and improve the most relevant field."""
        samples = self.dataset.get_eval_samples("dev", n=20)
        result = self.evaluator.evaluate(record.text, samples)
        failures = [d for d in result.per_sample_details if not d["correct"]]

        if not failures:
            return None

        structure = self._structures.get(record.id, {})
        if not structure:
            return None

        # Ask LLM which field needs improvement based on failures
        failure_text = "\n".join(
            f"- Expected: {f['target']} | Got: {f['prediction'][:40]}"
            for f in failures[:3]
        )

        fields_text = "\n".join(
            f"- {field}: {value[:50]}" for field, value in structure.items() if value
        )

        meta_prompt = (
            f"Given these failures and the current prompt structure, "
            f"which field needs the most improvement and how?\n\n"
            f"Failures:\n{failure_text}\n\n"
            f"Current structure:\n{fields_text}\n\n"
            f"Identify the weakest field and provide an improved version.\n"
            f"Format: FIELD_NAME: improved content"
        )
        result_text = self._generate_prompt(meta_prompt, temperature=0.6)

        if not result_text or not result_text.strip():
            return None

        # Parse the improvement
        new_fields = self._parse_structure(result_text, list(PROMPT_GRAMMAR.keys()))
        if not new_fields:
            return None

        # Apply the improvement
        new_structure = dict(structure)
        for field, value in new_fields.items():
            if value:
                new_structure[field] = value

        prompt_text = self._render_structure(new_structure)
        new_record = self._create_record(
            prompt_text, operator="gspe_failure_field_improve",
            parent_ids=[record.id],
            metadata={"structure": new_structure}
        )
        self._structures[new_record.id] = new_structure
        return new_record