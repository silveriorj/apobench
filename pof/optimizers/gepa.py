"""GEPA — Grammar-guided Evolutionary Prompt Architecture.

A proposed method inspired by Grammatical Evolution (Saletta & Ferretti, GECCO 2024)
that uses structured prompt templates with evolvable components:

1. Decomposes prompts into structural components (role, task, format, constraints)
2. Evolves each component independently via grammar rules
3. Recombines components into full prompts
4. Uses structured crossover at the component level

Key features:
- Component-level evolution (more targeted than full-prompt mutation)
- Grammar-guided generation ensures structural validity
- Modular prompt architecture enables fine-grained optimization
- Component-level crossover preserves good sub-structures
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer

logger = logging.getLogger(__name__)

# Prompt component categories
COMPONENTS = ["role", "task", "format", "constraints", "examples"]


@register_optimizer("gepa")
class GEPAOptimizer(BaseOptimizer):
    """GEPA — Grammar-guided Evolutionary Prompt Architecture.

    Proposed method: evolves prompt components independently.
    """

    name = "gepa"

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 6,
        num_iterations: int = 4,
        component_mutation_rate: float = 0.4,
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
        self.component_mutation_rate = component_mutation_rate
        # Store decomposed prompts: record_id -> {component: text}
        self._decompositions: Dict[str, Dict[str, str]] = {}

    def _init_population(self) -> List[PromptRecord]:
        """Initialize with structurally diverse prompts."""
        candidates: List[PromptRecord] = []
        train_samples = self.dataset.get_few_shot_examples(n=5)

        # Generate initial prompts with explicit structure
        for i in range(self.population_size + 2):
            components = self._generate_components(train_samples, style=i)
            full_prompt = self._assemble_prompt(components)
            record = self._create_record(
                full_prompt, operator="structured_init",
                metadata={"components": components}
            )
            self._decompositions[record.id] = components
            candidates.append(record)

        # Add seed if provided
        if self.seed_prompt:
            components = self._decompose_prompt(self.seed_prompt)
            record = self._create_record(self.seed_prompt, operator="seed")
            self._decompositions[record.id] = components
            candidates.append(record)

        self._evaluate_population(candidates)
        return self._select_top_k(candidates)

    def _step(self) -> List[PromptRecord]:
        """Component-level evolution step."""
        logger.info(f"[GEPA Gen {self.generation}] Component evolution")
        candidates = list(self.population)

        # Component-level crossover
        for i in range(0, len(self.population) - 1, 2):
            parent_a = self.population[i]
            parent_b = self.population[i + 1]
            child = self._component_crossover(parent_a, parent_b)
            if child:
                candidates.append(child)

        # Component-level mutation
        for record in self.population[:self.population_size]:
            if random.random() < self.component_mutation_rate:
                mutated = self._component_mutate(record)
                if mutated:
                    candidates.append(mutated)

        # Full-prompt refinement of best
        best = self.population[0]
        refined = self._refine_structure(best)
        if refined:
            candidates.append(refined)

        # Evaluate and select
        new_candidates = [c for c in candidates if c.score == 0.0]
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_racing(new_candidates, baseline)
        return self._select_top_k(candidates)

    def _generate_components(
        self, samples: List[Dict[str, str]], style: int = 0
    ) -> Dict[str, str]:
        """Generate prompt components with a specific style."""
        examples_text = "\n".join(
            f"Input: {s['input'][:60]}\nOutput: {s['target']}"
            for s in samples[:3]
        )

        styles = [
            "concise and direct",
            "detailed and thorough",
            "structured with clear steps",
            "focused on edge cases",
            "emphasizing output format",
            "using analogies and examples",
            "formal and precise",
            "conversational and clear",
        ]
        style_desc = styles[style % len(styles)]

        meta_prompt = (
            f"Generate a {style_desc} instruction for a task. "
            f"Structure your response with these labeled sections:\n"
            f"ROLE: (who the AI should act as)\n"
            f"TASK: (what to do)\n"
            f"FORMAT: (output format requirements)\n"
            f"CONSTRAINTS: (rules and limitations)\n\n"
            f"Based on these examples:\n{examples_text}\n\n"
            f"Structured instruction:"
        )

        result = self._generate_prompt(meta_prompt, temperature=0.8)
        return self._parse_components(result)

    def _parse_components(self, text: str) -> Dict[str, str]:
        """Parse structured text into components."""
        components = {
            "role": "",
            "task": "",
            "format": "",
            "constraints": "",
        }

        current_key = None
        lines = text.split("\n")

        for line in lines:
            line_lower = line.strip().lower()
            if line_lower.startswith("role:"):
                current_key = "role"
                components["role"] = line.split(":", 1)[1].strip()
            elif line_lower.startswith("task:"):
                current_key = "task"
                components["task"] = line.split(":", 1)[1].strip()
            elif line_lower.startswith("format:"):
                current_key = "format"
                components["format"] = line.split(":", 1)[1].strip()
            elif line_lower.startswith("constraints:") or line_lower.startswith("constraint:"):
                current_key = "constraints"
                components["constraints"] = line.split(":", 1)[1].strip()
            elif current_key and line.strip():
                components[current_key] += " " + line.strip()

        # Fallback: if parsing failed, put everything in task
        if not any(components.values()):
            components["task"] = text.strip()

        return components

    def _decompose_prompt(self, prompt: str) -> Dict[str, str]:
        """Decompose an existing prompt into components via LLM."""
        meta_prompt = (
            "Decompose this instruction into structured components:\n"
            "ROLE: (who the AI should act as)\n"
            "TASK: (what to do)\n"
            "FORMAT: (output format)\n"
            "CONSTRAINTS: (rules)\n\n"
            f"Instruction:\n{prompt}\n\n"
            "Decomposition:"
        )
        result = self._generate_prompt(meta_prompt, temperature=0.3)
        return self._parse_components(result)

    def _assemble_prompt(self, components: Dict[str, str]) -> str:
        """Assemble components into a full prompt."""
        parts = []
        if components.get("role"):
            parts.append(components["role"])
        if components.get("task"):
            parts.append(components["task"])
        if components.get("format"):
            parts.append(f"Output format: {components['format']}")
        if components.get("constraints"):
            parts.append(f"Constraints: {components['constraints']}")
        return "\n".join(parts) if parts else "Solve the task."

    def _component_crossover(
        self, parent_a: PromptRecord, parent_b: PromptRecord
    ) -> Optional[PromptRecord]:
        """Crossover at the component level."""
        comp_a = self._decompositions.get(parent_a.id)
        comp_b = self._decompositions.get(parent_b.id)

        if not comp_a or not comp_b:
            # Fallback to full-prompt crossover
            text = self._crossover(parent_a.text, parent_b.text)
            if text.strip():
                record = self._create_record(
                    text, operator="fallback_crossover",
                    parent_ids=[parent_a.id, parent_b.id]
                )
                self._decompositions[record.id] = self._parse_components(text)
                return record
            return None

        # Randomly select components from each parent
        child_components = {}
        for key in COMPONENTS[:4]:  # role, task, format, constraints
            if random.random() < 0.5:
                child_components[key] = comp_a.get(key, "")
            else:
                child_components[key] = comp_b.get(key, "")

        full_prompt = self._assemble_prompt(child_components)
        record = self._create_record(
            full_prompt, operator="component_crossover",
            parent_ids=[parent_a.id, parent_b.id],
            metadata={"components": child_components}
        )
        self._decompositions[record.id] = child_components
        return record

    def _component_mutate(self, record: PromptRecord) -> Optional[PromptRecord]:
        """Mutate a single component of the prompt."""
        components = self._decompositions.get(record.id)
        if not components:
            components = self._decompose_prompt(record.text)

        # Pick a random component to mutate
        mutable = [k for k, v in components.items() if v]
        if not mutable:
            return None

        target_component = random.choice(mutable)
        original_value = components[target_component]

        meta_prompt = (
            f"Rewrite this {target_component} section to be more effective. "
            f"Keep the same intent but improve clarity or precision.\n\n"
            f"Original {target_component}: {original_value}\n\n"
            f"Improved {target_component}:"
        )
        new_value = self._generate_prompt(meta_prompt, temperature=0.7)

        if not new_value or not new_value.strip():
            return None

        # Create mutated components
        new_components = dict(components)
        new_components[target_component] = new_value.strip()
        full_prompt = self._assemble_prompt(new_components)

        new_record = self._create_record(
            full_prompt, operator=f"mutate_{target_component}",
            parent_ids=[record.id],
            metadata={"mutated_component": target_component, "components": new_components}
        )
        self._decompositions[new_record.id] = new_components
        return new_record

    def _refine_structure(self, record: PromptRecord) -> Optional[PromptRecord]:
        """Refine the overall structure/coherence of the best prompt."""
        meta_prompt = (
            "This instruction works well but could be more coherent. "
            "Rewrite it to flow better while keeping all the key elements.\n\n"
            f"Instruction:\n{record.text}\n\n"
            "Refined instruction:"
        )
        result = self._generate_prompt(meta_prompt, temperature=0.5)
        if not result or not result.strip():
            return None

        new_record = self._create_record(
            result.strip(), operator="structure_refine",
            parent_ids=[record.id]
        )
        self._decompositions[new_record.id] = self._parse_components(result)
        return new_record