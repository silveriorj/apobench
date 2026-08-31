"""Offline tests that the racing config actually binds — no LLM, no dataset.

`racing_enabled`, `racing_confidence` and `racing_min_samples` were declared
in `EvalConfig` and set in YAML across many experiments, but `runner` never
passed them to the `Evaluator`. `racing_enabled: false` was a silent no-op
and the hardcoded alpha=0.05 / min=10 were always in force -- so an unknown
number of logged runs record settings that were never active. These tests
exist so that cannot recur: a declared knob must change behaviour.
"""
from pof.evaluation.evaluator import Evaluator


class _StubLLM:
    """Always wrong, so racing has every reason to eliminate."""

    def generate_batch(self, prompts, config=None, system_prompt=None):
        return ["wrong"] * len(prompts)

    def generate(self, prompt, config=None, system_prompt=None):
        return "wrong"


def _samples(n=40):
    return [{"input": f"q{i}", "target": "right"} for i in range(n)]


def _evaluator(**kw):
    return Evaluator(llm=_StubLLM(), task_type="text", batch_size=8, **kw)


class TestRacingEnabled:
    def test_defaults_are_the_historical_values(self):
        ev = _evaluator()
        assert ev.racing_enabled is True
        assert ev.racing_confidence == 0.05
        assert ev.racing_min_samples == 10

    def test_racing_terminates_early_when_enabled(self):
        ev = _evaluator()
        r = ev.evaluate_with_racing("p", _samples(), baseline_score=0.9)
        assert r.num_total < 40

    def test_disabling_racing_evaluates_every_sample(self):
        """The knob that used to do nothing."""
        ev = _evaluator(racing_enabled=False)
        r = ev.evaluate_with_racing("p", _samples(), baseline_score=0.9)
        assert r.num_total == 40

    def test_disabling_racing_also_disables_batch_racing(self):
        ev = _evaluator(racing_enabled=False)
        r = ev.evaluate_with_batch_racing("p", _samples(), threshold=0.9)
        assert r.num_total == 40

    def test_batch_racing_terminates_early_when_enabled(self):
        ev = _evaluator()
        r = ev.evaluate_with_batch_racing("p", _samples(), threshold=0.9)
        assert r.num_total < 40


class TestRacingParameters:
    def test_min_samples_from_config_delays_elimination(self):
        """No candidate may be cut before min_samples observations."""
        ev = _evaluator(racing_min_samples=30)
        r = ev.evaluate_with_racing("p", _samples(), baseline_score=0.9)
        assert r.num_total >= 30

    def test_explicit_argument_overrides_the_config_value(self):
        ev = _evaluator(racing_min_samples=30)
        r = ev.evaluate_with_racing("p", _samples(), baseline_score=0.9,
                                    min_samples=10)
        assert r.num_total < 30

    def test_confidence_from_config_is_used(self):
        """A tighter alpha widens the Hoeffding bound, so elimination is later."""
        loose = _evaluator(racing_confidence=0.5).evaluate_with_racing(
            "p", _samples(), baseline_score=0.9)
        tight = _evaluator(racing_confidence=1e-8).evaluate_with_racing(
            "p", _samples(), baseline_score=0.9)
        assert tight.num_total > loose.num_total
