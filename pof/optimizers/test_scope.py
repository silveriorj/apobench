"""Offline unit tests for SCOPE's five mechanisms — no LLM, no dataset.

Each mechanism is arithmetic over `performance_vector` data, so all of it is
testable without loading a model. These run in well under a second and are
the first line of defence: M1 in particular corrupts every downstream
comparison silently if its re-indexing is wrong, so it is tested against
hand-built vectors where the right answer is obvious by inspection.
"""
import pytest

from pof.core.types import PromptRecord
from pof.optimizers.scope import SCOPEOptimizer


def _bare_scope(**overrides):
    """A SCOPEOptimizer with only the attributes these methods read."""
    opt = object.__new__(SCOPEOptimizer)
    opt._calibrated = False
    opt._signal_pool = []
    opt.population = []
    opt.population_size = 5
    opt.frontier_pull_prob = 0.0
    opt.use_signal_calibration = True
    opt.use_retention_gate = True
    opt.use_cost_front = True
    opt.tracker = type("_T", (), {"add_note": lambda self, note: None})()
    for k, v in overrides.items():
        setattr(opt, k, v)
    return opt


def _rec(text, score=0.0, vector=None, details=None, rid=None):
    r = PromptRecord(text=text, score=score)
    if rid:
        r.id = rid
    if vector is not None:
        r.performance_vector = list(vector)
        r.scores["dev"] = score
    if details is not None:
        r.per_sample_details = details
    return r


# ---------------------------------------------------------------- M1 ----

class TestSignalCalibration:
    def test_discrimination_is_zero_for_unanimous_instances(self):
        # instance 0: everyone solves it; instance 1: nobody does;
        # instance 2: candidates disagree -> only instance 2 discriminates.
        vectors = [[1, 0, 1], [1, 0, 0], [1, 0, 1]]
        disc = SCOPEOptimizer._discrimination(vectors, 3)
        assert disc[0] == 0.0, "an instance everyone solves carries no signal"
        assert disc[1] == 0.0, "an instance nobody solves carries no signal"
        assert disc[2] > 0.0, "an instance candidates disagree on is the signal"

    def test_calibration_drops_unanimous_and_reindexes_vectors(self):
        opt = _bare_scope(
            _signal_pool=[{"input": f"i{i}"} for i in range(4)],
            MIN_SIGNAL_POOL=1,
        )
        # instances 0 and 3 unanimous; 1 and 2 discriminate.
        a = _rec("a", 0.5, [1, 1, 0, 0], details=[{"p": 0}, {"p": 1}, {"p": 2}, {"p": 3}])
        b = _rec("b", 0.25, [1, 0, 1, 0], details=[{"p": 0}, {"p": 1}, {"p": 2}, {"p": 3}])
        opt._calibrate_signal_pool([a, b])

        assert len(opt._signal_pool) == 2
        assert [s["input"] for s in opt._signal_pool] == ["i1", "i2"]
        # Vectors must be re-indexed onto the surviving instances, in order.
        assert a.performance_vector == [1, 0]
        assert b.performance_vector == [0, 1]
        # per_sample_details must be re-indexed identically, or later
        # retention/cost reads would refer to the wrong instances.
        assert [d["p"] for d in a.per_sample_details] == [1, 2]
        # Scores are recomputed on the calibrated pool.
        assert a.score == pytest.approx(0.5)
        assert b.score == pytest.approx(0.5)

    def test_calibration_respects_min_pool_floor(self):
        opt = _bare_scope(
            _signal_pool=[{"input": f"i{i}"} for i in range(5)],
            MIN_SIGNAL_POOL=4,
        )
        # only instance 2 discriminates, but the floor forbids shrinking to 1
        a = _rec("a", 0.4, [1, 1, 1, 0, 0])
        b = _rec("b", 0.2, [1, 1, 0, 0, 0])
        opt._calibrate_signal_pool([a, b])
        assert len(opt._signal_pool) == 4, "must not shrink below MIN_SIGNAL_POOL"
        assert len(a.performance_vector) == 4

    def test_calibration_is_a_noop_when_everything_discriminates(self):
        opt = _bare_scope(_signal_pool=[{"input": "x"}, {"input": "y"}], MIN_SIGNAL_POOL=1)
        a = _rec("a", 0.5, [1, 0])
        b = _rec("b", 0.5, [0, 1])
        opt._calibrate_signal_pool([a, b])
        assert len(opt._signal_pool) == 2
        assert a.performance_vector == [1, 0]

    def test_calibration_runs_only_once(self):
        opt = _bare_scope(_signal_pool=[{"input": f"i{i}"} for i in range(3)], MIN_SIGNAL_POOL=1)
        a = _rec("a", 0.33, [1, 1, 0])
        b = _rec("b", 0.0, [1, 0, 0])
        opt._calibrate_signal_pool([a, b])
        first = list(opt._signal_pool)
        opt._calibrate_signal_pool([a, b])
        assert opt._signal_pool == first, "second call must be inert"

    def test_calibration_needs_two_candidates(self):
        opt = _bare_scope(_signal_pool=[{"input": "x"}, {"input": "y"}], MIN_SIGNAL_POOL=1)
        opt._calibrate_signal_pool([_rec("a", 0.5, [1, 0])])
        assert len(opt._signal_pool) == 2, "one candidate cannot define discrimination"


# ---------------------------------------------------------------- M2 ----

class TestRetention:
    def test_full_retention(self):
        assert SCOPEOptimizer._retention([1, 1, 0], [1, 1, 0]) == 1.0

    def test_partial_retention(self):
        # parent solved 0 and 1; child kept only 0
        assert SCOPEOptimizer._retention([1, 0, 1], [1, 1, 0]) == pytest.approx(0.5)

    def test_parent_solved_nothing_is_not_a_regression(self):
        assert SCOPEOptimizer._retention([0, 0], [0, 0]) == 1.0

    def test_child_may_solve_new_instances_without_penalty(self):
        # retention measures preservation only; instance 2 is a bonus
        assert SCOPEOptimizer._retention([1, 1, 1], [1, 1, 0]) == 1.0

    def test_short_child_vector_does_not_crash(self):
        assert SCOPEOptimizer._retention([1], [1, 1]) == pytest.approx(0.5)


# ---------------------------------------------------------------- M3 ----

class TestGuardedAcceptance:
    def test_rejects_destructive_child(self):
        """Lost content AND gained nothing -> the only genuinely bad case."""
        opt = _bare_scope(RETENTION_FLOOR=0.8)
        parent = _rec("p", 0.75, [1, 1, 1, 0])
        child = _rec("c", 0.25, [1, 0, 0, 0])  # kept 1 of 3 -> 0.33, and scored worse
        assert opt._admits(child, [parent]) is False
        assert child.metadata["retention"] == pytest.approx(0.3333, abs=1e-3)

    def test_accepts_preserving_child(self):
        opt = _bare_scope(RETENTION_FLOOR=0.8)
        parent = _rec("p", 0.75, [1, 1, 1, 0])
        child = _rec("c", 1.0, [1, 1, 1, 1])
        assert opt._admits(child, [parent]) is True

    def test_low_retention_is_allowed_when_the_child_improves(self):
        """M1 keeps instances candidates disagree on, so retention is
        structurally low; a scoring gain must override the floor or the
        search stops exploring (measured: 43/44 rejected under a
        retention-only gate)."""
        opt = _bare_scope(RETENTION_FLOOR=0.8)
        parent = _rec("p", 0.50, [1, 1, 0, 0])
        child = _rec("c", 0.75, [0, 0, 1, 1])   # retention 0.0, but scores higher
        assert opt._admits(child, [parent]) is True
        assert child.metadata["retention"] == pytest.approx(0.0)

    def test_equal_score_still_requires_preservation(self):
        """No gain means the edit must at least not destroy anything."""
        opt = _bare_scope(RETENTION_FLOOR=0.8)
        parent = _rec("p", 0.5, [1, 1, 0, 0])
        child = _rec("c", 0.5, [0, 0, 1, 1])    # same score, destroyed both solves
        assert opt._admits(child, [parent]) is False

    def test_judged_against_best_matching_parent(self):
        """A crossover keeps one lineage; it must not be punished for the other."""
        opt = _bare_scope(RETENTION_FLOOR=0.8)
        kept = _rec("p1", 0.5, [1, 1, 0, 0])
        other = _rec("p2", 0.5, [0, 0, 1, 1])
        child = _rec("c", 0.5, [1, 1, 0, 0])  # 100% of p1, 0% of p2
        assert opt._admits(child, [kept, other]) is True

    def test_admits_when_no_parent_data(self):
        opt = _bare_scope(RETENTION_FLOOR=0.8)
        assert opt._admits(_rec("c", 0.5, [1, 0]), []) is True


# ---------------------------------------------------------------- M5 ----

class TestCostObjective:
    def test_cost_counts_prompt_and_induced_output(self):
        opt = _bare_scope(COST_INPUT_WEIGHT=1.0, COST_OUTPUT_WEIGHT=1.0)
        rec = _rec("one two three", 0.5, [1], details=[{"prediction": "a b"}])
        assert opt._cost_of(rec) == pytest.approx(3 + 2)

    def test_cost_handles_missing_details(self):
        opt = _bare_scope(COST_INPUT_WEIGHT=1.0, COST_OUTPUT_WEIGHT=1.0)
        assert opt._cost_of(_rec("one two", 0.5, [1])) == pytest.approx(2.0)

    def test_front_prefers_cheaper_at_equal_accuracy(self):
        opt = _bare_scope(COST_INPUT_WEIGHT=1.0, COST_OUTPUT_WEIGHT=0.0)
        cheap = _rec("short", 0.8, [1], rid="cheap")
        pricey = _rec("a much longer prompt here", 0.8, [1], rid="pricey")
        front = opt._cost_front([cheap, pricey])
        ids = {r.id for r in front}
        assert "cheap" in ids
        assert "pricey" not in ids, "equal accuracy at higher cost is dominated"

    def test_front_keeps_expensive_when_it_is_more_accurate(self):
        opt = _bare_scope(COST_INPUT_WEIGHT=1.0, COST_OUTPUT_WEIGHT=0.0)
        cheap = _rec("short", 0.5, [1], rid="cheap")
        pricey = _rec("a much longer prompt here", 0.9, [1], rid="pricey")
        ids = {r.id for r in opt._cost_front([cheap, pricey])}
        assert ids == {"cheap", "pricey"}, "neither dominates; both on the front"

    def test_front_excludes_strictly_worse_on_both_axes(self):
        opt = _bare_scope(COST_INPUT_WEIGHT=1.0, COST_OUTPUT_WEIGHT=0.0)
        good = _rec("short", 0.9, [1], rid="good")
        bad = _rec("a much longer prompt here", 0.4, [1], rid="bad")
        assert {r.id for r in opt._cost_front([good, bad])} == {"good"}


# ------------------------------------------------------- integration ----

class TestSelection:
    def test_elite_always_survives(self):
        opt = _bare_scope(population_size=3, frontier_pull_prob=0.0)
        recs = [_rec(f"p{i}", s, [1, 0]) for i, s in enumerate([0.9, 0.5, 0.4, 0.3])]
        selected = opt._select_next_population(recs)
        assert selected[0].score == 0.9
        assert len(selected) == 3

    def test_selection_never_exceeds_population_size(self):
        opt = _bare_scope(population_size=2, frontier_pull_prob=0.0)
        recs = [_rec(f"p{i}", 0.5 + i * 0.01, [1, 0]) for i in range(8)]
        assert len(opt._select_next_population(recs)) == 2

    def test_registered_and_importable(self):
        from pof.optimizers import get_optimizer
        assert get_optimizer("scope") is SCOPEOptimizer


class TestAblationSwitches:
    """A flag that silently fails to ablate would invalidate the whole
    isolation experiment, so each is asserted to actually change behaviour."""

    def test_signal_calibration_off_leaves_pool_untouched(self):
        opt = _bare_scope(
            _signal_pool=[{"input": f"i{i}"} for i in range(4)],
            MIN_SIGNAL_POOL=1, use_signal_calibration=False,
        )
        a = _rec("a", 0.5, [1, 1, 0, 0])
        b = _rec("b", 0.25, [1, 0, 1, 0])
        opt._calibrate_signal_pool([a, b])
        assert len(opt._signal_pool) == 4, "M1 off must not drop instances"
        assert a.performance_vector == [1, 1, 0, 0], "vectors must be untouched"
        assert opt._calibrated is True, "still marked done so it is not retried"

    def test_retention_gate_off_admits_destructive_child(self):
        destructive = dict(RETENTION_FLOOR=0.8)
        parent = _rec("p", 0.75, [1, 1, 1, 0])
        child = _rec("c", 0.25, [1, 0, 0, 0])   # retention 0.33, scores worse
        assert _bare_scope(**destructive)._admits(child, [parent]) is False
        off = _bare_scope(use_retention_gate=False, **destructive)
        assert off._admits(child, [parent]) is True, "M3 off must admit"
        assert child.metadata["retention"] == pytest.approx(0.3333, abs=1e-3), \
            "retention is still MEASURED (M2) even when not used to gate (M3)"

    def test_cost_front_off_selects_purely_on_score(self):
        recs = [
            _rec("a very long but accurate prompt here", 0.90, [1, 1], rid="long"),
            _rec("short", 0.50, [1, 0], rid="short"),
            _rec("mid length prompt", 0.70, [1, 1], rid="mid"),
        ]
        off = _bare_scope(population_size=2, frontier_pull_prob=0.0,
                          use_cost_front=False)
        picked = [r.id for r in off._select_next_population(recs)]
        assert picked[0] == "long", "elitism still applies"
        assert "short" not in picked, (
            "with M5 off, the cheap-but-weak prompt must not be pulled in"
        )
