"""Offline tests for the detectable-effect bound — no LLM, no dataset."""
import pytest

from pof.evaluation.power import describe_power, minimum_detectable_effect


class TestMinimumDetectableEffect:
    def test_humaneval_test_split_is_about_nine_points(self):
        """n=115 at p=0.85 -- the bound that explains 22 null results."""
        mde = minimum_detectable_effect(0.85, 115)
        assert 0.085 < mde < 0.100

    def test_shrinks_as_n_grows(self):
        assert minimum_detectable_effect(0.85, 500) < minimum_detectable_effect(0.85, 115)

    def test_scales_as_one_over_sqrt_n(self):
        assert minimum_detectable_effect(0.5, 400) == pytest.approx(
            minimum_detectable_effect(0.5, 100) / 2, rel=1e-9)

    def test_holdout_of_fourteen_is_hopeless(self):
        """The pool PACT's stage-2 argmax actually ran on."""
        assert minimum_detectable_effect(0.75, 14) > 0.25

    def test_higher_power_demands_a_larger_effect(self):
        assert (minimum_detectable_effect(0.85, 115, power=0.95)
                > minimum_detectable_effect(0.85, 115, power=0.80))

    def test_tighter_alpha_demands_a_larger_effect(self):
        assert (minimum_detectable_effect(0.85, 115, alpha=0.01)
                > minimum_detectable_effect(0.85, 115, alpha=0.05))

    def test_saturated_score_does_not_report_a_zero_bound(self):
        """p=1.0 has zero binomial variance; the floor keeps this honest."""
        assert minimum_detectable_effect(1.0, 115) > 0.0

    def test_zero_score_does_not_report_a_zero_bound(self):
        assert minimum_detectable_effect(0.0, 115) > 0.0

    def test_empty_sample_is_fully_undetectable(self):
        assert minimum_detectable_effect(0.85, 0) == 1.0


class TestDescribePower:
    def test_reports_n_score_and_bound(self):
        line = describe_power(0.85, 115)
        assert "n=115" in line and "0.8500" in line and "pp at 80% power" in line

    def test_reports_the_bound_in_items(self):
        assert "items" in describe_power(0.85, 115)
