"""Check context dependence and conservative small-sample damage bounds."""

import pytest

from dolos_audit.stats import binomial_upper_bound, stratified_paired_bootstrap


def test_stratification_preserves_fixed_design_weights_and_context_pairs():
    a = {"x": [1] * 100, "y": [0], "z": [0]}
    b = {"x": [1], "y": [0] * 100, "z": [0]}
    strata = {"x": "one", "y": "two", "z": "two"}
    first = stratified_paired_bootstrap(a, strata, draws=100)
    assert first == stratified_paired_bootstrap(b, strata, draws=100)
    assert first["estimate"] == pytest.approx(1 / 3)
    assert first["interval"] == [1 / 3, 1 / 3]


def test_zero_observed_damage_does_not_imply_zero_population_risk():
    assert binomial_upper_bound(0, 100, 0.05 / 6) == pytest.approx(1 - (0.05 / 6) ** (1 / 100))
    assert 0 < binomial_upper_bound(0, 100) < 0.05
    assert binomial_upper_bound(1, 100) > binomial_upper_bound(0, 100)
    assert binomial_upper_bound(100, 100) == 1
