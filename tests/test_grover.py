"""Tests for Grover geometry and the exact statevector simulation / GAS optimizer."""

import numpy as np
import pytest

from qcoptlib.quantum.grover import (
    grover_angle, grover_optimal_iterations, grover_success_probability,
    uniform_state, apply_oracle, apply_diffuser, grover_search, grover_adaptive_search,
)


# ---- geometry ----

def test_angle_known_values():
    assert grover_angle(2, 1) == pytest.approx(np.pi / 6)         # N=4,M=1 -> sinθ=1/2
    assert grover_optimal_iterations(2, 1) == 1
    assert grover_success_probability(2, 1, 1) == pytest.approx(1.0, abs=1e-12)


def test_optimal_iterations_scale_as_sqrt_N():
    assert grover_optimal_iterations(4, 1) == 3
    assert grover_optimal_iterations(6, 1) == 6
    assert grover_optimal_iterations(8, 1) == 12


def test_more_solutions_need_fewer_iterations():
    assert grover_optimal_iterations(6, 4) < grover_optimal_iterations(6, 1)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        grover_angle(3, 0)
    with pytest.raises(ValueError):
        grover_angle(2, 5)


# ---- exact simulation ----

def test_operators_preserve_norm():
    amp = uniform_state(3)
    assert np.isclose(np.sum(amp ** 2), 1.0)
    amp = apply_oracle(amp, [5])
    assert np.isclose(np.sum(amp ** 2), 1.0)      # phase flip preserves norm
    amp = apply_diffuser(amp)
    assert np.isclose(np.sum(amp ** 2), 1.0)      # reflection preserves norm


def test_grover_finds_single_marked_with_certainty():
    # N=4, M=1: one iteration -> probability 1 on the marked index
    amp, hist = grover_search(2, [2], grover_optimal_iterations(2, 1))
    probs = amp ** 2
    assert probs[2] == pytest.approx(1.0, abs=1e-9)
    assert len(hist) == 2                          # initial + 1 step


def test_grover_amplifies_marked_and_matches_theory():
    n, marked = 6, [40]
    r = grover_optimal_iterations(n, 1)
    amp, _ = grover_search(n, marked, r)
    p_marked = float((amp ** 2)[marked[0]])
    assert p_marked > 0.99
    assert p_marked == pytest.approx(grover_success_probability(n, 1, r), abs=1e-9)


def test_overshoot_reduces_success():
    n = 6
    r = grover_optimal_iterations(n, 1)
    p_opt = (grover_search(n, [1], r)[0] ** 2)[1]
    p_over = (grover_search(n, [1], 2 * r + 1)[0] ** 2)[1]
    assert p_over < p_opt                          # more iterations is worse past the optimum


# ---- Grover Adaptive Search as an optimizer ----

def test_gas_finds_global_minimum():
    rng = np.random.default_rng(0)
    costs = rng.normal(size=2 ** 6)               # 64 states, unique-ish minimum
    true_min_i = int(np.argmin(costs))
    res = grover_adaptive_search(costs, rounds=20, seed=1)
    assert res.best_index == true_min_i
    assert res.best_cost == pytest.approx(costs.min())
    # best-per-round is monotone non-increasing (a threshold search never gets worse)
    assert all(b2 <= b1 + 1e-12 for b1, b2 in zip(res.best_per_round, res.best_per_round[1:]))


def test_gas_requires_power_of_two():
    with pytest.raises(ValueError):
        grover_adaptive_search(np.zeros(6))
