"""Tests for QUBO constraint penalties (equality / one-hot / slack sizing)."""

from itertools import product

import numpy as np
import pytest

from qcoptlib.qubo import QUBO, equality_penalty, onehot_penalty, inequality_padding


def test_equality_penalty_matches_formula():
    n = 4
    coeffs = {0: 1.0, 1: 2.0, 3: 1.0}
    target, P = 2.0, 3.0
    q = equality_penalty(n, coeffs, target, weight=P)
    a = np.array([1.0, 2.0, 0.0, 1.0])
    for bits in product((0, 1), repeat=n):
        x = np.array(bits, float)
        expected = P * (a @ x - target) ** 2
        assert q.energy(bits) == pytest.approx(expected, abs=1e-9)


def test_equality_penalty_zero_iff_feasible():
    n = 3
    q = equality_penalty(n, [1, 1, 1], target=2, weight=5.0)
    for bits in product((0, 1), repeat=n):
        e = q.energy(bits)
        if sum(bits) == 2:
            assert e == pytest.approx(0.0, abs=1e-9)
        else:
            assert e > 0.0


def test_onehot_penalty_selects_exactly_one():
    n = 5
    q = onehot_penalty(n, indices=[1, 2, 3], weight=1.0)
    # exactly one of {1,2,3} set (others free) -> zero penalty
    assert q.energy([0, 1, 0, 0, 0]) == pytest.approx(0.0)
    assert q.energy([1, 1, 0, 0, 1]) == pytest.approx(0.0)   # var 0,4 not in the group
    assert q.energy([0, 0, 0, 0, 0]) > 0.0                    # none set
    assert q.energy([0, 1, 1, 0, 0]) > 0.0                    # two set


def test_penalty_composes_with_objective_and_moves_optimum():
    # objective: reward selecting high-value items (minimize -value·x); constraint: pick exactly 2
    n = 4
    values = [5, 4, 3, 1]
    obj = QUBO.zeros(n)
    for i, v in enumerate(values):
        obj.add_linear(i, -float(v))
    unconstrained = obj.brute_force()[0]
    assert sum(unconstrained) == n                      # unconstrained picks everything

    constrained = (obj + onehot_penalty_k(n, k=2, weight=100.0)).brute_force()[0]
    assert sum(constrained) == 2                          # penalty forces exactly two
    assert list(constrained) == [1, 1, 0, 0]             # the two highest-value items


def onehot_penalty_k(n, k, weight):
    return equality_penalty(n, [1] * n, target=k, weight=weight)


def test_inequality_padding_sizes_slack():
    assert inequality_padding(0) == 0
    assert inequality_padding(1) == 1
    assert inequality_padding(3) == 2
    assert inequality_padding(4) == 3
