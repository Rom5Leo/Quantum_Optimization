"""Tests for the stochastic (uncertainty-aware) partition builders.

Ground-truth strategy: for every QUBO we brute-force its argmin and compare against a
*direct* enumeration of the objective it is supposed to encode. This checks the algebra
(QUBO expansion) independently of the builder.
"""

from itertools import product

import numpy as np
import pytest

from qcoptlib.qubo.core import QUBO
from qcoptlib.qubo.partition import (
    partition_qubo,
    stochastic_partition_qubo,
    gap_variance_qubo,
    split_report,
)


def _spins(bits):
    return np.array([1 - 2 * b for b in bits], dtype=float)


# ------------------------- QUBO composition -------------------------

def test_qubo_add_is_pointwise():
    a = QUBO.zeros(3).add_linear(0, 1.0).add_quadratic(0, 1, 2.0).add_const(1.0)
    b = QUBO.zeros(3).add_linear(1, -3.0).add_quadratic(0, 1, 1.0).add_const(0.5)
    c = a + b
    for bits in product((0, 1), repeat=3):
        assert c.energy(bits) == pytest.approx(a.energy(bits) + b.energy(bits))


def test_qubo_scaled_matches_factor():
    a = QUBO.zeros(2).add_linear(0, 2.0).add_quadratic(0, 1, -1.0).add_const(3.0)
    s = a.scaled(2.5)
    for bits in product((0, 1), repeat=2):
        assert s.energy(bits) == pytest.approx(2.5 * a.energy(bits))


def test_qubo_sum_builtin():
    parts = [QUBO.zeros(2).add_linear(0, 1.0), QUBO.zeros(2).add_linear(1, 1.0)]
    total = sum(parts)
    assert total.energy([1, 1]) == pytest.approx(2.0)


# ------------------------- value-only equals deterministic -------------------------

def test_value_only_matches_partition_qubo():
    means = [4, 5, 6, 7, 8]
    q_det = partition_qubo(means)
    q_sto = stochastic_partition_qubo(means)  # no risk / robustness
    for bits in product((0, 1), repeat=len(means)):
        assert q_sto.energy(bits) == pytest.approx(q_det.energy(bits))


# ------------------------- the "trap": independent variance changes nothing -------------------------

def test_independent_variance_does_not_change_split():
    """Adding independent per-asset variance via the *gap-variance* term must not move the argmin."""
    means = [10, 12, 7, 9, 5, 8]
    variances = [4, 9, 1, 16, 2, 5]  # arbitrary, independent
    base = partition_qubo(means)
    with_var = stochastic_partition_qubo(means, variances=variances, robust_weight=3.0)
    assert with_var.brute_force()[0] == base.brute_force()[0]


# ------------------------- risk balance DOES change the split -------------------------

def test_risk_balance_changes_split_and_balances_risk():
    """A case engineered so value-only leaves risk lopsided; risk balancing must even it out."""
    # Values chosen so many equal-value splits exist; variances concentrated on a few assets.
    means = [10, 10, 10, 10, 10, 10]
    variances = [100, 100, 1, 1, 1, 1]  # two very volatile assets
    val_only = partition_qubo(means).brute_force()[0]
    risk_bal = stochastic_partition_qubo(means, variances=variances, risk_weight=1.0).brute_force()[0]

    r_val = split_report(val_only, means, variances=variances)
    r_bal = split_report(risk_bal, means, variances=variances)
    # both keep value fair
    assert r_bal.value_gap == pytest.approx(0.0, abs=1e-9)
    # risk balancing should not be worse on risk, and should split the two volatile assets
    assert r_bal.risk_gap <= r_val.risk_gap + 1e-9
    # the two volatile assets (0 and 1) end up on opposite siblings
    assert (risk_bal[0] == 0) != (risk_bal[1] == 0)


# ------------------------- gap_variance encodes Var(D) exactly -------------------------

def test_gap_variance_matches_direct_formula():
    rng = np.random.default_rng(0)
    n = 5
    A = rng.normal(size=(n, n))
    C = A @ A.T  # a valid (symmetric PSD) covariance matrix
    q = gap_variance_qubo(C)
    for bits in product((0, 1), repeat=n):
        s = _spins(bits)
        assert q.energy(bits) == pytest.approx(float(s @ C @ s))


def test_correlation_splits_correlated_assets():
    """Two perfectly-correlated assets should be pushed onto opposite siblings by robustness."""
    means = [10, 10, 10, 10]
    # assets 0 and 1 strongly positively correlated; 2,3 independent
    C = np.diag([4.0, 4.0, 4.0, 4.0])
    C[0, 1] = C[1, 0] = 3.9
    q = stochastic_partition_qubo(means, cov=C, robust_weight=1.0)
    best = q.brute_force()[0]
    assert (best[0] == 0) != (best[1] == 0)  # 0 and 1 on opposite sides
