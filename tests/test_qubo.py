"""Tests for qcoptlib.qubo — the QUBO container and the partition builder.

Ground-truth checks:
- the container's energy/brute_force/to_ising against hand-computable cases,
- the partition builder against the inheritance-notebook result (the values that split
  perfectly into $57M / $57M).
"""

import numpy as np
import pytest

from qcoptlib.qubo.core import QUBO
from qcoptlib.qubo.partition import partition_qubo, split_from_bits


# ----------------------------- container -----------------------------

def test_zeros_and_add_terms():
    q = QUBO.zeros(3)
    q.add_linear(0, 2.0).add_quadratic(0, 1, -3.0).add_const(1.0)
    # C(x) = 2 x0 - 3 x0 x1 + 1
    assert q.energy([0, 0, 0]) == pytest.approx(1.0)
    assert q.energy([1, 0, 0]) == pytest.approx(3.0)      # 2 + 1
    assert q.energy([1, 1, 0]) == pytest.approx(0.0)      # 2 - 3 + 1


def test_quadratic_self_term_is_linear():
    """x^2 == x for binaries, so add_quadratic(v, v) acts on the diagonal (linear)."""
    q = QUBO.zeros(1)
    q.add_quadratic(0, 0, 5.0)
    assert q.energy([1]) == pytest.approx(5.0)
    assert q.energy([0]) == pytest.approx(0.0)


def test_quadratic_is_order_independent():
    q1 = QUBO.zeros(2); q1.add_quadratic(0, 1, 4.0)
    q2 = QUBO.zeros(2); q2.add_quadratic(1, 0, 4.0)
    for bits in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert q1.energy(bits) == pytest.approx(q2.energy(bits))


def test_brute_force_finds_minimum():
    q = QUBO.zeros(2)
    q.add_linear(0, -1.0).add_linear(1, -1.0).add_quadratic(0, 1, 3.0)
    # minimum is a single 1 (energy -1), not both (−1−1+3 = 1) nor neither (0)
    best_x, best_e = q.brute_force()
    assert best_e == pytest.approx(-1.0)
    assert sum(best_x) == 1


def test_to_ising_preserves_energy_ordering():
    """Ising energy on spins must match QUBO energy on the corresponding bits."""
    q = QUBO.zeros(3)
    q.add_linear(0, 1.5).add_linear(2, -2.0).add_quadratic(0, 1, 3.0).add_quadratic(1, 2, -1.0)
    h_i, J_i, offset = q.to_ising()
    for bits in [(0,0,0), (1,0,1), (0,1,1), (1,1,1), (1,0,0)]:
        s = np.array([1 - 2*b for b in bits], dtype=float)   # x=0 -> s=+1, x=1 -> s=-1
        ising_e = h_i @ s + s @ J_i @ s + offset
        assert ising_e == pytest.approx(q.energy(bits))


# ----------------------------- partition -----------------------------

def test_partition_two_equal_values():
    """[3, 3] splits perfectly: one each, gap 0."""
    q = partition_qubo([3, 3])
    best_x, _ = q.brute_force()
    _, _, gap = split_from_bits(best_x, [3, 3])
    assert gap == pytest.approx(0.0)


def test_partition_matches_inheritance_notebook():
    """The inheritance values split perfectly into two piles of $57M (gap 0).

    From the inheritance-QAOA notebook: these 13 values partition exactly.
    """
    values = [1, 5, 17, 1.5, 4.3, 13.8, 2, 2.3, 8.9, 21.9, 4.5, 7.1, 24.7]
    q = partition_qubo(values)
    best_x, _ = q.brute_force()
    group_a, group_b, gap = split_from_bits(best_x, values)
    assert gap == pytest.approx(0.0, abs=1e-9)
    assert sum(group_a) == pytest.approx(57.0)
    assert sum(group_b) == pytest.approx(57.0)


def test_partition_qubo_argmin_equals_min_gap():
    """The QUBO's argmin must be the same assignment that minimises the actual gap."""
    values = [4, 5, 6, 7, 8]
    q = partition_qubo(values)
    best_x, _ = q.brute_force()
    _, _, qubo_gap = split_from_bits(best_x, values)

    # independent brute force over gaps
    from itertools import product
    best_gap = min(
        abs(sum(values[i] for i in range(len(values)) if a[i] == 0)
            - sum(values[i] for i in range(len(values)) if a[i] == 1))
        for a in product((0, 1), repeat=len(values))
    )
    assert qubo_gap == pytest.approx(best_gap)
