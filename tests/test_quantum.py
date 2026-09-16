"""Tests for qcoptlib.quantum.

- Shared helpers (adiabatic_init, best_bits_from_counts) — fast, deterministic.
- Qiskit backend — end-to-end on a small partition problem, checked against brute force.

The Qiskit end-to-end test is the real proof: QAOA on Aer must recover the optimal split
of a small instance whose answer we know from the QUBO's own brute_force.
"""

import numpy as np
import pytest

from qcoptlib.qubo.core import QUBO
from qcoptlib.qubo.partition import partition_qubo, split_from_bits
from qcoptlib.quantum.common import adiabatic_init, best_bits_from_counts, QAOAResult


# ----------------------------- shared helpers -----------------------------

def test_adiabatic_init_shape_and_endpoints():
    p = adiabatic_init(4)
    assert p.shape == (8,)
    gammas, betas = p[0::2], p[1::2]
    assert gammas[0] == pytest.approx(0.0) and gammas[-1] == pytest.approx(1.0)
    assert betas[0] == pytest.approx(1.0) and betas[-1] == pytest.approx(0.0)


def test_best_bits_from_counts_picks_lowest_energy():
    # cost = -x0 - x1 + 3 x0 x1  -> min at a single 1 (energy -1)
    q = QUBO.zeros(2)
    q.add_linear(0, -1.0).add_linear(1, -1.0).add_quadratic(0, 1, 3.0)
    counts = {"00": 10, "10": 5, "01": 5, "11": 50}  # 11 is most frequent but NOT best
    bits, energy = best_bits_from_counts(counts, q)
    assert energy == pytest.approx(-1.0)
    assert sum(bits) == 1  # a single 1, not the frequent-but-worse 11


# ----------------------------- Qiskit backend (end-to-end) -----------------------------

def test_qiskit_solves_small_partition():
    """QAOA on Aer recovers the optimal split of a small partition instance."""
    from qcoptlib.quantum.qiskit_backend import solve_qubo_qaoa

    values = [3, 4, 5, 6]           # small enough for reliable QAOA + brute-force check
    q = partition_qubo(values)
    _, brute_energy = q.brute_force()

    res = solve_qubo_qaoa(q, num_layers=3, maxiter=150, shots=4096, restarts=2, seed=0)

    assert isinstance(res, QAOAResult)
    # QAOA's best sampled bitstring should hit the true optimum on an instance this small
    assert res.best_energy == pytest.approx(brute_energy, abs=1e-6)
    # and it should correspond to a genuinely good (here, minimal-gap) split
    _, _, gap = split_from_bits(res.best_bits, values)
    assert gap == pytest.approx(0.0)   # [3,4,5,6] splits 3+6 = 4+5 = 9


def test_qiskit_sparse_pauli_matches_ising_energy():
    """The SparsePauliOp cost operator must reproduce the QUBO's Ising energies."""
    from qcoptlib.quantum.qiskit_backend import qubo_to_sparse_pauli
    from qiskit.quantum_info import Statevector, SparsePauliOp

    q = QUBO.zeros(3)
    q.add_linear(0, 1.5).add_quadratic(0, 1, 2.0).add_quadratic(1, 2, -1.0)
    op = qubo_to_sparse_pauli(q)
    h_i, J_i, offset = q.to_ising()

    # for each computational basis state, <op> should equal the Ising energy minus offset
    for bits in [(0,0,0), (1,0,1), (0,1,1), (1,1,1)]:
        # build the basis statevector (Qiskit little-endian: qubit 0 is rightmost)
        idx = sum(b << i for i, b in enumerate(bits))
        sv = Statevector.from_int(idx, dims=2**3)
        expval = sv.expectation_value(op).real
        s = np.array([1 - 2*b for b in bits], dtype=float)
        ising_no_offset = h_i @ s + s @ J_i @ s
        assert expval == pytest.approx(ising_no_offset, abs=1e-6)
