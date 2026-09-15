"""Tests for the Classiq backend's non-execution logic.

The execution path (solve_qubo_qaoa) needs a Classiq account, so it is NOT run here. But the
two pieces that surround execution — converting the QUBO to Pauli terms, and normalising a
sampled result into a histogram — are plain Python and fully testable with stubs. These are
also the pieces most likely to have a bug, so covering them is worthwhile.

On a machine with a Classiq account, add an end-to-end test that asserts the Classiq backend
agrees with the Qiskit backend on a small partition instance.
"""

import numpy as np
import pytest

from qc_core.qubo.core import QUBO
from qc_core.qaoa.classiq_backend import (
    qubo_to_pauli_terms,
    _parsed_counts_to_histogram,
)


def test_pauli_terms_match_ising():
    """The Pauli term list must reproduce the QUBO's Ising energies (up to the offset)."""
    q = QUBO.zeros(3)
    q.add_linear(0, 1.5).add_quadratic(0, 1, 2.0).add_quadratic(1, 2, -1.0)
    terms = qubo_to_pauli_terms(q)
    h_i, J_i, _offset = q.to_ising()

    # evaluate the term list on each basis state as an Ising energy and compare
    for bits in [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]:
        s = [1 - 2 * b for b in bits]  # x=0 -> s=+1, x=1 -> s=-1
        term_energy = 0.0
        for pauli_str, coeff in terms:
            val = 1.0
            for idx, ch in enumerate(pauli_str):
                if ch == "Z":
                    val *= s[idx]
            term_energy += coeff * val
        ising = h_i @ np.array(s) + np.array(s) @ J_i @ np.array(s)
        assert term_energy == pytest.approx(ising, abs=1e-9)


def test_pauli_terms_little_endian_positions():
    """A single linear term on variable 1 must place 'Z' at string index 1."""
    q = QUBO.zeros(3)
    q.add_linear(1, 2.0)
    terms = qubo_to_pauli_terms(q)
    # find the single-Z term; its 'Z' should be at position 1
    z_terms = [t for t, _ in terms if t.count("Z") == 1]
    assert any(t[1] == "Z" and t[0] == "I" and t[2] == "I" for t in z_terms)


class _StubParsedCount:
    """Minimal stand-in for a Classiq parsed_count entry."""
    def __init__(self, state, shots):
        self.state = state
        self.shots = shots


def test_parsed_counts_from_bit_list():
    """A register given as a bit list is joined into a little-endian bitstring."""
    parsed = [
        _StubParsedCount({"x": [1, 0, 1]}, 30),
        _StubParsedCount({"x": [0, 0, 0]}, 10),
    ]
    hist = _parsed_counts_to_histogram(parsed, n=3)
    assert hist == {"101": 30, "000": 10}


def test_parsed_counts_from_integer():
    """A register given as an integer is expanded to little-endian bits of width n."""
    parsed = [_StubParsedCount({"x": 5}, 12)]  # 5 = 101b -> little-endian '101'
    hist = _parsed_counts_to_histogram(parsed, n=3)
    assert hist == {"101": 12}


def test_parsed_counts_accumulates_duplicates():
    parsed = [
        _StubParsedCount({"x": [1, 1]}, 5),
        _StubParsedCount({"x": [1, 1]}, 7),
    ]
    hist = _parsed_counts_to_histogram(parsed, n=2)
    assert hist == {"11": 12}
