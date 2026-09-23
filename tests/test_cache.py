"""Tests for the QAOA disk cache (no Qiskit needed — a stub solver stands in)."""

import numpy as np
import pytest

from qcoptlib.qubo.core import QUBO
from qcoptlib.quantum.cache import cached_qaoa, qaoa_cache_key


def _qubo(seed=0):
    rng = np.random.default_rng(seed)
    q = QUBO.zeros(4)
    q.h = rng.normal(size=4)
    q.J = np.triu(rng.normal(size=(4, 4)), 1)
    return q


def test_key_is_deterministic_and_sensitive():
    q = _qubo()
    assert qaoa_cache_key(q, seed=0, num_layers=3) == qaoa_cache_key(q, seed=0, num_layers=3)
    assert qaoa_cache_key(q, seed=0) != qaoa_cache_key(q, seed=1)          # param change
    q2 = _qubo(); q2.h = q2.h + 1.0
    assert qaoa_cache_key(q, seed=0) != qaoa_cache_key(q2, seed=0)          # qubo change


def test_cache_computes_once_then_loads(tmp_path):
    calls = {"n": 0}

    def stub(qubo, **kw):
        calls["n"] += 1
        return {"marker": calls["n"], "kw": kw}

    q = _qubo()
    r1 = cached_qaoa(q, cache_dir=str(tmp_path), _solver=stub, seed=0, num_layers=3)
    r2 = cached_qaoa(q, cache_dir=str(tmp_path), _solver=stub, seed=0, num_layers=3)
    assert calls["n"] == 1              # second call loaded from disk
    assert r1 == r2

    # a different parameter recomputes
    cached_qaoa(q, cache_dir=str(tmp_path), _solver=stub, seed=1, num_layers=3)
    assert calls["n"] == 2

    # refresh=True forces recompute even for the same key
    cached_qaoa(q, cache_dir=str(tmp_path), _solver=stub, refresh=True, seed=0, num_layers=3)
    assert calls["n"] == 3


def test_import_does_not_require_qiskit():
    # importing the cache module / package must not pull Qiskit (lazy import inside the function)
    import qcoptlib.quantum  # noqa: F401
    assert hasattr(qcoptlib.quantum, "cached_qaoa")
