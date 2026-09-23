"""On-disk cache for QAOA solves.

QAOA on a simulator is slow, and a notebook re-runs every cell on "Run All" — so an unchanged
solve is recomputed each time for no reason. :func:`cached_qaoa` wraps
:func:`~qcoptlib.quantum.qiskit_backend.solve_qubo_qaoa`, keying a pickled result on the QUBO's
coefficients *and* every solver argument. Re-running with the same problem and settings loads the
saved :class:`~qcoptlib.quantum.common.QAOAResult` instantly; changing the QUBO, ``num_layers``,
``seed`` (etc.) misses the cache and recomputes only that call. Pass ``refresh=True`` (or delete
the cache directory) to force a fresh solve.

The cache directory is just a folder of ``.pkl`` files — add it to ``.gitignore``.
"""

from __future__ import annotations

import hashlib
import os
import pickle

import numpy as np

from qcoptlib.qubo.core import QUBO


def qaoa_cache_key(qubo: QUBO, **params) -> str:
    """A short content hash of the QUBO plus the solver parameters.

    Two calls collide (reuse a cached result) iff the QUBO coefficients and every solver
    argument match, so the cache never returns a result computed for a different problem or a
    different setting.
    """
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(qubo.h, dtype=float).tobytes())
    h.update(np.ascontiguousarray(qubo.J, dtype=float).tobytes())
    h.update(repr((round(float(qubo.const), 9), sorted(params.items()))).encode())
    return h.hexdigest()[:16]


def cached_qaoa(qubo: QUBO, *, cache_dir: str = "qaoa_cache", refresh: bool = False,
                _solver=None, **kwargs):
    """Run (or load) :func:`solve_qubo_qaoa`, caching the result to ``cache_dir``.

    Parameters
    ----------
    qubo      : the problem.
    cache_dir : folder for the pickled results (created if missing). Add it to ``.gitignore``.
    refresh   : if True, ignore any cached result and recompute (then overwrite the cache).
    _solver   : test hook; defaults to the real :func:`solve_qubo_qaoa` (imported lazily so this
                module imports without Qiskit installed).
    **kwargs  : forwarded to the solver (``num_layers``, ``maxiter``, ``shots``, ``restarts``,
                ``seed``, ``tol``) and folded into the cache key.

    Returns
    -------
    A :class:`QAOAResult` (freshly computed or loaded from disk).
    """
    solver = _solver
    if solver is None:
        from qcoptlib.quantum.qiskit_backend import solve_qubo_qaoa  # lazy: no Qiskit at import
        solver = solve_qubo_qaoa

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"qaoa_{qaoa_cache_key(qubo, **kwargs)}.pkl")

    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    result = solver(qubo, **kwargs)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result
