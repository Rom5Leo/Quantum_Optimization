"""Qiskit QAOA runner.

Solves a :class:`~qcoptlib.qubo.core.QUBO` with QAOA on the Aer simulator, following the
pattern from the inheritance-QAOA notebook: convert the QUBO to an Ising ``SparsePauliOp``,
build a ``qaoa_ansatz``, transpile for Aer, optimise the angles with SciPy (EstimatorV2 in the
loop), then sample the tuned circuit (SamplerV2) and read out the best bitstring.

This backend is fully runnable locally (no account needed), so it doubles as the reference
implementation the Classiq backend is checked against on small instances.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler

from qcoptlib.qubo.core import QUBO
from qcoptlib.quantum.common import QAOAResult, adiabatic_init, best_bits_from_counts


def qubo_to_sparse_pauli(qubo: QUBO) -> SparsePauliOp:
    """Convert a QUBO to the Ising cost operator as a Qiskit ``SparsePauliOp``.

    Uses :meth:`QUBO.to_ising` (x = (1 - s)/2) to get single-Z (``h_i``) and ZZ (``J_i``)
    coefficients, then builds the corresponding Pauli terms. The constant offset is dropped
    (it shifts energies but not the argmin, and SparsePauliOp expectation handles the rest).
    """
    h_i, J_i, _offset = qubo.to_ising()
    n = qubo.n
    terms: list[tuple[str, float]] = []

    for i in range(n):
        if h_i[i] != 0.0:
            label = ["I"] * n
            label[i] = "Z"
            terms.append(("".join(reversed(label)), float(h_i[i])))
    for i in range(n):
        for j in range(i + 1, n):
            if J_i[i, j] != 0.0:
                label = ["I"] * n
                label[i] = "Z"
                label[j] = "Z"
                terms.append(("".join(reversed(label)), float(J_i[i, j])))

    if not terms:  # degenerate all-zero cost; give SparsePauliOp something valid
        terms.append(("I" * n, 0.0))
    return SparsePauliOp.from_list(terms)


def solve_qubo_qaoa(
    qubo: QUBO,
    num_layers: int = 3,
    maxiter: int = 200,
    shots: int = 4096,
    restarts: int = 1,
    seed: int | None = None,
) -> QAOAResult:
    """Solve a QUBO with QAOA on the Aer simulator.

    Parameters
    ----------
    qubo       : the problem.
    num_layers : QAOA depth p.
    maxiter    : COBYLA iterations per restart.
    shots      : samples when reading out the tuned circuit.
    restarts   : random restarts; the best (lowest optimised cost) is kept. The first
                 restart uses the adiabatic init, the rest random, so restarts>=1 never
                 does worse than the principled start.
    seed       : RNG seed for reproducible restarts.

    Returns
    -------
    QAOAResult with the best sampled bitstring (lowest QUBO energy), its energy, the full
    counts, the tuned angles, and the optimiser history of the winning restart.
    """
    rng = np.random.default_rng(seed)

    cost_op = qubo_to_sparse_pauli(qubo)
    ansatz = qaoa_ansatz(cost_operator=cost_op, reps=num_layers)

    # transpile to basis gates so Aer can run it (raw qaoa_ansatz has a high-level instruction)
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    ansatz_isa = pm.run(ansatz)
    cost_isa = cost_op.apply_layout(ansatz_isa.layout)

    estimator = Estimator()

    def cost_fn(params, history):
        pub = (ansatz_isa, [cost_isa], [params])
        value = float(estimator.run([pub]).result()[0].data.evs[0])
        history.append(value)
        return value

    best = None
    for r in range(max(restarts, 1)):
        x0 = adiabatic_init(num_layers) if r == 0 else rng.uniform(0, np.pi, 2 * num_layers)
        history: list[float] = []
        res = minimize(cost_fn, x0, args=(history,), method="COBYLA",
                       options={"maxiter": maxiter})
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x, history)

    _best_cost, best_params, best_history = best

    # read out: sample the tuned circuit
    final = ansatz_isa.assign_parameters(best_params)
    final.measure_all()
    raw_counts = Sampler().run([final], shots=shots).result()[0].data.meas.get_counts()

    # normalise bitstrings to little-endian (bit 0 = variable 0) for the shared decoder
    counts = {bitstring[::-1]: count for bitstring, count in raw_counts.items()}

    best_bits, best_energy = best_bits_from_counts(counts, qubo)
    return QAOAResult(
        best_bits=best_bits,
        best_energy=best_energy,
        counts=counts,
        optimal_params=best_params,
        history=best_history,
    )
