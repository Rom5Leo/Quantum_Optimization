"""Classiq QAOA runner.

Solves a :class:`~qcoptlib.qubo.core.QUBO` with QAOA on the Classiq simulator, following the
verified pattern from the antenna notebooks: build the parametric ``main`` circuit, drive it
from Python with ``ExecutionSession`` + SciPy (NOT the IDE Execute button), then decode the
sampled result with the shared read-out helpers.

**Requires a Classiq account** (``pip install classiq``; ``authenticate()`` once). Because
execution needs that account, this module isolates the one execution call in
:func:`solve_qubo_qaoa` and keeps everything testable (Pauli assembly, decode) in plain
functions that are unit-tested without a backend. On small instances it should agree with the
Qiskit backend, which is the reference implementation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from qcoptlib.qubo.core import QUBO
from qcoptlib.quantum.common import QAOAResult, adiabatic_init, best_bits_from_counts


def qubo_to_pauli_terms(qubo: QUBO) -> list[tuple[str, float]]:
    """Convert a QUBO to Ising Pauli terms as ``(pauli_string, coefficient)`` tuples.

    Backend-neutral (plain strings, no Classiq import), so it is unit-testable without an
    account. ``pauli_string`` is length ``n`` over {'I','Z'} in little-endian order
    (character 0 = variable 0). Built from :meth:`QUBO.to_ising`; the constant offset is
    dropped (does not affect the argmin).
    """
    h_i, J_i, _offset = qubo.to_ising()
    n = qubo.n
    terms: list[tuple[str, float]] = []

    for i in range(n):
        if h_i[i] != 0.0:
            label = ["I"] * n
            label[i] = "Z"
            terms.append(("".join(label), float(h_i[i])))
    for i in range(n):
        for j in range(i + 1, n):
            if J_i[i, j] != 0.0:
                label = ["I"] * n
                label[i] = "Z"
                label[j] = "Z"
                terms.append(("".join(label), float(J_i[i, j])))
    return terms


def _parsed_counts_to_histogram(parsed_counts, n: int) -> dict[str, int]:
    """Normalise Classiq ``parsed_counts`` to a {little-endian bitstring: count} dict.

    Classiq returns sampled states as structured objects; this pulls each into a plain
    bitstring keyed the way the shared decoder expects (character 0 = variable 0). Kept
    separate so it can be unit-tested with a stub in place of a real Classiq result.
    """
    hist: dict[str, int] = {}
    for pc in parsed_counts:
        # pc.state maps the output register to its measured integer/bit list; we expect a
        # register named "x" holding the n solution qubits. Adapt the key if your model
        # names it differently.
        raw = pc.state["x"]
        if isinstance(raw, (list, tuple, np.ndarray)):
            bits = "".join(str(int(b)) for b in raw)
        else:  # an integer -> little-endian bits, width n
            bits = "".join(str((int(raw) >> i) & 1) for i in range(n))
        hist[bits] = hist.get(bits, 0) + int(pc.shots)
    return hist


def solve_qubo_qaoa(
    qubo: QUBO,
    num_layers: int = 3,
    maxiter: int = 100,
    shots: int = 2048,
) -> QAOAResult:
    """Solve a QUBO with QAOA on the Classiq simulator (Python-driven ExecutionSession).

    Requires a Classiq account. Mirrors the antenna-notebook flow:
    build parametric ``main`` -> synthesize -> ExecutionSession -> SciPy tunes the angles via
    ``estimate_cost`` -> sample the tuned circuit -> decode the best bitstring.

    The Classiq imports are inside the function so the rest of this module (the tested Pauli
    and decode helpers) imports without a Classiq install.
    """
    from classiq import (
        qfunc, QArray, QNum, Output, CArray, CReal,
        allocate, hadamard_transform, apply_to_all, RX, repeat, synthesize,
    )
    from classiq import Pauli, PauliTerm  # noqa: F401  (available for a hamiltonian route)
    from classiq.execution import ExecutionSession

    n = qubo.n
    pauli_terms = qubo_to_pauli_terms(qubo)

    # Build the cost as a Classiq Hamiltonian phase layer. We use the SparsePauliOp-style
    # term list; Classiq's suzuki/ hamiltonian_evolution consumes PauliTerm objects.
    def _to_classiq_hamiltonian():
        from classiq import Pauli, PauliTerm
        char_to_pauli = {"I": Pauli.I, "Z": Pauli.Z}
        return [
            PauliTerm(pauli=[char_to_pauli[c] for c in term], coefficient=coeff)
            for term, coeff in pauli_terms
        ]

    hamiltonian = _to_classiq_hamiltonian()

    from classiq import suzuki_trotter

    @qfunc
    def main(params: CArray[CReal, 2 * num_layers], x: Output[QArray]) -> None:
        allocate(n, x)
        hadamard_transform(x)
        gammas = params[0:num_layers]
        betas = params[num_layers : 2 * num_layers]
        repeat(
            num_layers,
            lambda i: [
                suzuki_trotter(hamiltonian, evolution_coefficient=gammas[i],
                               order=1, repetitions=1, qbv=x),
                apply_to_all(lambda q: RX(betas[i], q), x),
            ],
        )

    qprog = synthesize(main)
    es = ExecutionSession(qprog)

    def cost_of_state(state) -> float:
        raw = state["x"]
        if isinstance(raw, (list, tuple, np.ndarray)):
            bits = [int(b) for b in raw]
        else:
            bits = [(int(raw) >> i) & 1 for i in range(n)]
        return qubo.energy(bits)

    history: list[float] = []

    def objective(p):
        val = es.estimate_cost(cost_func=cost_of_state, parameters={"params": p.tolist()})
        history.append(float(val))
        return val

    res = minimize(objective, adiabatic_init(num_layers), method="COBYLA",
                   options={"maxiter": maxiter})

    final = es.sample({"params": res.x.tolist()})
    counts = _parsed_counts_to_histogram(final.parsed_counts, n)
    es.close()

    best_bits, best_energy = best_bits_from_counts(counts, qubo)
    return QAOAResult(
        best_bits=best_bits,
        best_energy=best_energy,
        counts=counts,
        optimal_params=res.x,
        history=history,
    )
