"""Grover search & amplitude amplification — exact statevector simulation + the geometry.

Grover finds a marked item among ``N = 2**n`` in ``O(√(N/M))`` oracle calls (``M`` marked) — a
quadratic speedup over classical ``O(N/M)``. The circuit is: uniform superposition, then ``r``
repetitions of *oracle* (phase-flip the marked states) + *diffuser* (reflection about the mean).
Each repetition rotates the state by ``2θ`` toward the marked subspace, ``sinθ = √(M/N)``.

This module simulates Grover **exactly on the amplitude vector with NumPy** (no quantum backend),
which is transparent and fast for the small instances used to *learn* the algorithm and to run
Grover Adaptive Search (GAS) as an optimizer. The same operators map directly onto a Qiskit circuit
for real hardware (shown in the learning notebook). Two groups of functions:

- **Geometry (exact numbers):** :func:`grover_angle`, :func:`grover_optimal_iterations`,
  :func:`grover_success_probability`.
- **Simulation:** :func:`uniform_state`, :func:`apply_oracle`, :func:`apply_diffuser`,
  :func:`grover_search`, and :func:`grover_adaptive_search` (minimise a cost).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Geometry — exact, backend-free.
# --------------------------------------------------------------------------- #

def grover_angle(n_qubits: int, num_solutions: int = 1) -> float:
    """The rotation half-angle θ with ``sinθ = √(M/N)`` (M marked of N = 2**n)."""
    if num_solutions <= 0:
        raise ValueError("num_solutions must be >= 1")
    N = 2 ** n_qubits
    if num_solutions > N:
        raise ValueError("num_solutions cannot exceed N = 2**n_qubits")
    return float(np.arcsin(np.sqrt(num_solutions / N)))


def grover_optimal_iterations(n_qubits: int, num_solutions: int = 1) -> int:
    """Optimal number of Grover iterations, ``round(π/(4θ) − 1/2)`` (≈ (π/4)·√(N/M)).

    Iterating *past* this overshoots and the success probability falls again.
    """
    theta = grover_angle(n_qubits, num_solutions)
    return int(max(0, round(np.pi / (4.0 * theta) - 0.5)))


def grover_success_probability(n_qubits: int, num_solutions: int, iterations: int) -> float:
    """Probability of measuring a marked state after ``iterations`` steps: ``sin²((2r+1)θ)``."""
    theta = grover_angle(n_qubits, num_solutions)
    return float(np.sin((2 * iterations + 1) * theta) ** 2)


# --------------------------------------------------------------------------- #
# Exact statevector simulation.
# --------------------------------------------------------------------------- #

def uniform_state(n_qubits: int) -> np.ndarray:
    """The equal superposition over all ``2**n`` basis states (real amplitudes)."""
    N = 2 ** n_qubits
    return np.full(N, 1.0 / np.sqrt(N))


def apply_oracle(amplitudes: np.ndarray, marked) -> np.ndarray:
    """Phase oracle: flip the sign (phase) of the marked basis-state amplitudes."""
    amp = np.asarray(amplitudes, dtype=float).copy()
    idx = list(marked)
    amp[idx] *= -1.0
    return amp


def apply_diffuser(amplitudes: np.ndarray) -> np.ndarray:
    """Diffuser: reflect every amplitude about the mean ("inversion about the average")."""
    amp = np.asarray(amplitudes, dtype=float)
    return 2.0 * amp.mean() - amp


def grover_search(n_qubits: int, marked, iterations: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run Grover from the uniform state for ``iterations`` steps.

    Returns ``(final_amplitudes, history)`` where ``history`` is the amplitude vector after each
    step (index 0 is the initial uniform state), for visualising the amplification.
    """
    amp = uniform_state(n_qubits)
    history = [amp.copy()]
    for _ in range(int(iterations)):
        amp = apply_diffuser(apply_oracle(amp, marked))
        history.append(amp.copy())
    return amp, history


@dataclass
class GASResult:
    """Outcome of :func:`grover_adaptive_search`.

    best_index / best_cost : the minimiser found and its cost.
    thresholds : the cost threshold at the start of each round.
    best_per_round : running best cost after each round (convergence curve).
    n_marked_per_round : how many states were below threshold each round (the shrinking set).
    """
    best_index: int
    best_cost: float
    thresholds: list = field(default_factory=list)
    best_per_round: list = field(default_factory=list)
    n_marked_per_round: list = field(default_factory=list)


def grover_adaptive_search(costs, rounds: int = 12, seed: int | None = None) -> GASResult:
    """Minimise ``costs`` (a length-``2**n`` array of per-state costs) with Grover Adaptive Search.

    Each round marks the states cheaper than the current threshold, runs Grover for the optimal
    number of iterations, samples the amplified distribution, and lowers the threshold to the best
    cost seen — the quantum analogue of Dürr–Høyer minimum finding. Exact statevector simulation.
    """
    costs = np.asarray(costs, dtype=float)
    N = len(costs)
    n = int(round(np.log2(N)))
    if 2 ** n != N:
        raise ValueError("costs length must be a power of two")
    rng = np.random.default_rng(seed)

    cur = int(rng.integers(N))                       # random starting sample
    best_i, best_c = cur, float(costs[cur])
    res = GASResult(best_index=best_i, best_cost=best_c)

    for _ in range(rounds):
        threshold = best_c
        marked = np.flatnonzero(costs < threshold)
        res.thresholds.append(threshold)
        res.n_marked_per_round.append(int(marked.size))
        if marked.size == 0:                         # nothing better exists -> optimum reached
            res.best_per_round.append(best_c)
            break
        r = max(1, grover_optimal_iterations(n, marked.size))
        amp, _ = grover_search(n, marked, r)
        probs = amp ** 2
        probs = probs / probs.sum()
        sample = int(rng.choice(N, p=probs))         # measure the amplified state
        if costs[sample] < best_c:
            best_i, best_c = sample, float(costs[sample])
        res.best_per_round.append(best_c)

    res.best_index, res.best_cost = best_i, best_c
    return res
