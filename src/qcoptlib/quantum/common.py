"""Backend-agnostic QAOA helpers shared by every runner.

The parts of a QAOA solve that don't depend on Qiskit or Classiq: how to initialise the
variational angles, how to hold a result, and how to pick the best sampled bitstring. Both
:mod:`optlib.quantum.qiskit_backend` and :mod:`optlib.quantum.classiq_backend` build on this, so
the schedule and read-out logic is written once and shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from qcoptlib.qubo.core import QUBO


def adiabatic_init(num_layers: int) -> np.ndarray:
    """Adiabatic-inspired initial QAOA angles: gammas 0->1, betas 1->0, interleaved.

    Mirrors the annealing schedule (start mixer-dominated, end cost-dominated), which is a
    better starting point for the classical optimiser than random angles. Returns a flat
    array of length ``2 * num_layers`` ordered ``[gamma_0, beta_0, gamma_1, beta_1, ...]``.
    """
    gammas = np.linspace(0.0, 1.0, num_layers)
    betas = np.linspace(1.0, 0.0, num_layers)
    out = np.empty(2 * num_layers)
    out[0::2] = gammas
    out[1::2] = betas
    return out


@dataclass
class QAOAResult:
    """The outcome of a QAOA solve.

    Attributes
    ----------
    best_bits   : the lowest-energy sampled bitstring (tuple of 0/1).
    best_energy : its QUBO energy.
    counts      : the full sampled histogram {bitstring: count}.
    optimal_params : the tuned variational angles.
    history     : optimiser cost per iteration (for a convergence plot), if recorded.
    """

    best_bits: tuple[int, ...]
    best_energy: float
    counts: Mapping[str, int]
    optimal_params: np.ndarray
    history: list[float]


def best_bits_from_counts(counts: Mapping[str, int], qubo: QUBO) -> tuple[tuple[int, ...], float]:
    """Pick the sampled bitstring with the lowest QUBO energy.

    QAOA output is a histogram; the most *frequent* string isn't always the best, so we
    score every sampled string by the real QUBO energy and return the minimum. This also
    makes the read-out robust to weak QAOA concentration (the two-stage-hybrid lesson).

    ``counts`` keys are bitstrings; bit ordering is assumed little-endian (bit 0 = variable
    0), matching how the runners normalise their samples before calling this.
    """
    best_bits, best_energy = None, float("inf")
    for bitstring, _count in counts.items():
        bits = tuple(int(c) for c in bitstring)
        e = qubo.energy(bits)
        if e < best_energy:
            best_bits, best_energy = bits, e
    return best_bits, best_energy
