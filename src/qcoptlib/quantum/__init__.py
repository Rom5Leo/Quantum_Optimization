"""Quantum solvers: QAOA (Qiskit + Classiq) and Grover amplitude-amplification / search."""

from qcoptlib.quantum.common import QAOAResult, adiabatic_init, best_bits_from_counts
from qcoptlib.quantum.cache import cached_qaoa, qaoa_cache_key
from qcoptlib.quantum.grover import (
    grover_angle, grover_optimal_iterations, grover_success_probability,
    uniform_state, apply_oracle, apply_diffuser, grover_search,
    grover_adaptive_search, GASResult,
)

__all__ = [
    "QAOAResult", "adiabatic_init", "best_bits_from_counts",
    "cached_qaoa", "qaoa_cache_key",
    "grover_angle", "grover_optimal_iterations", "grover_success_probability",
    "uniform_state", "apply_oracle", "apply_diffuser", "grover_search",
    "grover_adaptive_search", "GASResult",
]
