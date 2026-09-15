"""QAOA runners. Shared helpers here; backends in qiskit_backend / classiq_backend."""

from qc_core.qaoa.common import QAOAResult, adiabatic_init, best_bits_from_counts

__all__ = ["QAOAResult", "adiabatic_init", "best_bits_from_counts"]
