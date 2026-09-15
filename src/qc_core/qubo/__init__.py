"""QUBO construction and manipulation (backend-agnostic)."""

from qc_core.qubo.core import QUBO
from qc_core.qubo.partition import partition_qubo, split_from_bits

__all__ = ["QUBO", "partition_qubo", "split_from_bits"]
