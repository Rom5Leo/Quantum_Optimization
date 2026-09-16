"""QUBO construction and manipulation (backend-agnostic)."""

from qcoptlib.qubo.core import QUBO
from qcoptlib.qubo.partition import partition_qubo, split_from_bits

__all__ = ["QUBO", "partition_qubo", "split_from_bits"]
