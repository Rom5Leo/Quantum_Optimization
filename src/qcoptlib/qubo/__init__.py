"""QUBO construction and manipulation (backend-agnostic)."""

from qcoptlib.qubo.core import QUBO
from qcoptlib.qubo.partition import (
    partition_qubo,
    split_from_bits,
    stochastic_partition_qubo,
    gap_variance_qubo,
    split_report,
    SplitReport,
)

__all__ = [
    "QUBO",
    "partition_qubo",
    "split_from_bits",
    "stochastic_partition_qubo",
    "gap_variance_qubo",
    "split_report",
    "SplitReport",
]
