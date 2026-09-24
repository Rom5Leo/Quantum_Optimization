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
from qcoptlib.qubo.constraints import (
    equality_penalty,
    onehot_penalty,
    inequality_padding,
)

__all__ = [
    "QUBO",
    "partition_qubo",
    "split_from_bits",
    "stochastic_partition_qubo",
    "gap_variance_qubo",
    "split_report",
    "SplitReport",
    "equality_penalty",
    "onehot_penalty",
    "inequality_padding",
]
