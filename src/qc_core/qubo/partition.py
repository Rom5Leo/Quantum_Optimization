"""Number-partitioning QUBO builder.

The number-partitioning problem: split a set of values into two groups so their sums are as
equal as possible. This is the encoding behind the "fair inheritance split" example — an
NP-hard problem that maps cleanly onto a QUBO.

Encoding (from the inheritance-QAOA notebook):
    Give each value i a spin s_i in {-1, +1} (which group). The difference between the two
    group sums is  D = sum_i v_i s_i,  and we minimise D^2:

        D^2 = (sum_i v_i s_i)^2
            = sum_i v_i^2                       (constant, s_i^2 = 1)
            + 2 sum_{i<j} v_i v_j s_i s_j       (the interaction term)

    Dropping the constant, the Ising cost is  sum_{i<j} 2 v_i v_j s_i s_j  — one ZZ term per
    pair, weighted by the product of values. We express it here as a :class:`QUBO` (binary
    x_i, 0 = group A, 1 = group B), which :meth:`QUBO.to_ising` maps back to that spin form.
"""

from __future__ import annotations

import numpy as np

from qc_core.qubo.core import QUBO


def partition_qubo(values) -> QUBO:
    """Build the QUBO whose minimum is the most-balanced two-way split of ``values``.

    Parameters
    ----------
    values : sequence of item values (the numbers to split).

    Returns
    -------
    QUBO over ``len(values)`` binary variables; ``x[i] = 0`` puts item i in group A,
    ``x[i] = 1`` puts it in group B. Minimising the QUBO minimises the squared difference
    between the two group sums.

    Notes
    -----
    In binary variables, with ``s_i = 1 - 2 x_i`` (so x=0 -> s=+1, x=1 -> s=-1),
    ``D = sum_i v_i s_i = sum_i v_i (1 - 2 x_i)``. Expanding ``D^2`` in x gives linear and
    pairwise terms; we add them straight to a QUBO. (Equivalently one can build the Ising
    form directly — this route keeps everything in the shared QUBO container.)
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    q = QUBO.zeros(n)

    total = v.sum()
    # D = sum_i v_i (1 - 2 x_i) = total - 2 sum_i v_i x_i
    # D^2 = total^2 - 4 total sum_i v_i x_i + 4 (sum_i v_i x_i)^2
    # (sum_i v_i x_i)^2 = sum_i v_i^2 x_i (since x^2=x) + 2 sum_{i<j} v_i v_j x_i x_j
    q.add_const(total * total)
    for i in range(n):
        # -4*total*v_i x_i  +  4*v_i^2 x_i
        q.add_linear(i, -4.0 * total * v[i] + 4.0 * v[i] * v[i])
    for i in range(n):
        for j in range(i + 1, n):
            q.add_quadratic(i, j, 8.0 * v[i] * v[j])  # 4 * 2 * v_i v_j
    return q


def split_from_bits(bits, values) -> tuple[list, list, float]:
    """Decode a bit vector into the two groups and the absolute sum difference.

    Returns ``(group_A, group_B, gap)`` where ``x[i] = 0`` -> group A, ``x[i] = 1`` -> B.
    """
    bits = np.asarray(bits, dtype=int)
    v = np.asarray(values, dtype=float)
    group_a = [values[i] for i in range(len(values)) if bits[i] == 0]
    group_b = [values[i] for i in range(len(values)) if bits[i] == 1]
    gap = abs(sum(group_a) - sum(group_b))
    return group_a, group_b, float(gap)
