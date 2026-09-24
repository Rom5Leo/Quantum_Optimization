"""Constraint penalties for QUBOs.

QAOA/QUBO solvers are *unconstrained*: a hard constraint is enforced by adding a quadratic
penalty that is zero on feasible states and positive otherwise, then choosing the penalty weight
large enough that violating it never pays. These builders return the penalty as its own
:class:`~qcoptlib.qubo.core.QUBO`, so you compose it with the objective by addition
(``objective + equality_penalty(...).scaled(P)`` — or pass ``weight`` here).

- **Equality** ``Σ aᵢ xᵢ = t``  → penalty ``(Σ aᵢ xᵢ − t)²`` (workshop Part 1).
- **One-hot** "exactly one of these is 1" → the equality special case ``Σ xᵢ = 1``.
- **Inequality** ``Σ aᵢ xᵢ ≤ t`` → add a slack variable ``s ∈ {0..}`` and penalise
  ``(Σ aᵢ xᵢ + s − t)²`` (workshop Part 2); :func:`inequality_padding` sizes the slack bits.

The weight must exceed the largest objective swing a violation could otherwise buy; too small
lets infeasible states win, too large flattens the objective and hurts QAOA.
"""

from __future__ import annotations

import numpy as np

from qcoptlib.qubo.core import QUBO


def equality_penalty(n: int, coeffs, target: float, weight: float = 1.0) -> QUBO:
    """QUBO for ``weight * (Σ aᵢ xᵢ − target)²`` over ``n`` binary variables.

    Parameters
    ----------
    n       : total number of variables in the QUBO the penalty will join.
    coeffs  : the aᵢ, as a dict ``{var_index: coeff}`` (vars not listed have coeff 0) or a
              length-``n`` sequence.
    target  : the required value ``t`` of ``Σ aᵢ xᵢ``.
    weight  : penalty strength ``P``.

    Expansion (``xᵢ² = xᵢ``): const ``P t²``; linear ``P(aᵢ² − 2 t aᵢ)``; quadratic ``2 P aᵢ aⱼ``.
    """
    a = np.zeros(n)
    if isinstance(coeffs, dict):
        for i, c in coeffs.items():
            a[i] = c
    else:
        a[: len(coeffs)] = np.asarray(coeffs, dtype=float)

    q = QUBO.zeros(n)
    q.add_const(weight * target * target)
    nz = np.flatnonzero(a)
    for i in nz:
        q.add_linear(int(i), weight * (a[i] * a[i] - 2.0 * target * a[i]))
    for idx in range(len(nz)):
        for jdx in range(idx + 1, len(nz)):
            i, j = int(nz[idx]), int(nz[jdx])
            q.add_quadratic(i, j, weight * 2.0 * a[i] * a[j])
    return q


def onehot_penalty(n: int, indices, weight: float = 1.0) -> QUBO:
    """QUBO for ``weight * (Σ_{i∈indices} xᵢ − 1)²`` — enforce exactly one of ``indices`` set."""
    return equality_penalty(n, {int(i): 1.0 for i in indices}, target=1.0, weight=weight)


def inequality_padding(max_slack: int) -> int:
    """Number of binary slack bits needed to represent slack values ``0 .. max_slack``.

    For ``Σ aᵢ xᵢ ≤ t`` with non-negative integer coefficients, the slack can be as large as
    ``t`` (or the max achievable sum); size the slack register with this, then apply
    :func:`equality_penalty` to ``Σ aᵢ xᵢ + Σ 2ᵏ sₖ = t`` over the combined variables.
    """
    return int(max(0, max_slack)).bit_length()
