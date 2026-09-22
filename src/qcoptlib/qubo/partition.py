"""Number-partitioning QUBO builders — deterministic and under uncertainty.

The number-partitioning problem: split a set of values into two groups so their sums are as
equal as possible. This is the encoding behind the "fair inheritance split" example — an
NP-hard problem that maps cleanly onto a QUBO.

Deterministic encoding
----------------------
Give each value ``i`` a spin ``s_i in {-1, +1}`` (which group). The difference between the two
group sums is  ``D = sum_i v_i s_i``, and we minimise ``D^2``:

    D^2 = (sum_i v_i s_i)^2
        = sum_i v_i^2                       (constant, s_i^2 = 1)
        + 2 sum_{i<j} v_i v_j s_i s_j       (the interaction term)

Dropping the constant, the Ising cost is ``sum_{i<j} 2 v_i v_j s_i s_j`` — one ZZ term per
pair, weighted by the product of values. We express it as a :class:`QUBO` (binary ``x_i``,
0 = group A, 1 = group B), which :meth:`QUBO.to_ising` maps back to that spin form.

Splitting under uncertainty
---------------------------
When each asset's *future* value is uncertain — the Tel Aviv house is worth ``mu +/- sigma`` —
a fair split is no longer just about matching expected value. Three objectives, all in the
same variables, capture that (see :func:`stochastic_partition_qubo`):

1. **Value fairness** ``(sum_i mu_i s_i)^2`` — equal expected value (the deterministic term).
2. **Risk balance** ``(sum_i sigma_i^2 s_i)^2`` — equal *risk exposure*: neither sibling
   should end up holding all the volatile assets. This is a second number-partitioning
   problem, on the variances.
3. **Gap robustness** ``Var(D) = s^T C s`` — how much the value gap can swing once the true
   values are revealed. For *independent* assets this term is a **constant** (it does not
   depend on the split), so it cannot change the optimum — a genuinely counter-intuitive
   fact. It only bites when assets are **correlated** (off-diagonal covariance -> ZZ terms),
   where it pushes correlated assets onto opposite siblings so their swings cancel in the gap.

Because QUBOs compose (:meth:`QUBO.scaled` / :meth:`QUBO.__add__`), the combined objective is
just a weighted sum of these three builders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qcoptlib.qubo.core import QUBO


# --------------------------------------------------------------------------- #
# Core builder: balance a set of weights across the two groups.
# --------------------------------------------------------------------------- #

def _balance_qubo(weights) -> QUBO:
    """QUBO whose minimum most-evenly balances ``weights`` across the two groups.

    Encodes ``(sum_i w_i s_i)^2`` with ``s_i = 1 - 2 x_i`` (x=0 -> group A / s=+1,
    x=1 -> group B / s=-1). Expanding in x:

        D = sum_i w_i (1 - 2 x_i) = W - 2 sum_i w_i x_i,   W = sum_i w_i
        D^2 = W^2 - 4 W sum_i w_i x_i + 4 (sum_i w_i x_i)^2
        (sum_i w_i x_i)^2 = sum_i w_i^2 x_i  +  2 sum_{i<j} w_i w_j x_i x_j   (x^2 = x)

    giving const ``W^2``, linear ``-4 W w_i + 4 w_i^2`` and quadratic ``8 w_i w_j``.
    :func:`partition_qubo` is exactly ``_balance_qubo(values)``.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    q = QUBO.zeros(n)

    total = w.sum()
    q.add_const(total * total)
    for i in range(n):
        q.add_linear(i, -4.0 * total * w[i] + 4.0 * w[i] * w[i])
    for i in range(n):
        for j in range(i + 1, n):
            q.add_quadratic(i, j, 8.0 * w[i] * w[j])
    return q


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
    """
    return _balance_qubo(values)


def gap_variance_qubo(cov) -> QUBO:
    """QUBO for the variance of the value gap, ``Var(D) = s^T C s``.

    ``cov`` is the (n, n) covariance matrix of the asset values. Writing ``D = sum_i v_i s_i``
    with random ``v``, ``Var(D) = sum_{i,j} C_ij s_i s_j``. The diagonal contributes
    ``sum_i C_ii`` (a constant, since ``s_i^2 = 1``); each off-diagonal pair contributes
    ``2 C_ij s_i s_j``. Substituting ``s_i s_j = 1 - 2 x_i - 2 x_j + 4 x_i x_j`` gives the
    QUBO terms below.

    Consequence worth internalising: if ``cov`` is diagonal (independent assets) the whole
    thing collapses to a constant, so **adding independent per-asset variance to the split
    objective changes nothing**. Only genuine correlations (non-zero off-diagonals) make this
    term depend on the assignment.
    """
    C = np.asarray(cov, dtype=float)
    n = C.shape[0]
    q = QUBO.zeros(n)

    # diagonal -> constant offset (does not affect the argmin, keeps energies meaningful)
    q.add_const(float(np.trace(C)))

    for i in range(n):
        for j in range(i + 1, n):
            a = 2.0 * C[i, j]              # symmetric pair contribution: C_ij + C_ji
            if a == 0.0:
                continue
            q.add_const(a)                 # +a * 1
            q.add_linear(i, -2.0 * a)      # +a * (-2 x_i)
            q.add_linear(j, -2.0 * a)      # +a * (-2 x_j)
            q.add_quadratic(i, j, 4.0 * a) # +a * (4 x_i x_j)
    return q


def stochastic_partition_qubo(
    means,
    variances=None,
    cov=None,
    risk_weight: float = 0.0,
    robust_weight: float = 0.0,
) -> QUBO:
    """Fair-split QUBO under value uncertainty: value balance + risk balance + gap robustness.

    Minimises

        (sum_i mu_i s_i)^2  +  risk_weight * (sum_i sigma_i^2 s_i)^2  +  robust_weight * Var(D)

    Parameters
    ----------
    means : expected asset values ``mu_i`` (the split-by-value term is always included).
    variances : per-asset variances ``sigma_i^2``. If omitted, taken from ``diag(cov)``.
        Used by the risk-balance term. Required if ``risk_weight > 0`` and ``cov`` is None.
    cov : full covariance matrix. Used by the gap-robustness term when ``robust_weight > 0``.
        If omitted, ``diag(variances)`` is used (independent assets) — in which case the
        robustness term is constant and has no effect on the optimum (see
        :func:`gap_variance_qubo`).
    risk_weight : lambda for the risk-balance objective (equalise each sibling's volatility).
    robust_weight : gamma for the gap-variance objective (keep the gap stable across scenarios).

    Returns
    -------
    A single :class:`QUBO` over ``len(means)`` binaries, ready for brute force or QAOA.
    """
    mu = np.asarray(means, dtype=float)
    n = len(mu)

    if variances is None and cov is not None:
        variances = np.diag(np.asarray(cov, dtype=float))

    q = _balance_qubo(mu)  # objective 1: value fairness

    if risk_weight:
        if variances is None:
            raise ValueError("risk_weight > 0 requires `variances` (or a `cov` to read its diagonal)")
        var = np.asarray(variances, dtype=float)
        q = q + _balance_qubo(var).scaled(risk_weight)  # objective 2: risk balance

    if robust_weight:
        if cov is None:
            if variances is None:
                raise ValueError("robust_weight > 0 requires `cov` (or `variances` for the independent case)")
            C = np.diag(np.asarray(variances, dtype=float))
        else:
            C = np.asarray(cov, dtype=float)
        q = q + gap_variance_qubo(C).scaled(robust_weight)  # objective 3: gap robustness

    if q.n != n:  # defensive: every term is over the same n variables
        raise AssertionError("internal QUBO size mismatch")
    return q


# --------------------------------------------------------------------------- #
# Read-out helpers.
# --------------------------------------------------------------------------- #

def split_from_bits(bits, values) -> tuple[list, list, float]:
    """Decode a bit vector into the two groups and the absolute sum difference.

    Returns ``(group_A, group_B, gap)`` where ``x[i] = 0`` -> group A, ``x[i] = 1`` -> B.
    """
    bits = np.asarray(bits, dtype=int)
    group_a = [values[i] for i in range(len(values)) if bits[i] == 0]
    group_b = [values[i] for i in range(len(values)) if bits[i] == 1]
    gap = abs(sum(group_a) - sum(group_b))
    return group_a, group_b, float(gap)


@dataclass
class SplitReport:
    """A fairness read-out of one split, in value and in risk.

    Attributes
    ----------
    idx_a, idx_b : asset indices assigned to sibling A / sibling B.
    value_a, value_b : each sibling's total expected value.
    value_gap : ``|value_a - value_b|``.
    risk_a, risk_b : each sibling's bundle standard deviation (same units as value),
        computed from the within-group covariance (so correlations are respected).
    risk_gap : ``|risk_a - risk_b|`` — how unevenly the volatility is shared.
    gap_std : standard deviation of the value gap ``D`` itself (``sqrt(Var(D))``) — how much
        the split's fairness can drift once the true values are revealed.
    """

    idx_a: list
    idx_b: list
    value_a: float
    value_b: float
    value_gap: float
    risk_a: float
    risk_b: float
    risk_gap: float
    gap_std: float


def split_report(bits, means, variances=None, cov=None) -> SplitReport:
    """Summarise a split's fairness in both expected value and risk.

    ``cov`` (if given) is used for the risk quantities so correlations are respected;
    otherwise ``diag(variances)`` is assumed (independent assets). ``variances`` alone is
    enough for the independent case.
    """
    bits = np.asarray(bits, dtype=int)
    mu = np.asarray(means, dtype=float)
    n = len(mu)

    if cov is not None:
        C = np.asarray(cov, dtype=float)
    elif variances is not None:
        C = np.diag(np.asarray(variances, dtype=float))
    else:
        C = np.zeros((n, n))

    idx_a = [i for i in range(n) if bits[i] == 0]
    idx_b = [i for i in range(n) if bits[i] == 1]

    value_a = float(mu[idx_a].sum()) if idx_a else 0.0
    value_b = float(mu[idx_b].sum()) if idx_b else 0.0

    def _bundle_std(idx):
        if not idx:
            return 0.0
        sub = C[np.ix_(idx, idx)]
        return float(np.sqrt(max(sub.sum(), 0.0)))

    risk_a = _bundle_std(idx_a)
    risk_b = _bundle_std(idx_b)

    s = 1.0 - 2.0 * bits  # x=0 -> +1 (A), x=1 -> -1 (B)
    gap_var = float(s @ C @ s)
    gap_std = float(np.sqrt(max(gap_var, 0.0)))

    return SplitReport(
        idx_a=idx_a, idx_b=idx_b,
        value_a=value_a, value_b=value_b, value_gap=abs(value_a - value_b),
        risk_a=risk_a, risk_b=risk_b, risk_gap=abs(risk_a - risk_b),
        gap_std=gap_std,
    )
