"""A backend-agnostic QUBO (Quadratic Unconstrained Binary Optimization) container.

A QUBO is the standard form that QAOA and quantum annealers consume:

    C(x) = sum_v h[v] x[v]  +  sum_{v<w} J[v,w] x[v] x[w]  +  const

over binary variables x[v] in {0, 1}. This module holds that object and the operations
that don't depend on any quantum backend — building it, evaluating it, brute-forcing it,
and converting it to the Ising (Z/ZZ) form. Problem-specific builders (see
:mod:`qcoptlib.qubo.partition`) construct a :class:`QUBO`; the QAOA layer consumes one.

The linear/quadratic/constant split mirrors the standard QUBO tutorials (Glover et al.) and
keeps everything pure NumPy so the same object runs through a Qiskit or a Classiq backend.

QUBOs compose: :meth:`scaled` and :meth:`__add__` let you weight and sum objectives that
share the same variables (e.g. balance *value* and balance *risk* in the same split), which
is exactly the linearity that makes multi-objective QUBO modelling clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np


@dataclass
class QUBO:
    """A QUBO over ``n`` binary variables.

    Attributes
    ----------
    n : number of binary variables.
    h : linear coefficients, shape ``(n,)``. ``h[v]`` multiplies ``x[v]``.
    J : quadratic coefficients, shape ``(n, n)``, kept strictly upper-triangular.
        ``J[v, w]`` (v < w) multiplies ``x[v] x[w]``.
    const : constant offset (does not affect the argmin, but keeps energies meaningful).

    Build one with :meth:`zeros` then :meth:`add_linear` / :meth:`add_quadratic`, or hand
    the arrays directly.
    """

    n: int
    h: np.ndarray
    J: np.ndarray
    const: float = 0.0

    def __post_init__(self) -> None:
        self.h = np.asarray(self.h, dtype=float).reshape(self.n)
        self.J = np.asarray(self.J, dtype=float).reshape(self.n, self.n)
        # normalize to strictly upper-triangular so a pair is never double-counted
        self.J = np.triu(self.J, k=1) + np.tril(self.J, k=-1).T
        self.J = np.triu(self.J, k=1)
        self.const = float(self.const)

    @classmethod
    def zeros(cls, n: int) -> "QUBO":
        """An all-zero QUBO over ``n`` variables, ready to add terms to."""
        return cls(n=n, h=np.zeros(n), J=np.zeros((n, n)), const=0.0)

    def add_linear(self, v: int, coeff: float) -> "QUBO":
        """Add ``coeff * x[v]`` to the cost. Returns self for chaining."""
        self.h[v] += coeff
        return self

    def add_quadratic(self, v: int, w: int, coeff: float) -> "QUBO":
        """Add ``coeff * x[v] x[w]`` to the cost (order-independent). Returns self."""
        if v == w:
            # x[v]^2 == x[v] for binaries, so a "quadratic" self-term is really linear
            self.h[v] += coeff
        else:
            self.J[min(v, w), max(v, w)] += coeff
        return self

    def add_const(self, c: float) -> "QUBO":
        """Add a constant offset. Returns self."""
        self.const += c
        return self

    def energy(self, x) -> float:
        """Evaluate C(x) on a bit vector (list or array of 0/1)."""
        x = np.asarray(x, dtype=float).reshape(self.n)
        return float(self.h @ x + x @ self.J @ x + self.const)

    def brute_force(self) -> tuple[tuple[int, ...], float]:
        """Return the (argmin bitstring, min energy) by exhaustive search.

        Only tractable for small ``n`` (<= ~20). Intended for verifying a QAOA result
        against ground truth on small instances, exactly as the hackathon notebooks do.
        """
        best_x, best_e = None, float("inf")
        for bits in product((0, 1), repeat=self.n):
            e = self.energy(bits)
            if e < best_e:
                best_x, best_e = bits, e
        return best_x, best_e

    # ---- composition (weight and sum objectives over the same variables) ----

    def copy(self) -> "QUBO":
        """A deep copy — useful before mutating a shared builder result."""
        return QUBO(n=self.n, h=self.h.copy(), J=self.J.copy(), const=self.const)

    def scaled(self, factor: float) -> "QUBO":
        """Return a new QUBO with every coefficient multiplied by ``factor``.

        Scaling a cost by a positive factor leaves the argmin unchanged; it is how a
        weighted objective (e.g. ``lambda * risk_term``) is expressed before summing.
        """
        return QUBO(n=self.n, h=self.h * factor, J=self.J * factor, const=self.const * factor)

    def __add__(self, other: "QUBO") -> "QUBO":
        """Sum two QUBOs over the same variables (coefficient-wise).

        The QUBO cost is linear in its coefficients, so ``(A + B).energy(x) ==
        A.energy(x) + B.energy(x)`` for every ``x``. This is what lets a multi-objective
        model — value balance + risk balance + gap variance — be built as a sum of
        independently-derived terms.
        """
        if isinstance(other, (int, float)) and other == 0:
            return self.copy()  # identity, so sum([...]) works (starts from int 0)
        if not isinstance(other, QUBO):
            return NotImplemented
        if other.n != self.n:
            raise ValueError(f"QUBO size mismatch: {self.n} vs {other.n}")
        return QUBO(n=self.n, h=self.h + other.h, J=self.J + other.J,
                    const=self.const + other.const)

    __radd__ = __add__  # so sum([...]) works (starts from int 0)

    def to_ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Convert to Ising form over spins s in {-1, +1} via ``x = (1 - s) / 2``.

        Returns ``(h_ising, J_ising, offset)`` where the Ising energy is
        ``sum_v h_ising[v] s[v] + sum_{v<w} J_ising[v,w] s[v] s[w] + offset``.
        This is the Z/ZZ Hamiltonian the QAOA cost layer applies (s <-> Z eigenvalue).
        """
        # x_v = (1 - s_v)/2. Substitute into h.x + x.J.x + const.
        h_i = np.zeros(self.n)
        J_i = np.zeros((self.n, self.n))
        offset = self.const

        # linear part: h_v * (1 - s_v)/2 = h_v/2  -  (h_v/2) s_v
        offset += 0.5 * self.h.sum()
        h_i += -0.5 * self.h

        # quadratic part: J_vw * (1 - s_v)/2 * (1 - s_w)/2
        #   = J_vw/4 * (1 - s_v - s_w + s_v s_w)
        for v in range(self.n):
            for w in range(v + 1, self.n):
                j = self.J[v, w]
                if j == 0.0:
                    continue
                offset += j / 4.0
                h_i[v] += -j / 4.0
                h_i[w] += -j / 4.0
                J_i[v, w] += j / 4.0

        return h_i, J_i, float(offset)
