# [project-name]-qc

[One-line problem statement] — solved as a staged quantum-classical hybrid.

## Goal

[What real problem this solves, and where quantum fits as one component.]

## Approach: quantum is a helper, not the whole

[State plainly: the classical method X does the heavy lifting; QAOA handles the
combinatorial core Y. This hybrid framing is the honest, effective one.]

## Project stages

1. **Faithful classical model + baseline** — the problem, a strong classical solver, verified ground truth.
2. **Quantum formulation** — QUBO/QAOA on the hard combinatorial sub-problem.
3. **Hybrid integration** — quantum + classical working together; the two-stage split.
4. **Honest benchmark** — where quantum helps, where it doesn't, and the scaling argument.

## Depends on

[qc-core](https://github.com/Rom5Leo/quantum-computation) — shared QAOA/QUBO/RF library.

    poetry add git+https://github.com/Rom5Leo/quantum-computation.git

## Status

Stage [N] of 4. Worked at a deliberate pace alongside other projects.
