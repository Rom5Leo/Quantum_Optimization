# Notebooks — `qcoptlib`

Thin, narrated notebooks. All logic lives in `qcoptlib`; notebooks import it, tell the story, and
verify against brute force.

> **Status: in build.**

## `learning/` — concept notes as the library is built
A short series distilling the quantum-optimization methods behind `qcoptlib`, each runnable on the
local Qiskit backend (no account needed):

| Notebook | Topic |
|---|---|
| `learning/01_qaoa.ipynb` | QAOA end-to-end: cost→Ising, cost/mixer layers, the Classiq `phase` vs Qiskit-Pauli cost layer, a Max-Cut solve, and constraint penalties. |
| `learning/02_grover.ipynb` | *(planned)* Grover / amplitude amplification. |
| `learning/03_qml.ipynb` | *(planned)* QML solvers. |
| `learning/inheritance_demo.ipynb` | The flagship demo: number partitioning as a fair inheritance split, then splitting under uncertainty (the constant-variance trap, risk-balancing, correlations). |

See `learning/README.md` for the series overview.
