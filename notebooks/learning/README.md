# Learning series — quantum optimization methods

Notes I keep as I build `qcoptlib`: each notebook teaches one method, grounded in the library and
runnable on the local Qiskit/Aer backend (no Classiq account needed). Distilled from the QUBIT ×
AT&T hackathon workshop material and the reference papers.

> **Status: in build.** QAOA and Grover are up; QML follows. With QAOA + Grover done, the
> antenna-tilt project execution begins next.

| # | Notebook | What it teaches | Uses |
|---|---|---|---|
| 1 | `01_qaoa.ipynb` | QAOA from a cost function to a quantum solve: cost→Ising, cost/mixer layers, the two cost-layer front-ends (Classiq `phase` vs Qiskit Pauli), a Max-Cut solve vs brute force, and constraint penalties (equality + slack). | `qcoptlib.qubo`, `qcoptlib.quantum`, `qcoptlib.viz` |
| 2 | `02_grover.ipynb` | Grover / amplitude amplification (exact NumPy statevector): the amplification visualized, the rotation geometry + overshoot, and **Grover Adaptive Search finding a graph's max cut** — the same problem QAOA solves, by search. Qiskit circuit sketched. | `qcoptlib.quantum.grover` |
| 3 | `03_qml.ipynb` *(planned)* | Quantum machine-learning solvers. | — |
| — | `inheritance_demo.ipynb` | Flagship library demo: fair division, then division under uncertainty. | `qcoptlib.qubo`, `qcoptlib.quantum` |

Each notebook caches its QAOA solves to `qaoa_cache/` (via `cached_qaoa`), so re-running to tweak
plots or text doesn't recompute.
