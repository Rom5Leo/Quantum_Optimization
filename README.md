# quantum-optimization (`qcoptlib`)

A reusable library of **quantum solvers for combinatorial optimization**, built on Classiq
and Qiskit. QUBO construction, QAOA (two backends), and the read-out/plotting helpers that
sit around a variational solve.

This is a **library**, not an application. Domain problems (telecom, finance, …) live in
their own repositories and depend on this one to solve their QUBOs with quantum methods.

## What's inside

| Package | What it does |
|---|---|
| `qcoptlib.qubo` | Build QUBOs (linear + quadratic + constant), evaluate, brute-force, convert to Ising. Includes a number-partitioning demo builder. |
| `qcoptlib.quantum` | QAOA runners — `qiskit_backend` (Aer, fully local) and `classiq_backend` (Classiq simulator). Shared angle-init and read-out in `common`. Grover / QML planned. |
| `qcoptlib.viz` | Plotting helpers: convergence, sampled-counts, before/after, 2-D network. |

## The pipeline

    problem  ->  QUBO (qcoptlib.qubo)  ->  QAOA solve (qcoptlib.quantum)  ->  read out + plot (qcoptlib.viz)

## Install

    poetry install
    poetry run pytest -v        # 24 tests

`classiq` is an optional extra — the library imports and tests without it; you only need it
to execute the Classiq backend.

## Roadmap
- [x] QUBO layer + QAOA (Qiskit + Classiq)
- [ ] Grover / amplitude amplification
- [ ] QML solvers
