# quantum-optimization (`qcoptlib`)

A reusable library of **quantum solvers for combinatorial optimization**, built on Qiskit and
Classiq. QUBO construction and composition, QAOA (two backends) with a disk cache, and the
read-out / plotting helpers that sit around a variational solve.

This is a **library**, not an application. Domain problems (telecom, and later dispatch and
routing) live in their own repositories and depend on this one to turn their QUBOs into quantum
solves. Classical-first, quantum as a pluggable, honestly-benchmarked component.

> **Status: in build.** The QUBO + QAOA core is complete and tested; Grover / QML are planned.
> APIs may still shift as the domain repos exercise them.

## What's inside

| Package | What it does |
|---|---|
| `qcoptlib.qubo` | Build QUBOs (linear + quadratic + constant); `energy`, `brute_force`, `to_ising`; **compose** them (`scaled`, `__add__`) to weight and sum objectives over the same variables. Problem builders: `partition_qubo` (number partitioning / fair split) and `stochastic_partition_qubo` (splitting under value uncertainty — value balance + risk balance + gap-variance), with `split_report` read-out. |
| `qcoptlib.quantum` | QAOA runners — `qiskit_backend` (Aer, fully local) and `classiq_backend` (Classiq simulator); shared angle-init and score-and-keep-best read-out in `common`; **`cache.cached_qaoa`** memoizes solves to disk so notebooks don't recompute unchanged runs. |
| `qcoptlib.viz` | Plotting helpers: convergence, sampled-counts, before/after, 2-D network — each draws onto a given Axes. |

## The pipeline

    problem  ->  QUBO (qcoptlib.qubo)  ->  QAOA solve (qcoptlib.quantum)  ->  read out + plot (qcoptlib.viz)

Every builder returns the same `QUBO` object; every backend consumes it. Brute force
(`QUBO.brute_force`) is the ground-truth check on small instances.

## Notebooks
- `notebooks/inheritance_demo.ipynb` — the flagship demo: number partitioning as a *fair
  inheritance split*, then splitting **under uncertainty** (the counter-intuitive
  constant-variance result, risk-balancing, and correlations). See `notebooks/README.md`.
- A **QC learning notebook** distilling the QAOA workshop / hackathon material is in progress.

## Design notes
- QAOA read-out scores every sampled bitstring by real QUBO energy and keeps the best, so a
  weakly-concentrated distribution still yields a good answer.
- `solve_qubo_qaoa` treats `maxiter` as a cap; COBYLA self-terminates on convergence (`tol`),
  and the run reads out from the *best angles seen* (the optimizer's trace is non-monotone).
- Convergence-independent facts (qubit count, term counts, exact optima) are checked by brute
  force in the tests, not asserted.

## Install & test

    poetry install
    poetry run pytest -v        # qubo · partition · stochastic split · QAOA · cache · viz

`classiq` is an optional extra — the library imports and tests without it; you only need it to
*run* the Classiq backend (`poetry add classiq`).

## Roadmap
- [x] QUBO layer (+ composition) and QAOA (Qiskit + Classiq)
- [x] Stochastic / robust partition builders; QAOA disk cache
- [ ] Grover / amplitude amplification
- [ ] QML solvers
