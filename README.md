# Quantum Computation

Reusable quantum-optimization library, learning material, and the QUBIT×AT&T
Hackathon 2026 solution — built on Classiq / Qmod.

## Goal

Build a professional quantum-computing codebase around QAOA and combinatorial
optimization: a reusable core library, a set of learning notebooks, and real
telecom optimization problems tackled as staged projects.

## Contents

| Path | What it is |
|---|---|
| `src/qc_core/` | Reusable library — QAOA ansatz/encodings, QUBO assembly, RF models, plotting |
| `notebooks/learning/` | Practice notebooks (Qiskit basics, Grover, QAOA encodings) |
| `notebooks/classiq_workshop/` | Classiq/Qmod workshop material |
| `hackathon-2026/` | The AT&T hackathon: team solution, an alternative approach, and their comparison |
| `docs/` | Concept write-ups and cited RF references |

## The three challenge projects

The hackathon seeded three telecom optimization problems, each continued as its own
staged project repository that depends on this library:

- [antenna-tilt-qc](https://github.com/Rom5Leo/antenna-tilt-qc) — antenna down-tilt optimization
- [dispatch-qc](https://github.com/Rom5Leo/dispatch-qc) — field-technician dispatch (VRP)
- [routing-qc](https://github.com/Rom5Leo/routing-qc) — network traffic routing

Each treats quantum computing as **one component** of a hybrid solution, not the whole
answer — the honest and effective framing for near-term quantum optimization.

## Install

    poetry install

## Stack

Python · Poetry · Classiq (Qmod) · NumPy · Pyomo · matplotlib
