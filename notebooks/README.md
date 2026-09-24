# Notebooks — `qcoptlib` demos

Thin, narrated demonstrations of the library. All logic lives in `qcoptlib`; notebooks import it,
tell the story, and verify against brute force.

> **Status: in build.** More demos (a QAOA learning notebook, and the domain problems' walk-throughs
> in the sibling repos) are on the way.

## `inheritance_demo.ipynb` — number partitioning, and fair division under uncertainty

The clearest instance of the whole pipeline, on a classic NP-hard problem: split a set of values
into two groups whose sums are as equal as possible (`D² = (Σ vᵢ sᵢ)²` → one ZZ term per pair).
The friendly framing is dividing an inheritance fairly between two siblings.

**Part I — the raw showcase.** Thirteen asset values that split perfectly into two \$57M piles.
`partition_qubo` builds the QUBO; brute force gives the ground truth; `solve_qubo_qaoa` (Qiskit/Aer,
no account needed) recovers it — the full `QUBO → QAOA → verify` loop end to end.

**Parts II–IV — dividing under uncertainty.** What if the assets' *future* values are uncertain
(the Tel Aviv house is worth `μ ± σ`)? Three ideas, each a QUBO objective that simply adds to the
first:
- **The trap (proved, not asserted):** with *independent* per-asset variance, the expected squared
  gap gains only a constant term — so it **cannot** change the optimal split. A genuinely
  counter-intuitive result worth internalising.
- **Risk-balancing:** minimise value gap *and* risk gap `(Σ σᵢ² sᵢ)²` — equalise each sibling's
  volatility, not just their expected value. The split changes; risk is shared evenly.
- **Correlations:** when assets move together, `Var(D) = sᵀ C s` depends on the split, and the
  robust objective pushes correlated assets onto opposite siblings.

Uses `stochastic_partition_qubo`, `split_report`, and the same `QUBO → QAOA` pipeline; every result
is checked against brute force.

### Run it
```
poetry install
poetry run jupyter lab            # open notebooks/inheritance_demo.ipynb
```
The QAOA cells are cheap (13 qubits) and, where a solve recurs, cached via `cached_qaoa`.
