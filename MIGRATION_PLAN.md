# Migration & Build Plan

How to turn the hackathon work into this professional repo structure, step by step.
Do it at a slow pace — each phase is independently shippable.

## Phase 0 — Create the hub repo (1 sitting)
1. Create `quantum-computation` on GitHub.
2. Drop in this scaffold (README, pyproject.toml, .gitignore, src/ tree, docs/).
3. `poetry install`, commit, push. You now have a professional skeleton.

## Phase 1 — Extract the library from notebooks (your 3.1.1 + 3.1.2)
Move inline notebook code into `src/qc_core/`:
- `rf/` ← the Ericsson SINR model (path loss, antenna pattern, SINR, spectral efficiency).
  Cite each function to its reference equation (you already have RF_model_references.md).
- `qubo/` ← QUBO assembly + penalty helpers (adapt from Michael's qubo.py, generalize).
- `qaoa/encodings.py` ← one-hot AND binary encodings (from Michael's qaoa.py).
- `qaoa/runner.py` ← the ExecutionSession two-stage hybrid loop.
- `qaoa/metrics.py` ← circuit depth/width/cx extraction.
- `viz/` ← the house-style plots (from figures.py).

Then rewrite notebooks to be THIN: `import qc_core`, call functions, markdown narrates.
Rule: no notebook cell > ~10 lines of logic; physics lives in src/.

## Phase 2 — Organize the learning material (your 3.1.3)
- `notebooks/learning/` ← the Qiskit basics, Grover, QAOA-encoding practice notebooks.
- `notebooks/classiq_workshop/` ← the workshop material, kept separate from challenges.
These are self-contained and depend only on qc_core + classiq.

## Phase 3 — The hackathon subdirectory (your 3.1.4)
- `hackathon-2026/team-solution/` ← add Michael's repo as a git submodule (keeps attribution,
  no duplication):  `git submodule add https://github.com/MichaelMorami/Classiq-Hackathon-2026`
- `hackathon-2026/alternative-approach/` ← your SINR-direct notebook + its src helpers.
- `hackathon-2026/comparison.md` ← write the two-approaches analysis (portfolio gold).
- `hackathon-2026/pitch/` ← the deck + speaker notes.

## Phase 4 — Spin up the three project repos (your section 3)
For each of antenna-tilt-qc / dispatch-qc / routing-qc:
1. Create the repo, copy PROJECT_TEMPLATE_README.md, adapt.
2. `poetry add git+.../quantum-computation.git` to depend on the hub.
3. Work Stage 1 (classical model + baseline) first — shippable on its own.
4. Add quantum stages later, at your pace.

Suggested order: antenna-tilt-qc first (you have the most material), then dispatch-qc
(the richest hybrid story), then routing-qc.

## The professional touches that matter (from FAHM)
- Staged README with a clear roadmap (signals deliberate pace, not abandonment).
- src/ layout + Poetry (not loose scripts).
- docs/ with concept write-ups and cited references.
- .gitignore that excludes Classiq tokens (.env, .classiq/) — never commit credentials.
- One clear "what/why" per README; results up top with a headline number.
