# papers/

Each file is a self-contained instance of `make_attack_synth` built from one paper in the
comparison table (`paper/threat_model_framework.tex`, Example 5), not from the framework in the
abstract. Model, property, adversary and verifier are all taken from that paper's own definitions
-- see each file's module docstring for the specific section/definition numbers and for what had to
be scaled down or approximated to run on the standard library alone (no SMT solver, no SPIN, no
PRISM, no package installs).

Run one: `python3 papers/oakley.py`
Run all: `python3 papers/run_all.py` (or `python3 -m papers.run_all` from the repo root)

| File | Paper | What's new in `attack_synth.py` that it exercises |
|---|---|---|
| `oakley.py` | Oakley, Oprea, Tripakis 2021 | `neighbors` (epsilon-ball sampling instead of enumerate+filter), `budget` (bounded best-first search standing in for the paper's SLSQP optimizer) |
| `vonhippel.py` | von Hippel, Vick, Tripakis, Nita-Rotaru 2020 | reuses the trace-extension search shape, but with a protocol-specific `neighbors` (DAISY's offered actions synchronized with a deterministic process) instead of arbitrary trace mutation |
| `bailluet.py` | Bailluet, Fleury, Puaut, Rohou 2025 | `robust=True` + `ModelType.perturb` (the paper's "for all unpredictable variables" universal quantifier, checked here by sampling instead of an SMT taint encoding) |
| `wardega.py` | Wardega et al. 2023 | `neighbors` restricted to graph moves that stay unobserved, `max_steps` set to the paper's own horizon bound `u* - s` |

Caveats are stated in-line in each docstring and are not hidden in this README: in particular,
`oakley.py`'s epsilon-ball search is a sampling approximation of the paper's exact optimizer, and
`bailluet.py`'s universal quantifier is checked by sampling a coarse grid rather than solving one
SMT query. Each illustrative instance (the warehouse grid, the handshake protocol, the backoff
chain, the gadget set) is built in the paper's style for this repository; none of the four papers'
own reported numbers are reproduced here.
