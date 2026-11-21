# attack_synth

Generic attack synthesis for the threat-model framework `T = (m, φ, ⇝)` over arbitrary logics.

```python
from attack_synth import make_attack_synth, ModelType
synth = make_attack_synth(verifier, ModelType("name", universe_fn), edit_relation, measure=mu)
synth(m, phi)                                   # attacks: m ⇝ m' with m' ⊭ φ   (ranked by severity σ)
synth(m, phi, goal="dual", dual_phi=phi_d)      # m' ⊨ (φ)^d
synth(m, phi, max_steps=2)                      # allow 2 composed edits (⇝ is not assumed transitive)
```
The edit relation is encoded as a boolean function `f(m, m2) -> bool` (a relation R ⊆ X×Y as f: X×Y → B);
`ModelType.universe` supplies the finite candidate set the relation filters.

`python3 demos.py [-v] [--only LTL,CTL,...] [--steps K]` runs LTL, CTL, DEL, PCTL, STL, Turing machine,
micro-architectural side channel (cache prime+probe) and Hoare-on-Python-source demos. Standard library only.

## Generalizing beyond small, enumerable models

`ModelType` can give the search a `neighbors(m)` generator instead of (or in addition to)
`universe(m)` + `edit_relation`: a generator already restricted to `m ~> cand`, for spaces too
large or continuous to enumerate and filter (sampling, bounded synthesis, ...). The search itself
is a budgeted best-first search: with no `measure` and no `budget` it is plain, exhaustive
breadth-first search (exactly the original behavior); with a `measure` it expands the most severe
candidates first, so a `budget` (a cap on candidates examined) still finds good attacks on huge
spaces instead of stalling on an exhaustive sweep. `robust=True` adds the "for all unpredictable
inputs" universal quantifier: an attack must hold for every `model_type.perturb(m')` of it too.

## papers/

Four more instances, one per paper in `paper/threat_model_framework.tex`'s Example 5, built from
each paper's own definitions rather than from the framework in the abstract -- see
[papers/README.md](papers/README.md). Run `python3 papers/run_all.py`.
