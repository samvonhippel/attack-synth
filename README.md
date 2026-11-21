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
