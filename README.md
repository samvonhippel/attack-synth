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

## The paper-literal API

`attack_synth.py` also exposes a second, parallel API that names the paper's own Definitions
directly rather than the shorthand above -- useful when you want the code's correspondence to the
paper to be checkable by reading it, not just by running it:

```python
from attack_synth import Logic, ThreatModel

logic = Logic("name", verifier)                           # Definition 1: L = (M, Phi, |=)
tm = ThreatModel(logic, m, phi, model_type, edit_relation) # Definition 2: T = (m, phi, ~>)
                                                            #   (raises unless m |= phi)
tm.synthesize()                  # Definition 3: an attack (m ~> m') and (m' |/= phi)
tm.candidate_space()             # Definition 6's C(m) = {m' in M | m ~> m'}
tm.dual().realized()             # Definition 5/6: the dual threat model's realized dual attack
```
`make_attack_synth`/`ModelType` and `Logic`/`ThreatModel` are two views of the same engine --
`ThreatModel.synthesizer()` builds the exact same `AttackSynthesizer` the shorthand API returns, so
neither is more "real" than the other. `attack_synth.py`'s own comments cite the exact Definition
number (1-10, per `paper/threat_model_framework.tex`'s Section 6) each piece of code realizes.

## papers/

Four more instances, one per paper in `paper/threat_model_framework.tex`'s Example 5, built from
each paper's own definitions rather than from the framework in the abstract -- see
[papers/README.md](papers/README.md). Run `python3 papers/run_all.py`.

## tools/

Nine more instances, one per tool in `paper/threat_model_framework.tex`'s Example 4 (Tamarin, SPIN,
ProVerif, nuXmv, PRISM, HyTech, Alloy, UPPAAL, Margrave), each built from that tool's own documented
model class and adversary notion -- see [tools/README.md](tools/README.md). Run `python3 tools/run_all.py`.

## examples/

Two further instances in model classes neither `tools/` nor `papers/` names (adversarial-ML
robustness, distributed-consensus safety) -- generality stress tests beyond the paper's own counted
instances, grounded in their fields' established threat models rather than one specific tool or
paper -- see [examples/README.md](examples/README.md). Run `python3 examples/run_all.py`.

## tests/

`tests/test_conformance.py` -- a standard-library `unittest` suite checking the paper-literal API
against the paper's own Definitions directly: structural-dual involution, the threat-model
baseline precondition, that a found attack is genuine (checked against the raw verifier/edit
relation, not just trusted from the search), dual-goal correctness, severity's clamping and its
three named cases, and exact equivalence between the paper-literal API and the `make_attack_synth`
shorthand on a shared instance (reusing `demos.py`'s own LTL demo). Run
`python3 -m unittest tests.test_conformance -v`.
