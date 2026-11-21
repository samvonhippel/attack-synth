"""
attack_synth.py -- a generic attack-synthesis factory for the threat-model framework.

Framework (from the paper)
--------------------------
  Logic         L = (M, Phi, |=)        -> `verifier(m, phi) -> bool`
  Threat model  T = (m, phi, ~>)        -> baseline model m, invariant phi, edit relation
  Attack        (m ~> m') and (m' |/= phi)
  Dual goal     m' |= (phi)^d           -> goal="dual"
  Severity      sigma = 1 - mu(M')/mu(M)-> optional `measure(m) -> float`

The "trick": a relation R subset X x Y is encoded as a boolean function
f(x, y) = True iff x R y.  So `edit_relation(m, m2) -> bool` is *the* relation.
A relation cannot be enumerated for infinite model sets, so a `ModelType` supplies a
(finite) candidate universe for each model; the relation then filters that universe.

    synth = make_attack_synth(verifier, model_type, edit_relation, measure=...)
    result = synth(m, phi)                       # attacks: m ~> m' with m' |/= phi
    result = synth(m, phi, goal="dual", dual_phi=...)   # m' |= phi^d
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Optional


# --------------------------------------------------------------------------- formulas
# A tiny, logic-agnostic formula AST used by the LTL / CTL / DEL / STL / PCTL demos.
@dataclass(frozen=True)
class Atom:
    name: str


@dataclass(frozen=True)
class Not:
    f: Any


@dataclass(frozen=True)
class Op:
    name: str
    args: tuple


def neg(f):
    """Negation with double-negation elimination."""
    return f.f if isinstance(f, Not) else Not(f)


def op(name: str, *args) -> Op:
    return Op(name, tuple(args))


def And(*a) -> Op:
    return Op("and", tuple(a))


def Or(*a) -> Op:
    return Op("or", tuple(a))


def Implies(a, b) -> Op:
    return Op("implies", (a, b))


def show(f) -> str:
    if isinstance(f, Atom):
        return f.name
    if isinstance(f, Not):
        return "¬" + (show(f.f) if isinstance(f.f, (Atom, Not)) else "(" + show(f.f) + ")")
    sym = {"and": " ∧ ", "or": " ∨ ", "implies": " ⇒ "}
    if f.name in sym:
        return "(" + sym[f.name].join(show(a) for a in f.args) + ")"
    return f.name + "(" + ", ".join(show(a) for a in f.args) + ")"


def structural_dual(f):
    """The structural (De Morgan) dual of the paper:
         (p)^d = ¬p,  (¬p)^d = p,  (w(psi_1..psi_n))^d = ¬ w(¬psi_1, .., ¬psi_n)
       extended to ¬chi (chi compound) by (¬chi)^d = ¬(chi)^d, so that it is an involution."""
    if isinstance(f, Atom):
        return Not(f)
    if isinstance(f, Not):
        return f.f if isinstance(f.f, Atom) else neg(structural_dual(f.f))
    return neg(Op(f.name, tuple(neg(a) for a in f.args)))


# --------------------------------------------------------------------------- core
@dataclass(frozen=True)
class ModelType:
    """The set M of models, given operationally.

    universe(m)  -> iterable of candidate models that *might* be reachable from m
                    (a finite over-approximation; the edit relation filters it)
    key(m)       -> hashable identity used to de-duplicate models
    """
    name: str
    universe: Callable[[Any], Iterable[Any]]
    key: Callable[[Any], Hashable] = lambda m: m


@dataclass(frozen=True)
class Attack:
    model: Any                 # the attacked model m'
    path: tuple                # (m, m1, ..., m')  -- length 2 for a single edit step
    sigma: Optional[float] = None


@dataclass
class SynthResult:
    goal: str
    attacks: list
    explored: int
    seconds: float
    steps: int

    @property
    def found(self) -> bool:
        return bool(self.attacks)

    @property
    def best(self) -> Optional[Attack]:
        return self.attacks[0] if self.attacks else None


class AttackSynthesizer:
    """The program returned by make_attack_synth."""

    def __init__(self, verifier, model_type, edit_relation, measure=None, max_steps=1, name=""):
        self.verifier = verifier
        self.model_type = model_type
        self.edit_relation = edit_relation
        self.measure = measure
        self.max_steps = max_steps
        self.name = name or model_type.name

    def sigma(self, m, m2) -> Optional[float]:
        """Weighted cut-off ratio  sigma = 1 - mu(m')/mu(m), clamped to [0, 1]."""
        if self.measure is None:
            return None
        mu0, mu1 = self.measure(m), self.measure(m2)
        if mu0 <= 0:
            return 0.0
        return min(1.0, max(0.0, 1.0 - mu1 / mu0))

    def __call__(self, m, phi, *, goal: str = "violate", dual_phi=None,
                 max_steps: Optional[int] = None, limit: Optional[int] = None) -> SynthResult:
        """Search for attacks.

        goal="violate": attack iff m' does not satisfy phi            (Definition: attack)
        goal="dual":    attack iff m' satisfies dual_phi (default: structural_dual(phi))
        max_steps=k:    allow k composed edit steps m ~> m1 ~> ... ~> m' (default 1 = the paper's
                        definition; ~> is deliberately not assumed transitive, so k bounds capability)
        """
        if goal not in ("violate", "dual"):
            raise ValueError("goal must be 'violate' or 'dual'")
        if not self.verifier(m, phi):
            raise ValueError("threat model requires the baseline to satisfy the invariant (m |= phi)")
        if goal == "dual":
            goal_phi = dual_phi if dual_phi is not None else structural_dual(phi)
            is_attack = lambda c: self.verifier(c, goal_phi)
        else:
            is_attack = lambda c: not self.verifier(c, phi)

        key, universe, rel = self.model_type.key, self.model_type.universe, self.edit_relation
        steps = self.max_steps if max_steps is None else max_steps
        t0 = time.perf_counter()
        seen = {key(m)}
        frontier = [(m, (m,))]
        attacks, explored = [], 0
        for _ in range(steps):
            nxt = []
            for cur, path in frontier:
                for cand in universe(cur):
                    k = key(cand)
                    if k in seen or not rel(cur, cand):     # relation encoded as a boolean function
                        continue
                    seen.add(k)
                    explored += 1
                    p = path + (cand,)
                    if is_attack(cand):
                        attacks.append(Attack(cand, p, self.sigma(m, cand)))
                    else:
                        nxt.append((cand, p))               # only extend non-attacks
            frontier = nxt
        attacks.sort(key=lambda a: (-(a.sigma if a.sigma is not None else 0.0), len(a.path)))
        if limit is not None:
            attacks = attacks[:limit]
        return SynthResult(goal, attacks, explored, time.perf_counter() - t0, steps)


def make_attack_synth(verifier: Callable[[Any, Any], bool],
                      model_type: ModelType,
                      edit_relation: Callable[[Any, Any], bool],
                      *, measure: Optional[Callable[[Any], float]] = None,
                      max_steps: int = 1, name: str = "") -> AttackSynthesizer:
    """Build an attack synthesizer for one (logic, model type, edit relation) triple.

    verifier       the satisfaction relation |= : (model, property) -> bool
    model_type     ModelType giving the candidate universe of models
    edit_relation  the adversarial capability ~>, encoded as f(m, m2) -> bool
    measure        optional mu(model) -> float for the severity ratio sigma
    max_steps      number of composed edits allowed (1 = the paper's definition)
    """
    return AttackSynthesizer(verifier, model_type, edit_relation, measure, max_steps, name)
