"""
attack_synth.py -- a generic attack-synthesis factory for the threat-model framework.

Framework (from paper/threat_model_framework.tex -- Definition numbers are that document's,
read off the compiled PDF; Sections 1-4 define them, Section 6 "Mechanization" names this
correspondence explicitly as Table 5)
-------------------------------------------------------------------------------------------
  Definition 1  Logic         L = (M, Phi, |=)        -> `verifier(m, phi) -> bool`
  Definition 2  Threat model  T = (m, phi, ~>)        -> baseline model m, invariant phi, edit relation
  Definition 3  Attack        (m ~> m') and (m' |/= phi)
  Definition 4  Structural dual (phi)^d               -> `structural_dual(phi)`
  Definition 5  Dual threat model T^d = (m, (phi)^d, ~>)
  Definition 6  Realized dual attack  m' |= (phi)^d   -> goal="dual"
  Definition 7  Probability measure space (M, Sigma, mu)
  Definition 8  Measure mass mu(M) on subsets of the model set
  Definition 9  Severity      sigma = 1 - mu(M')/mu(M)-> optional `measure(m) -> float`
  Definition 10 Realized-attack search problem (decidability/enumerability) -> this whole module

The "trick" (introduced informally alongside Definition 2, used throughout): a relation
R subset X x Y is encoded as a boolean function f(x, y) = True iff x R y.
`edit_relation(m, m2) -> bool` is *that* function, encoding ~> subset M x M.

Two ways to give the search the models reachable by one edit (a ModelType may offer either,
or both):
  1. universe(m) -> candidates, filtered by edit_relation(m, cand)   (the boolean-function trick,
     exact for small/enumerable candidate sets -- used by the Example 1/2 demos)
  2. neighbors(m) -> candidates *already restricted* to m ~> cand     (a generator, for spaces too
     large or continuous to enumerate and filter -- used by the Oakley/Bailluet instances, which
     sample an epsilon-ball or expand a gadget library rather than iterating all of M)

Search is a budgeted best-first search (a strict generalization of the original plain BFS):
with no `measure` and no `budget` it degenerates to breadth-first and is exhaustive up to
max_steps, exactly as before; with a `measure` it expands the most severe candidates first, so a
`budget` (a cap on how many candidates get examined) still finds good attacks on huge or
unenumerable spaces instead of stalling on an exhaustive sweep.

    synth = make_attack_synth(verifier, model_type, edit_relation, measure=...)
    result = synth(m, phi)                                  # attacks: m ~> m' with m' |/= phi
    result = synth(m, phi, goal="dual", dual_phi=...)        # m' |= (phi)^d
    result = synth(m, phi, max_steps=2, budget=5000, limit=1)
    result = synth(m, phi, robust=True)                      # m' is an attack only if every
                                                              # perturb(m') of it is too (the
                                                              # "for all unpredictable inputs"
                                                              # quantifier of Bailluet et al.)
"""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Optional

_UNSET = object()   # distinguishes "use the constructor's max_steps" from an explicit max_steps=None


# --------------------------------------------------------------------------- formulas
# A tiny, logic-agnostic formula AST used by the LTL / CTL / DEL / STL / PCTL demos.
# This AST is exactly "Phi, the set of full formulas generated inductively over atomic
# variables AP and operators Omega" from Definition 4's preamble: Atom = a literal p in AP,
# Not = explicit negation, Op = a compound formula omega(psi_1, ..., psi_n) for any operator
# omega in Omega (G/F/X/U for LTL, AG/EF/... for CTL, K_a/[alpha] for DEL, etc. -- the AST
# itself is logic-agnostic; each demo supplies its own `verifier` to interpret Op.name).
@dataclass(frozen=True)
class Atom:
    name: str   # a literal p in AP (Definition 4)


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
    """Definition 4: the structural (De Morgan) dual (.)^d : Phi -> Phi, defined over the
    primary constructor at the root of f's AST, satisfying ((phi)^d)^d = phi (an involution,
    checked directly in tests/test_conformance.py's TestStructuralDualInvolution):
         Literals:          (p)^d = not p,   (not p)^d = p
         Compound formulas: (w(psi_1, ..., psi_n))^d = not (w(not psi_1, ..., not psi_n))
    The `isinstance(f, Not)` branch below is the two literal cases taken together (Not(Atom)
    is "(not p)", handled by returning f.f = p); the recursive `neg(structural_dual(f.f))` case
    for Not(compound) is this implementation's own extension of the literal definition above
    (the paper's two bullets only cover Not(Atom) and a bare compound at the root), chosen
    specifically to preserve the stated involution property when the root happens to be a
    negated compound rather than a bare one."""
    if isinstance(f, Atom):
        return Not(f)                                    # (p)^d = not p
    if isinstance(f, Not):
        return f.f if isinstance(f.f, Atom) else neg(structural_dual(f.f))   # (not p)^d = p; extension for not(compound)
    return neg(Op(f.name, tuple(neg(a) for a in f.args)))  # (w(psi_1..psi_n))^d = not w(not psi_1, .., not psi_n)


# --------------------------------------------------------------------------- core
@dataclass(frozen=True)
class ModelType:
    """Definition 1's M (the set of models) given operationally, plus (jointly with
    AttackSynthesizer.edit_relation) Definition 2's ~> subset M x M, via candidates_of()
    below -- together these realize Definition 6's candidate space C(m) = {m' in M | m ~> m'}.

    universe(m)   -> candidates that *might* be reachable from m (an over-approximation that
                     edit_relation then filters) -- required unless `neighbors` is given.
    neighbors(m)  -> candidates *already* restricted to the ones with m ~> cand -- use this
                     instead of (or alongside) `universe` when the candidate space is too large
                     or continuous to enumerate and filter (sampling, bounded synthesis, ...).
    key(m)        -> hashable identity used to de-duplicate models.
    perturb(m)    -> other instantiations of whatever in m is adversary-*un*predictable (default:
                     just m itself). Used only when a search asks for `robust=True` attacks: an
                     attack must then hold for every perturb(m'), the "for all unpredictable
                     inputs" quantifier of Bailluet et al.'s robust gadget chains (not itself a
                     numbered Definition in the paper -- see Example 5's Bailluet et al. column).
    """
    name: str
    universe: Optional[Callable[[Any], Iterable[Any]]] = None
    neighbors: Optional[Callable[[Any], Iterable[Any]]] = None
    key: Callable[[Any], Hashable] = lambda m: m
    perturb: Callable[[Any], Iterable[Any]] = lambda m: (m,)

    def candidates_of(self, cur):
        """Candidates reachable from `cur`, before edit_relation filtering (see
        AttackSynthesizer.candidate_space for the fully-filtered Definition 6 C(m))."""
        return self.neighbors(cur) if self.neighbors is not None else self.universe(cur)


@dataclass(frozen=True)
class Attack:
    """One witness of Definition 3 (goal="violate") or Definition 6 (goal="dual")."""
    model: Any                 # the attacked model m' (Definitions 3 and 6's m')
    path: tuple                # (m, m1, ..., m')  -- length 2 for a single edit step
    sigma: Optional[float] = None   # Definition 9's severity sigma, if a measure was supplied


@dataclass
class SynthResult:
    goal: str
    attacks: list
    explored: int
    seconds: float
    steps: int
    truncated: bool = False    # True iff `budget` stopped the search before it was exhaustive

    @property
    def found(self) -> bool:
        return bool(self.attacks)

    @property
    def best(self) -> Optional[Attack]:
        return self.attacks[0] if self.attacks else None


class AttackSynthesizer:
    """The program returned by make_attack_synth -- Definitions 1 and 2's (L, T) bundled with a
    search procedure for Definition 10's realized-attack search problem."""

    def __init__(self, verifier, model_type, edit_relation=None, measure=None, max_steps: Optional[int] = 1, name=""):
        self.verifier = verifier                                       # Definition 1: |=
        self.model_type = model_type                                   # Definition 1: M (operationally)
        self.edit_relation = edit_relation or (lambda cur, cand: True)  # Definition 2: ~> subset M x M
        self.measure = measure                                         # Definition 7/8: mu
        self.max_steps = max_steps                                     # Definition 2's k composed edits (default 1)
        self.name = name or model_type.name

    def sigma(self, m, m2) -> Optional[float]:
        """Definition 9: severity sigma = 1 - mu(m')/mu(m), clamped to [0, 1].
        Delegates to the standalone `severity()` function below."""
        if self.measure is None:
            return None
        return severity(self.measure, m, m2)

    def candidate_space(self, m) -> Iterable[Any]:
        """Definition 6's C(m) = {m' in M | m ~> m'}, materialized directly: every candidate
        ModelType offers from m, filtered by edit_relation -- exactly the set the search below
        draws from at depth 1. Useful for inspecting or testing C(m) without running a full
        search (e.g. to check `m' in synth.candidate_space(m)` the way the conformance tests do)."""
        return (cand for cand in self.model_type.candidates_of(m) if self.edit_relation(m, cand))

    def __call__(self, m, phi, *, goal: str = "violate", dual_phi=None,
                 max_steps=_UNSET, limit: Optional[int] = None,
                 budget: Optional[int] = None, robust: bool = False) -> SynthResult:
        """Search for attacks.

        goal="violate": attack iff m' does not satisfy phi            (Definition: attack)
        goal="dual":    attack iff m' satisfies dual_phi (default: structural_dual(phi))
        max_steps=k:    allow k composed edit steps m ~> m1 ~> ... ~> m' (default 1 = the paper's
                        definition; ~> is deliberately not assumed transitive, so k bounds capability).
                        Omit to use the value given to make_attack_synth; pass max_steps=None for
                        *no* depth bound at all, when the model has no natural step horizon (a
                        continuous-time system discretized into steps of its own, say). Unbounded
                        search still terminates once the reachable space is exhausted, but on an
                        effectively infinite space it terminates only via `budget` -- pass one.
        budget=n:       stop after examining n candidates, even if the search is not exhaustive
                        (`result.truncated` says whether it stopped early). Needed when the
                        candidate space is too large to enumerate in full -- see `neighbors` above,
                        and required in practice whenever max_steps=None on an unbounded space.
        limit=k:        stop as soon as k attacks are found, in addition to any budget
        robust=True:    an attack must hold for every model_type.perturb(m') of it too (the
                        universally-quantified "unpredictable variable" reading of an attack)

        Candidates are expanded best-first by severity (sigma) when a `measure` is given --
        otherwise, in insertion order, which is exactly breadth-first and (with budget=None)
        reproduces the original exhaustive search.
        """
        if goal not in ("violate", "dual"):
            raise ValueError("goal must be 'violate' or 'dual'")
        if not self.verifier(m, phi):
            # Definition 2's precondition on a threat model: "phi in Phi is a safety invariant
            # satisfied by m, i.e. (m |= phi)". A threat model that doesn't satisfy this isn't
            # one per the paper, so we refuse to search rather than silently proceeding.
            raise ValueError("threat model requires the baseline to satisfy the invariant (m |= phi)")
        if goal == "dual":
            # Definition 6: realized dual attack, m' |= (phi)^d, for T^d = (m, (phi)^d, ~>)
            # (Definition 5). dual_phi defaults to structural_dual(phi) (Definition 4) but may be
            # supplied explicitly when phi isn't in the Atom/Not/Op AST (see tests/test_conformance's
            # TestDualThreatModel.test_realized_dual_attack for a plain-predicate example).
            goal_phi = dual_phi if dual_phi is not None else structural_dual(phi)
            holds = lambda c: self.verifier(c, goal_phi)
        else:
            # Definition 3: attack iff (m ~> m') and (m' |/= phi) -- the "(m' |/= phi)" conjunct.
            holds = lambda c: not self.verifier(c, phi)
        is_attack = (lambda c: holds(c) and all(holds(p) for p in self.model_type.perturb(c))) if robust else holds

        key, rel = self.model_type.key, self.edit_relation
        steps = self.max_steps if max_steps is _UNSET else max_steps   # steps may be None: unbounded
        t0 = time.perf_counter()
        seen = {key(m)}
        counter = itertools.count()                  # stable tie-break; avoids comparing models
        heap = [(0.0, next(counter), 0, m, (m,))]     # (priority, tiebreak, depth, model, path)
        attacks, explored, truncated = [], 0, False
        while heap:
            if budget is not None and explored >= budget:
                truncated = True
                break
            _, _, depth, cur, path = heapq.heappop(heap)
            for cand in self.model_type.candidates_of(cur):   # Definition 1's M, offered operationally
                k = key(cand)
                if k in seen or not rel(cur, cand):           # `rel` = Definition 2's ~> (the "m ~> m'" conjunct)
                    continue
                seen.add(k)
                explored += 1
                p = path + (cand,)
                s = self.sigma(m, cand)                       # Definition 9: severity, if a measure is given
                if is_attack(cand):
                    # Definition 3 (or 6, if goal="dual"): both conjuncts now hold for `cand`.
                    attacks.append(Attack(cand, p, s))
                    if limit is not None and len(attacks) >= limit:
                        return self._finish(goal, attacks, explored, t0, steps, truncated, limit)
                elif steps is None or depth + 1 < steps:
                    # Not yet an attack: keep expanding, composing another edit step (Definition 2's
                    # remark that ~> need not be associative -- composing k edits is this search's
                    # own choice, bounded by `steps`, not a property of ~> itself).
                    heapq.heappush(heap, (-(s if s is not None else 0.0), next(counter), depth + 1, cand, p))
                if budget is not None and explored >= budget:
                    truncated = True
                    break
        return self._finish(goal, attacks, explored, t0, steps, truncated, limit)

    @staticmethod
    def _finish(goal, attacks, explored, t0, steps, truncated, limit):
        attacks.sort(key=lambda a: (-(a.sigma if a.sigma is not None else 0.0), len(a.path)))
        if limit is not None:
            attacks = attacks[:limit]
        return SynthResult(goal, attacks, explored, time.perf_counter() - t0, steps, truncated)


def make_attack_synth(verifier: Callable[[Any, Any], bool],
                      model_type: ModelType,
                      edit_relation: Optional[Callable[[Any, Any], bool]] = None,
                      *, measure: Optional[Callable[[Any], float]] = None,
                      max_steps: Optional[int] = 1, name: str = "") -> AttackSynthesizer:
    """Build an attack synthesizer for one (logic, model type, edit relation) triple --
    Definitions 1 and 2's (L, T), minus the baseline m and invariant phi (supplied per-call to
    the returned AttackSynthesizer, since one (L, T)-shaped search is usually reused across
    several (m, phi) pairs -- see demos.py for many examples).

    verifier       Definition 1's satisfaction relation |= : (model, property) -> bool
    model_type     ModelType giving the candidate models reachable by one edit (see ModelType)
    edit_relation  Definition 2's adversarial capability ~>, encoded as f(m, m2) -> bool. Optional
                   when model_type.neighbors already restricts candidates to m ~> cand.
    measure        optional mu(model) -> float for Definition 9's severity ratio sigma
    max_steps      number of composed edits allowed (1 = Definition 2's literal single m ~> m')
    """
    return AttackSynthesizer(verifier, model_type, edit_relation, measure, max_steps, name)


# --------------------------------------------------------------------------- paper conformance
# The constructs above (ModelType / AttackSynthesizer / make_attack_synth) are deliberately loose
# about naming -- they are built to be used directly, not to recite the paper back. The classes
# below are an *additional*, parallel layer that names each paper Definition literally and wires
# it to the exact same engine, so the correspondence between code and paper is something you can
# read off the field names (and that `tests/test_conformance.py` checks), not just assert in prose.
# Nothing elsewhere in this repository (demos.py, papers/*.py, tools/*.py) is required to use this
# layer -- they keep calling make_attack_synth directly, and still do exactly what they did before.

@dataclass(frozen=True)
class Logic:
    """Definition 1: a logic L = (M, Phi, |=).

    M and Phi are left implicit -- whatever `verifier`'s two arguments range over -- exactly as
    the paper leaves M and Phi as an arbitrary model class and property language; `verifier` *is*
    the satisfaction relation |=, encoded (per the module docstring's "the trick") as the boolean
    function f(m, phi) = True iff m |= phi. Formally |= subset M x Phi.
    """
    name: str
    verifier: Callable[[Any, Any], bool]


@dataclass(frozen=True)
class ThreatModel:
    """Definition 2: a threat model T = (m, phi, ~>) over a fixed Logic.

    m              the baseline model, "representing the uncompromised system" (required to
                   satisfy phi -- see __post_init__)
    phi            the invariant/property the baseline is meant to uphold ("satisfied by m")
    model_type     gives ~>, the adversary's one-edit capability (ModelType.universe/neighbors);
                   Definition 2 does not assume ~> is associative, "since this would rule out
                   notions of bounded capability" -- see max_steps below
    edit_relation  the boolean-function encoding of ~> subset M x M, when it isn't already built
                   into model_type.neighbors (see the module docstring's "the trick")
    measure        optional mu for the severity ratio sigma (Definition 7/8: measure space)
    max_steps      composed edits allowed (1 = Definition 2's literal single m ~> m'; k > 1 is
                   this implementation's own choice of how much capability to grant, per the
                   remark above that ~> need not be associative -- see Section 6's Remark on
                   Definition 10)
    """
    logic: Logic
    m: Any
    phi: Any
    model_type: ModelType
    edit_relation: Optional[Callable[[Any, Any], bool]] = None
    measure: Optional[Callable[[Any], float]] = None
    max_steps: Optional[int] = 1

    def __post_init__(self):
        if not self.logic.verifier(self.m, self.phi):
            # Definition 2's precondition: "phi in Phi is a safety invariant satisfied by m,
            # i.e., (m |= phi)". Without this, (m, phi, ~>) is not a threat model per the paper.
            raise ValueError("Definition 2 (threat model) requires the baseline to satisfy "
                              "the invariant: m |= phi")

    def candidate_space(self) -> Iterable[Any]:
        """Definition 6's C(m) = {m' in M | m ~> m'}, for this threat model's own m."""
        return self.synthesizer().candidate_space(self.m)

    def dual(self) -> "DualThreatModel":
        """Definition 5: the dual threat model T^d = (m, (phi)^d, ~>) -- same m, same ~>,
        phi replaced by its structural dual (phi)^d (Definition 4)."""
        return DualThreatModel(self, structural_dual(self.phi))

    def synthesizer(self) -> AttackSynthesizer:
        return make_attack_synth(self.logic.verifier, self.model_type, self.edit_relation,
                                  measure=self.measure, max_steps=self.max_steps,
                                  name=self.logic.name)

    def synthesize(self, *, goal: str = "violate", **kwargs) -> SynthResult:
        """Definition 3: attack -- (m ~> m') and (m' |/= phi), found by searching ~> from m
        (Definition 10's realized-attack search problem, instantiated for this threat model)."""
        return self.synthesizer()(self.m, self.phi, goal=goal, **kwargs)


@dataclass(frozen=True)
class DualThreatModel:
    """Definition 5: T^d = (m, (phi)^d, ~>). Deliberately NOT a ThreatModel subclass: Definition 5
    does not require m |= (phi)^d at the baseline (only Definition 2's original T requires
    m |= phi), so reusing ThreatModel.__post_init__ here would wrongly reject valid dual threat
    models whose baseline happens to already violate the dual property."""
    base: ThreatModel
    phi_d: Any

    def realized(self, **kwargs) -> SynthResult:
        """Definition 6: a realized dual attack -- some m' in C(m) = {m' in M | m ~> m'}
        (candidate_space() below) with m' |= (phi)^d."""
        return self.base.synthesize(goal="dual", dual_phi=self.phi_d, **kwargs)

    def candidate_space(self) -> Iterable[Any]:
        """Definition 6's C(m), identical to the base threat model's -- the dual only replaces
        phi with (phi)^d (Definition 5); ~> and m are unchanged, so C(m) is unchanged too."""
        return self.base.candidate_space()


@dataclass(frozen=True)
class MeasureSpace:
    """Definition 7: a probability measure space (M, Sigma, mu). Sigma (the sigma-algebra) is
    left implicit, as it is never consulted directly -- only mu : M -> R is ever evaluated,
    exactly as `AttackSynthesizer.sigma` only ever calls `measure(m)` (Definition 8's "each
    subset of models is assigned a probability mass mu(M)")."""
    mu: Callable[[Any], float]


def severity(space_or_mu, m, m_prime) -> float:
    """Definition 9: severity sigma = 1 - mu(M')/mu(M), clamped to [0, 1].
       sigma == 0          : nominal operation (the attack cut off nothing mu-weighted)
       0 < sigma < 1       : some branches/mass severed, system still partially executable
       sigma == 1          : fully compromised (no surviving mu-weighted behavior)
    Accepts either a MeasureSpace or a bare mu callable, so it can be used standalone or via
    MeasureSpace.mu -- the same formula AttackSynthesizer.sigma delegates to below.

    Definition 9 does not itself state the [0, 1] clamp (it follows from M' being a subset of M
    under a genuine Definition 7 measure, where mu(M') <= mu(M) always). The clamp here is a
    defensive fallback: several instances in this repository (e.g. papers/oakley.py,
    tools/prism.py) use `mu` as a per-model scalar (a reach/satisfaction probability) evaluated
    on *every* candidate the search examines, for best-first ranking -- not only on confirmed
    attacks -- so it is routine and expected for a non-attack candidate explored mid-search to
    have mu(m') slightly above mu(m) (a candidate that turned out to be no worse, or momentarily
    better, than the baseline). The clamp folds that case to sigma=0 rather than a meaningless
    small negative number; it is not itself evidence of anything wrong with `mu`.
    """
    mu = space_or_mu.mu if isinstance(space_or_mu, MeasureSpace) else space_or_mu
    mu0, mu1 = mu(m), mu(m_prime)
    if mu0 <= 0:
        return 0.0
    return min(1.0, max(0.0, 1.0 - mu1 / mu0))
