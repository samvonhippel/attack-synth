"""
papers/oakley.py -- L. Oakley, A. Oprea, S. Tripakis, "Adversarial Robustness Verification and
Attack Synthesis in Stochastic Systems" (arXiv:2110.02125).

Model m = a DTMC  C = (S, s0, P, AP, L).
Property phi = a PCTL bound  P_{>=p}(psi)  for a bounded path formula psi (F<=k or G<=k over a
label); the paper's PCTL* allows richer psi, which the bounded-horizon reach_prob below covers for
F/G but not arbitrary nesting.
Adversary ~> : C ~> C' iff C' has the same (S, s0, L) and a matrix P' with every row within
max-entry distance epsilon of P's row (the paper's epsilon-ball), restricted to one of four
variants:
  SS   -- selected states:      any entries of the chosen rows may move
  ST   -- selected transitions: only a declared allow-list of (row, col) pairs may move
  SPSS, SPST -- the structure-preserving versions of the above: an entry that was exactly 0 must
               stay exactly 0 (the adversary cannot add a transition that did not exist)
Attack: a perturbed C' with Pr_{C'}(psi) < Pr_C(psi) - delta, found by a single-shot
epsilon-ball sample (max_steps=1, since the paper's threat model is one perturbed matrix, not a
sequence of edits) ranked by severity sigma = 1 - Pr_{C'}(psi) / Pr_C(psi).

The paper solves the *exact* worst case with SLSQP (a continuous nonlinear optimizer) or, for the
structure-preserving case, PARAM's closed-form parametric model checking. Both are unavailable
here (no network access to install SciPy or PARAM), so this is a *sampling* approximation: the
best-first, severity-ranked search in attack_synth already behaves like a (black-box, derivative-
free) local search for the worst case, and increasing `budget` only tightens the approximation.
"""
from __future__ import annotations

import itertools
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_synth import Atom, ModelType, make_attack_synth  # noqa: E402


@dataclass(frozen=True)
class DTMC:
    P: tuple            # tuple of rows, each a tuple of floats summing to 1
    labels: tuple        # per-state frozenset of atomic propositions
    init: int = 0


def reach_prob(M: DTMC, target: set, bound: int) -> float:
    """Pr_C(F<=bound target) by bound steps of backward value iteration (or to a fixpoint if
    bound is None), which also gives Pr_C(G<=bound (complement of target)) = 1 - this."""
    x = [1.0 if s in target else 0.0 for s in range(len(M.P))]
    for _ in range(bound if bound is not None else 100_000):
        y = [1.0 if s in target else sum(p * v for p, v in zip(M.P[s], x)) for s in range(len(M.P))]
        if bound is None and max(abs(a - b) for a, b in zip(x, y)) < 1e-13:
            x = y
            break
        x = y
    return x[M.init]


@dataclass(frozen=True)
class PCTLBound:
    """P_{cmp p}(F<=bound label) or P_{cmp p}(G<=bound label)."""
    kind: str       # "F" or "G"
    label: str
    bound: int
    cmp: str        # ">=", ">", "<=", "<"
    p: float


def pctl_sat(M: DTMC, phi: PCTLBound) -> bool:
    holds = {s for s in range(len(M.P)) if phi.label in M.labels[s]}
    pr = reach_prob(M, holds, phi.bound) if phi.kind == "F" else \
        1.0 - reach_prob(M, set(range(len(M.P))) - holds, phi.bound)
    return {">=": pr >= phi.p - 1e-12, ">": pr > phi.p + 1e-12,
            "<=": pr <= phi.p + 1e-12, "<": pr < phi.p - 1e-12}[phi.cmp]


def prob_of(M: DTMC, phi: PCTLBound) -> float:
    holds = {s for s in range(len(M.P)) if phi.label in M.labels[s]}
    pr = reach_prob(M, holds, phi.bound)
    return pr if phi.kind == "F" else 1.0 - reach_prob(M, set(range(len(M.P))) - holds, phi.bound)


Variant = str   # "SS", "ST", "SPSS", "SPST"
GRID = (0.002, 0.005, 0.01, 0.02, 0.04, 0.08)


def epsilon_ball(M: DTMC, epsilon: float, variant: Variant, allowed: frozenset, rng: random.Random, samples_per_row: int):
    """Sample matrices C' with ||P - P'||_max <= epsilon, one row perturbed at a time.

    allowed: for ST/SPST, the set of (row, col) pairs the adversary may touch at all; ignored
    for SS/SPSS, where every column of a selected row is fair game.
    """
    n = len(M.P)
    structure_preserving = variant in ("SPSS", "SPST")
    transition_restricted = variant in ("ST", "SPST")
    for i in range(n):
        cols = [j for j in range(n) if (not transition_restricted or (i, j) in allowed)
                and (not structure_preserving or M.P[i][j] > 0)]
        if len(cols) < 2:
            continue
        for _ in range(samples_per_row):
            j, k = rng.sample(cols, 2)
            d = rng.choice(GRID)
            if d > epsilon or M.P[i][k] < d:
                continue
            row = list(M.P[i])
            row[k] = round(row[k] - d, 10)
            row[j] = round(row[j] + d, 10)
            yield DTMC(M.P[:i] + (tuple(row),) + M.P[i + 1:], M.labels, M.init)


def worst_case_attack(M: DTMC, phi: PCTLBound, *, epsilon: float, variant: Variant = "SS",
                      allowed: frozenset = frozenset(), budget: int = 4000, samples_per_row: int = 60,
                      seed: int = 0):
    """Problem 3 of the paper: find (an approximation of) the C' in the epsilon-ball maximizing
    the probability drop Pr_C(psi) - Pr_C'(psi). Returns the attack_synth SynthResult."""
    rng = random.Random(seed)
    mt = ModelType("DTMC (Oakley)", neighbors=lambda m: epsilon_ball(m, epsilon, variant, allowed, rng, samples_per_row),
                  key=lambda m: m.P)
    mu = lambda m: prob_of(m, phi)                      # measure = Pr_m(psi); sigma = 1 - Pr'/Pr
    synth = make_attack_synth(pctl_sat, mt, measure=mu, max_steps=1)
    return synth(M, phi, budget=budget)


def robustness_report(M: DTMC, phi: PCTLBound, delta: float, **kw):
    """The paper's three problems, derived from one worst-case search:
       (3) worst-case attack, (2) the maximal delta the search can certify, (1) delta-robustness."""
    base = prob_of(M, phi)
    r = worst_case_attack(M, phi, **kw)
    if not r.found:
        return {"base_prob": base, "worst_prob": base, "max_delta_found": 0.0,
                "robust_at_delta": True, "result": r}
    worst = min(prob_of(a.model, phi) for a in r.attacks)
    max_delta = base - worst
    return {"base_prob": base, "worst_prob": worst, "max_delta_found": max_delta,
            "robust_at_delta": max_delta <= delta + 1e-9, "result": r}


# --------------------------------------------------------------------------- an illustrative instance
def backoff_chain() -> DTMC:
    """A small binary-exponential-backoff wireless MAC in the spirit of the paper's collision-
    avoidance case study (not its reported transition probabilities, which the source does not
    give here): idle -> sense -> {transmit (success) | backoff (collision), deeper each retry} ->
    delivered, or dropped after too many retries."""
    # states: 0 idle, 1 sense, 2 backoff1, 3 backoff2, 4 backoff3, 5 delivered, 6 dropped
    P = (
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),           # idle -> sense
        (0.0, 0.0, 0.3, 0.0, 0.0, 0.7, 0.0),           # sense: 0.7 clear channel -> delivered
        (0.0, 0.0, 0.0, 0.2, 0.0, 0.8, 0.0),           # backoff1: 0.8 retry succeeds
        (0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0),           # backoff2: 0.9 retry succeeds
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.05),         # backoff3: last try, 0.05 dropped
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),           # delivered: absorbing
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),           # dropped: absorbing
    )
    labels = (frozenset(), frozenset(), frozenset(), frozenset(), frozenset(),
             frozenset({"delivered"}), frozenset({"dropped"}))
    return DTMC(P, labels, init=0)


def main():
    M = backoff_chain()
    phi = PCTLBound("F", "delivered", 20, ">=", 0.95)
    print("Oakley et al. -- DTMC epsilon-ball attack synthesis")
    print(f"  base model:  Pr(F<=20 delivered) = {prob_of(M, phi):.4f}   phi: {phi}")
    for variant in ("SS", "ST", "SPSS", "SPST"):
        allowed = frozenset({(1, 5), (1, 2), (2, 5), (2, 3), (3, 5), (3, 6)}) if variant in ("ST", "SPST") else frozenset()
        rep = robustness_report(M, phi, delta=0.02, epsilon=0.1, variant=variant, allowed=allowed, budget=3000)
        r = rep["result"]
        print(f"  [{variant:4}] sampled {r.explored:5} candidates in {r.seconds*1e3:6.1f} ms  "
              f"worst Pr={rep['worst_prob']:.4f}  max delta found={rep['max_delta_found']:.4f}  "
              f"robust at delta=0.02: {rep['robust_at_delta']}")
        if r.found:
            b = r.best
            diffs = [(i, j, M.P[i][j], b.model.P[i][j]) for i in range(len(M.P)) for j in range(len(M.P[i]))
                    if M.P[i][j] != b.model.P[i][j]]
            print(f"         worst-case perturbation: " + ", ".join(f"P[{i}][{j}]: {a:.3f}->{bb:.3f}" for i, j, a, bb in diffs))


if __name__ == "__main__":
    main()
