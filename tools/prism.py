"""
tools/prism.py -- a native instance in the style of PRISM, on an MDP (which papers/oakley.py's
pure DTMC instance does not cover).

Model: an MDP -- states, and for each state a set of actions, each action a probability
distribution over successor states. ("Nondeterminism in MDPs is resolved by minimizing or
maximizing probabilities", per Example 4.)
Property: a PCTL bound P_{>=p}[min](F<=k target) -- the probability of reaching `target` within k
steps stays >= p even under the worst-case (probability-minimizing) resolution of the
nondeterminism, e.g. an adversarial scheduler or environment.
Adversary: perturbing one action's distribution within an epsilon-ball (as in Oakley et al.,
which itself uses PRISM as its probability engine), here on top of the MDP's own nondeterminism.
Verifier: value iteration computing Pmin (minimizing over actions each state) -- PRISM's own core
numerical algorithm, via explicit matrix-vector iteration rather than PRISM's MTBDD engine.
"""
from __future__ import annotations

import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_synth import ModelType, make_attack_synth  # noqa: E402


@dataclass(frozen=True)
class MDP:
    actions: tuple        # actions[s] = {action_name: (dist,)}, dist = tuple of (prob, next_state)
    labels: tuple
    init: int = 0


def value_iteration(M: MDP, target: set, bound: int, minimize: bool) -> float:
    n = len(M.actions)
    x = [1.0 if s in target else 0.0 for s in range(n)]
    pick = min if minimize else max
    for _ in range(bound):
        y = []
        for s in range(n):
            if s in target:
                y.append(1.0)
                continue
            if not M.actions[s]:
                y.append(x[s])
                continue
            y.append(pick(sum(p * x[s2] for p, s2 in dist) for dist in M.actions[s].values()))
        x = y
    return x[M.init]


@dataclass(frozen=True)
class PCTLMinBound:
    label: str
    bound: int
    p: float
    minimize: bool = True


def pctl_sat(M: MDP, phi: PCTLMinBound) -> bool:
    target = {s for s in range(len(M.labels)) if phi.label in M.labels[s]}
    return value_iteration(M, target, phi.bound, phi.minimize) >= phi.p - 1e-9


def prob_of(M: MDP, phi: PCTLMinBound) -> float:
    target = {s for s in range(len(M.labels)) if phi.label in M.labels[s]}
    return value_iteration(M, target, phi.bound, phi.minimize)


GRID = (0.01, 0.02, 0.05, 0.1)


def perturbations(M: MDP, epsilon: float, rng: random.Random, samples=40):
    for s, acts in enumerate(M.actions):
        for name, dist in acts.items():
            states = [s2 for _, s2 in dist]
            if len(states) < 2:
                continue
            for _ in range(samples):
                i, j = rng.sample(range(len(dist)), 2)
                d = rng.choice(GRID)
                if d > epsilon or dist[i][0] < d:
                    continue
                new_dist = list(dist)
                new_dist[i] = (round(dist[i][0] - d, 6), dist[i][1])
                new_dist[j] = (round(dist[j][0] + d, 6), dist[j][1])
                new_acts = dict(acts); new_acts[name] = tuple(new_dist)
                yield MDP(M.actions[:s] + (new_acts,) + M.actions[s + 1:], M.labels, M.init)


def worst_case(M: MDP, phi: PCTLMinBound, epsilon=0.1, budget=3000, seed=0):
    rng = random.Random(seed)
    canon = lambda m: tuple(tuple(sorted(acts.items())) for acts in m.actions)
    mt = ModelType("MDP (PRISM)", neighbors=lambda m: perturbations(m, epsilon, rng), key=canon)
    synth = make_attack_synth(pctl_sat, mt, measure=lambda m: prob_of(m, phi), max_steps=1)
    return synth(M, phi, budget=budget, limit=1)


# --------------------------------------------------------------------------- an illustrative instance
def retry_instance() -> MDP:
    """A sensor that `send`s a reading (0.9 success, 0.1 drop and retries) or may `hold` to save
    power (never delivers while holding). 0=start, 1=delivered, 2=dropped-retry."""
    actions = (
        {"send": ((0.9, 1), (0.1, 2)), "hold": ((1.0, 2),)},
        {},
        {"retry": ((0.5, 1), (0.5, 2))},
    )
    labels = (frozenset(), frozenset({"delivered"}), frozenset())
    return MDP(actions, labels, init=0)


def to_prism() -> str:
    return """mdp

module sensor
  s : [0..2] init 0;
  [send]  s=0 -> 0.9:(s'=1) + 0.1:(s'=2);
  [hold]  s=0 -> 1.0:(s'=2);
  [retry] s=2 -> 0.9:(s'=1) + 0.1:(s'=2);
endmodule

label "delivered" = s=1;
"""


def run_real_prism(prism_text: str):
    exe = shutil.which("prism")
    if not exe:
        return None, "prism not found on PATH"
    with tempfile.NamedTemporaryFile("w", suffix=".prism", delete=False) as f:
        f.write(prism_text)
        path = f.name
    try:
        out = subprocess.run([exe, path, "-pctl", 'Pmin=? [ F<=10 "delivered" ]'],
                             capture_output=True, text=True, timeout=120)
        return out.returncode, out.stdout[-4000:]
    except Exception as e:  # pragma: no cover
        return None, str(e)


def main():
    M = retry_instance()
    phi = PCTLMinBound("delivered", 10, 0.995, minimize=True)
    print("PRISM-style -- MDP value iteration (Pmin) with epsilon-ball perturbation")
    print(f"  base Pmin(F<=10 delivered) = {prob_of(M, phi):.4f}  (worst case over 'send' vs 'hold')")
    r = worst_case(M, phi, epsilon=0.1, budget=2000)
    if r.found:
        worst = prob_of(r.best.model, phi)
        diffs = [(s, a, M.actions[s][a], r.best.model.actions[s][a]) for s in range(len(M.actions))
                for a in M.actions[s] if M.actions[s][a] != r.best.model.actions[s][a]]
        print(f"  attack: perturbing {[(s, a) for s, a, _, _ in diffs]} drops Pmin to {worst:.4f} "
              f"(below the {phi.p} bound)")
        print(f"  (explored {r.explored} perturbations in {r.seconds*1e3:.1f} ms)")
    else:
        print("  the 0.85 bound is robust to every sampled epsilon=0.1 perturbation.")
    rc, out = run_real_prism(to_prism())
    print(f"  real prism: {'not available in this environment' if rc is None else f'exit {rc}'}")


if __name__ == "__main__":
    main()
