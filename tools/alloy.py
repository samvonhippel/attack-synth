"""
tools/alloy.py -- a native instance in the style of the Alloy Analyzer.

Model: a *relational structure* over a finite scope -- fixed sets of atoms (here, Users and Files)
and relations over them (here, `owner`, `sharedWith`, and the free relation `canAccess`), rather
than a dynamical system. Alloy's own description: "Signatures define the vocabulary of a model by
creating new sets," and relations are declared fields over them (Wikipedia,
https://en.wikipedia.org/wiki/Alloy_(specification_language)). No fact in this instance constrains
`canAccess` at all -- a classic Alloy footgun: forgetting to pin a derived relation down to exactly
what it should be leaves room for unintended instances.
Property: an assertion -- every (u, f) in `canAccess` is justified by `owner` or `sharedWith`.
Adversary: Alloy has no dynamic adversary notion at all (per the source above, it says nothing of
an attacker or threat model). Its `check` command instead asks whether EVERY instance satisfying
the facts also satisfies the assertion. That makes the natural ⇝ here the *universal* relation over
every instance satisfying the facts (not an incremental edit): m ⇝ m' for any two valid instances
m, m' of the same signatures and facts. An attack is then exactly an Alloy counterexample: a valid
instance m' that violates the assertion.
Verifier: exhaustive enumeration over the bounded scope -- the search problem is the same one
Alloy's own "fully automated analysis" solves by "translat[ing] relational logic into boolean logic
and invok[ing] a SAT solver" (same source); this instance enumerates directly since the scope here
is tiny, instead of compiling to CNF. Like Alloy's own "small scope hypothesis," finding no
counterexample within the scope is not a proof that none exists at a larger scope.
"""
from __future__ import annotations

import itertools
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_synth import ModelType, make_attack_synth  # noqa: E402

USERS = ("u1", "u2")
FILES = ("f1", "f2")
OWNER = {"f1": "u1", "f2": "u2"}
SHARED_WITH = {"f1": {"u2"}, "f2": set()}
ALL_PAIRS = tuple(itertools.product(USERS, FILES))


def assertion_holds(can_access: frozenset) -> bool:
    """Every (u, f) in canAccess is justified by ownership or sharing."""
    return all(u == OWNER[f] or u in SHARED_WITH[f] for u, f in can_access)


def intended_instance() -> frozenset:
    return frozenset((u, f) for u, f in ALL_PAIRS if u == OWNER[f] or u in SHARED_WITH[f])


def all_instances():
    """Every subset of ALL_PAIRS is a valid instance: no fact constrains canAccess at all."""
    n = len(ALL_PAIRS)
    for bits in itertools.product((False, True), repeat=n):
        yield frozenset(p for p, b in zip(ALL_PAIRS, bits) if b)


def find_counterexample(budget=1000):
    """phi: the assertion holds. An attack is any valid instance (of which every subset of
    ALL_PAIRS is one) that violates it."""
    verifier = lambda m, _phi: assertion_holds(m)
    mt = ModelType("Alloy instance (canAccess)", neighbors=lambda _m: all_instances(), key=lambda m: m)
    synth = make_attack_synth(verifier, mt, max_steps=1)
    return synth(intended_instance(), object(), budget=budget, limit=1)


def to_als() -> str:
    return """sig User {}
sig File {
  owner: one User,
  sharedWith: set User,
  canAccess: set User     -- left unconstrained: the bug this instance finds
}

assert NoUnauthorizedAccess {
  all f: File, u: f.canAccess | u = f.owner or u in f.sharedWith
}

check NoUnauthorizedAccess for 2 but 2 User, 2 File
"""


def run_real_alloy(als_text: str):
    exe = shutil.which("alloy")
    if not exe:
        return None, "alloy (the command-line analyzer) not found on PATH"
    with tempfile.NamedTemporaryFile("w", suffix=".als", delete=False) as f:
        f.write(als_text)
        path = f.name
    try:
        out = subprocess.run([exe, path], capture_output=True, text=True, timeout=120)
        return out.returncode, out.stdout[-4000:]
    except Exception as e:  # pragma: no cover
        return None, str(e)


def main():
    print("Alloy-style -- bounded relational-instance enumeration")
    print(f"  intended canAccess: {sorted(intended_instance())}")
    print(f"  assertion holds for the intended instance: {assertion_holds(intended_instance())}")
    r = find_counterexample()
    if r.found:
        extra = sorted(r.best.model - intended_instance())
        missing = sorted(intended_instance() - r.best.model)
        print(f"  counterexample instance found: canAccess = {sorted(r.best.model)}")
        print(f"  unauthorized pairs present: {extra}" + (f"; legitimate pairs dropped: {missing}" if missing else ""))
        print(f"  (enumerated {r.explored} of {2 ** len(ALL_PAIRS)} instances in the scope)")
    else:
        print("  no counterexample within this scope -- but see the small-scope-hypothesis caveat above.")
    rc, out = run_real_alloy(to_als())
    print(f"  real alloy: {'not available in this environment' if rc is None else f'exit {rc}'}")


if __name__ == "__main__":
    main()
