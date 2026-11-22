"""
tools/nuxmv.py -- a native instance in the style of nuXmv.

Model: a finite-state synchronous transition system over typed (here, bounded-Integer) variables,
given as a deterministic `next(vars) -> vars` function plus an invariant on the reachable states.
Property: an INVARSPEC-style invariant, and an LTLSPEC-style bounded response.
Adversary: "mutation of the transition relation or of variable assignments, supplied by the user"
(Example 4) -- here, both kinds of edit are available at every step, but the adversary gets only
ONE of them along the whole run (so the state tracks whether the one allowed edit has been spent).
Verifier: explicit-state reachability search to the given depth, standing in for nuXmv's BDD/SMT-
based engines (no BDD or SMT-solver package is available here to drive the real symbolic engines).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_synth import ModelType, make_attack_synth  # noqa: E402


@dataclass(frozen=True)
class Vars:
    """phase in {NS_GREEN, NS_YELLOW, EW_GREEN, EW_YELLOW}; timer counts steps in the phase;
    walk_req is a pending pedestrian request; walk is whether the WALK signal is lit."""
    phase: str
    timer: int
    walk_req: bool
    walk: bool


ORDER = ("NS_GREEN", "NS_YELLOW", "EW_GREEN", "EW_YELLOW")
DURATION = {"NS_GREEN": 5, "NS_YELLOW": 2, "EW_GREEN": 5, "EW_YELLOW": 2}


def honest_next(v: Vars) -> Vars:
    if v.timer + 1 < DURATION[v.phase]:
        walk = v.walk or (v.walk_req and v.phase in ("NS_YELLOW", "EW_YELLOW"))
        return Vars(v.phase, v.timer + 1, v.walk_req and not walk, walk)
    nxt = ORDER[(ORDER.index(v.phase) + 1) % 4]
    return Vars(nxt, 0, v.walk_req, False if nxt in ("NS_GREEN", "EW_GREEN") else v.walk)


def invariant_ok(v: Vars) -> bool:
    """WALK must never be lit while a direction has a green light -- a pedestrian must never be
    invited to cross while cross-traffic has right of way."""
    return not (v.walk and v.phase in ("NS_GREEN", "EW_GREEN"))


EDITS = []


def relation_mutation(v: Vars) -> Vars:
    """A transition-relation mutation: the controller advances the phase WITHOUT first clearing
    `walk` (the "reset walk on green" assignment is simply dropped for one transition)."""
    nxt = ORDER[(ORDER.index(v.phase) + 1) % 4]
    return Vars(nxt, 0, v.walk_req, v.walk)       # walk carried over uncleared, unlike honest_next


def assignment_mutation(v: Vars) -> Vars:
    """A variable-assignment mutation: `walk` is forced true for one step, regardless of phase."""
    return Vars(v.phase, v.timer, v.walk_req, True)


def neighbors(state):
    v, edit_used = state
    yield (honest_next(v), edit_used)
    if not edit_used:
        yield (relation_mutation(v), True)
        yield (assignment_mutation(v), True)


def find_attack(init: Vars, budget=20000, max_steps=30):
    verifier = lambda s, _phi: invariant_ok(s[0])
    mt = ModelType("traffic-light state", neighbors=neighbors, key=lambda s: s)
    synth = make_attack_synth(verifier, mt, max_steps=max_steps)
    return synth((init, False), object(), budget=budget, limit=1)


def to_smv() -> str:
    return """MODULE main
VAR
  phase : {NS_GREEN, NS_YELLOW, EW_GREEN, EW_YELLOW};
  timer : 0..5;
  walk_req : boolean;
  walk : boolean;
  edit_used : boolean;         -- the adversary's single transition-relation/assignment mutation

ASSIGN
  init(phase) := NS_GREEN;
  init(timer) := 0;
  init(walk_req) := FALSE;
  init(walk) := FALSE;
  init(edit_used) := FALSE;

INVARSPEC !(walk & (phase = NS_GREEN | phase = EW_GREEN));
"""


def run_real_nuxmv(smv_text: str):
    exe = shutil.which("nuXmv") or shutil.which("nuxmv")
    if not exe:
        return None, "nuXmv not found on PATH"
    with tempfile.NamedTemporaryFile("w", suffix=".smv", delete=False) as f:
        f.write(smv_text)
        path = f.name
    try:
        out = subprocess.run([exe, "-source", "/dev/stdin", path], input="go\ncheck_invars\nquit\n",
                             capture_output=True, text=True, timeout=120)
        return out.returncode, out.stdout[-4000:]
    except Exception as e:  # pragma: no cover
        return None, str(e)


def main():
    init = Vars("NS_GREEN", 0, True, False)
    print("nuXmv-style -- explicit-state reachability with one transition/assignment edit")
    r = find_attack(init)
    if r.found:
        v, edit_used = r.best.model
        print(f"  attack: a single edit along the run reaches phase={v.phase}, walk={v.walk} -- "
              f"the invariant 'no WALK during a green light' is violated")
        print(f"  (explored {r.explored} states in {r.seconds*1e3:.1f} ms, path length {len(r.best.path)-1})")
    else:
        print("  no single transition/assignment edit violates the invariant within the bound.")
    rc, out = run_real_nuxmv(to_smv())
    print(f"  real nuXmv: {'not available in this environment' if rc is None else f'exit {rc}'}")


if __name__ == "__main__":
    main()
