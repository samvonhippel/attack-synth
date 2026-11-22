"""
tools/spin.py -- a native instance in the style of the SPIN model checker.

Model: Promela-style processes over shared variables, composed by nondeterministic interleaving
(exactly one enabled process steps at a time -- "execution interleaves enabled statements").
Property: an invariant (SPIN's "assertions, invalid end states"), checked here as a state
predicate rather than full never-claim LTL.
Adversary: SPIN has none built in ("scheduler perturbations ... modeled as ordinary processes",
per Example 4) -- here, ADDING one extra process to the composition.
Verifier: exhaustive depth-first search over reachable interleavings, exactly as SPIN does; no
partial-order reduction or bitstate hashing (this toy system is small enough not to need them).

`to_promela()` renders the instance as real Promela; `run_real_spin()` shells out to the installed
`spin` binary if one is found (none is, in this environment).
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
class Process:
    """`steps[pc]` is a list of (guard, pc', effect) triples: a transition is enabled from `pc`
    when `guard(vars)` holds, and moves to `pc'` while applying `effect(vars) -> new_vars`."""
    name: str
    init_pc: str
    steps: dict


def enabled(proc: Process, pc, shared: dict):
    for guard, pc2, effect in proc.steps.get(pc, []):
        if guard(shared):
            yield pc2, effect(dict(shared))


def interleavings(state, procs: tuple):
    """state = (pcs: tuple, shared: tuple of sorted (var, value) items) -- both hashable, so this
    doubles as the model's own `key`. Exactly one enabled process steps at a time."""
    pcs, shared_items = state
    shared = dict(shared_items)
    for i, proc in enumerate(procs):
        for pc2, shared2 in enabled(proc, pcs[i], shared):
            yield (pcs[:i] + (pc2,) + pcs[i + 1:], tuple(sorted(shared2.items())))


def mutex_invariant(state, pcs_in_cs: frozenset):
    pcs, _ = state
    return sum(1 for pc in pcs if pc in pcs_in_cs) <= 1


def find_attack(init_state, honest_procs: tuple, rogue: Process, invariant, budget=20000, max_steps=12):
    """phi: the invariant holds. An attack is adding `rogue` to the composition, after which some
    reachable interleaving violates it."""
    all_procs = honest_procs + (rogue,)
    verifier = lambda s, _phi: invariant(s)
    mt = ModelType("global state", neighbors=lambda s: interleavings(s, all_procs), key=lambda s: s)
    base = (init_state[0] + (rogue.init_pc,), init_state[1])   # rogue present in the state, but its
    synth = make_attack_synth(verifier, mt, max_steps=max_steps)  # first step is itself the "edit"
    return synth(base, object(), budget=budget, limit=1)


# --------------------------------------------------------------------------- an illustrative instance
def mutex_instance():
    """Two honest processes P1, P2 share `turn in {1, 2}` and correctly alternate: process i may
    enter its critical section only when turn == i, and hands `turn` to the other on exit. A rogue
    process P3 (the adversary's addition) can set `turn` to 1 at any time, letting P1 and P2 both
    see turn == themselves and both enter the critical section."""
    def honest(pid, other):
        return Process(f"P{pid}", "idle", {
            "idle": [(lambda sh, pid=pid: sh["turn"] == pid, "cs", lambda sh: sh)],
            "cs": [(lambda sh: True, "idle", lambda sh, other=other: {**sh, "turn": other})],
        })
    P1, P2 = honest(1, 2), honest(2, 1)
    rogue = Process("P3", "idle", {"idle": [(lambda sh: True, "idle", lambda sh: {**sh, "turn": 1})]})
    init = (("idle", "idle"), (("turn", 1),))
    invariant = lambda s: mutex_invariant(s, frozenset({"cs"}))
    return init, (P1, P2), rogue, invariant


def to_promela() -> str:
    return """mtype = { P1, P2 };
int turn = 1;
bool in_cs[2];

proctype Honest(int id; int other) {
  do
  :: turn == id ->
       in_cs[id-1] = true;
       assert(!(in_cs[0] && in_cs[1]));
       in_cs[id-1] = false;
       turn = other
  od
}

proctype Rogue() {          /* the adversary's added process */
  do
  :: turn = 1
  od
}

init {
  run Honest(1, 2);
  run Honest(2, 1);
  run Rogue();
}
"""


def run_real_spin(promela_text: str):
    exe = shutil.which("spin")
    if not exe:
        return None, "spin not found on PATH"
    with tempfile.NamedTemporaryFile("w", suffix=".pml", delete=False) as f:
        f.write(promela_text)
        path = f.name
    try:
        out = subprocess.run([exe, "-a", path], capture_output=True, text=True, timeout=120)
        return out.returncode, out.stdout[-4000:]
    except Exception as e:  # pragma: no cover
        return None, str(e)


def main():
    init, honest, rogue, invariant = mutex_instance()
    print("SPIN-style -- interleaving search with an added process")
    print(f"  honest system alone satisfies mutual exclusion: {invariant(init)} (trivially, P3 absent)")
    r = find_attack(init, honest, rogue, invariant)
    if r.found:
        print(f"  attack: adding process {rogue.name} reaches a state violating mutual exclusion "
              f"after {len(r.best.path) - 1} interleaved steps")
        print(f"  violating state: pcs={r.best.model[0]}, shared={dict(r.best.model[1])}")
        print(f"  (explored {r.explored} interleavings in {r.seconds*1e3:.1f} ms)")
    else:
        print("  no interleaving with the rogue process violates mutual exclusion within the bound.")
    rc, out = run_real_spin(to_promela())
    print(f"  real spin: {'not available in this environment' if rc is None else f'exit {rc}'}")


if __name__ == "__main__":
    main()
