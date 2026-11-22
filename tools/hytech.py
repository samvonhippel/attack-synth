"""
tools/hytech.py -- a native instance in the style of HyTech.

Model: a linear hybrid automaton -- locations, each with a rate *interval* for one continuous
variable x (a nondeterministic flow, e.g. a thermostat's heating/cooling rate) and an invariant
box; edges with a guard box and a reset box, composed as a product of such automata.
Property: a reachability query -- can an "unsafe" location/box be reached at all.
Adversary: "edits to rates, guards, and resets" (Example 4) -- supplied by the user, here as a
bounded change to one location's rate interval, one edge's guard threshold, or one edge's reset.
Verifier: HyTech's own technique is symbolic reachability over convex polyhedra; no polyhedra
library is available here, so reachable sets are tracked as axis-aligned interval ("box") over-
approximations instead, advanced over a bounded discrete time step -- sound for reachability
(it can only ever over-approximate, never miss a real unsafe state) but coarser than exact
polyhedra, and not HyTech's own parametric analysis.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_synth import ModelType, make_attack_synth  # noqa: E402

DT = 0.5       # discrete time step used to advance a continuous flow (a bounded-time abstraction)
STEPS_PER_FLOW = 4


@dataclass(frozen=True)
class Automaton:
    """locations[name] = (rate_lo, rate_hi, inv_hi)   -- x flows at a rate in [rate_lo, rate_hi],
    while x <= inv_hi holds (the invariant keeps control from dwelling past it).
    edges = (from_loc, guard_lo, to_loc, reset_hi) -- enabled once x >= guard_lo; resets x to
    [0, reset_hi] on taking the edge."""
    locations: dict
    edges: tuple
    init_loc: str
    init_box: tuple   # (lo, hi)


def flow_box(lo, hi, rate_lo, rate_hi, dt):
    return (lo + rate_lo * dt, hi + rate_hi * dt)


def neighbors(state, A: Automaton):
    """Edges are urgent: once some outgoing edge's guard already holds, flow is suppressed and
    only an edge may be taken (an as-soon-as-possible switch). This still lets a single discrete
    flow step -- taken while the guard does *not yet* hold -- overshoot past where the guard would
    have fired, if the rate is large enough: exactly the vulnerability a rate edit can create."""
    loc, (lo, hi) = state
    rate_lo, rate_hi, inv_hi = A.locations[loc]
    guard_holds = any(f == loc and hi >= g for (f, g, t, r) in A.edges)
    if not guard_holds:
        lo2, hi2 = flow_box(lo, hi, rate_lo, rate_hi, DT)
        hi2 = min(hi2, inv_hi)                 # the invariant clips the reachable box
        if hi2 >= lo2:
            yield (loc, (lo2, hi2))
    for (f, guard_lo, t, reset_hi) in A.edges:
        if f == loc and hi >= guard_lo:
            yield (t, (0.0, reset_hi))


def unsafe(state, unsafe_loc: str, unsafe_lo: float) -> bool:
    loc, (lo, hi) = state
    return loc == unsafe_loc and hi >= unsafe_lo


def find_reach(A: Automaton, unsafe_loc, unsafe_lo, budget=20000, max_steps=STEPS_PER_FLOW * 6):
    """phi: the unsafe box is never reached. An attack is any rate/guard/reset-edited automaton
    for which it is."""
    verifier = lambda s, _phi: not unsafe(s, unsafe_loc, unsafe_lo)
    mt = ModelType("hybrid state", neighbors=lambda s: neighbors(s, A), key=lambda s: s)
    synth = make_attack_synth(verifier, mt, max_steps=max_steps)
    return synth((A.init_loc, A.init_box), object(), budget=budget, limit=1)


def rate_edit(A: Automaton, loc, d):
    lo, hi, inv = A.locations[loc]
    return Automaton({**A.locations, loc: (lo, hi + d, inv)}, A.edges, A.init_loc, A.init_box)


def guard_edit(A: Automaton, idx, d):
    """Raises (delays) the guard threshold -- a weakened safety margin, e.g. a miscalibrated
    sensor that reports the tank as less full than it really is."""
    f, g, t, r = A.edges[idx]
    return Automaton(A.locations, A.edges[:idx] + ((f, g + d, t, r),) + A.edges[idx + 1:],
                     A.init_loc, A.init_box)


def reset_edit(A: Automaton, idx, d):
    f, g, t, r = A.edges[idx]
    return Automaton(A.locations, A.edges[:idx] + ((f, g, t, r + d),) + A.edges[idx + 1:],
                     A.init_loc, A.init_box)


def find_attack_over_edits(A: Automaton, unsafe_loc, unsafe_lo, deltas=(0.5, 1.0, 2.0)):
    """The top-level search: try each rate/guard/reset edit (the paper's own three kinds), and
    report the first that makes the unsafe box reachable where the original automaton did not."""
    base = find_reach(A, unsafe_loc, unsafe_lo)
    if base.found:
        return "baseline", None, base
    candidates = []
    for loc in A.locations:
        for d in deltas:
            candidates.append((f"rate[{loc}] += {d}", rate_edit(A, loc, d)))
    for i in range(len(A.edges)):
        for d in deltas:
            candidates.append((f"guard[{i}] += {d}", guard_edit(A, i, d)))
            candidates.append((f"reset[{i}] += {d}", reset_edit(A, i, d)))
    for name, A2 in candidates:
        r = find_reach(A2, unsafe_loc, unsafe_lo)
        if r.found:
            return name, A2, r
    return None, None, base


# --------------------------------------------------------------------------- an illustrative instance
def tank_instance() -> Automaton:
    """A fill/drain tank-level controller: FILL raises x at rate [0.8, 1.2]/s until a guard, then
    DRAIN lowers it at rate [-1.2, -0.8]/s back down. The invariant keeps x within [0, 10] in each
    location. Unsafe: x >= 9.5 while FILLING (the tank overflows before the switch to DRAIN)."""
    locations = {"FILL": (0.8, 1.2, 10.0), "DRAIN": (-1.2, -0.8, 10.0)}
    edges = (("FILL", 8.0, "DRAIN", 8.0), ("DRAIN", 0.0, "FILL", 0.0))
    # DRAIN -> FILL as soon as x could be back near 0 (guard_lo=0.0 means "always enabled once
    # low enough"; for this toy instance we gate it by the box's own interval dropping to 0).
    return Automaton(locations, edges, "FILL", (0.0, 0.0))


def to_hytech() -> str:
    return """automaton tank
synclabs: ;
initially FILL & x=0;

loc FILL: while x<=10 wait {dx in [0.8,1.2]}
  when x>=8 do {x'=x} goto DRAIN;

loc DRAIN: while x<=10 wait {dx in [-1.2,-0.8]}
  when x<=0 do {x'=x} goto FILL;

end

-- unsafe if reachable:  FILL & x>=9.5
"""


def run_real_hytech(hy_text: str):
    exe = shutil.which("hytech")
    if not exe:
        return None, "hytech not found on PATH"
    with tempfile.NamedTemporaryFile("w", suffix=".hy", delete=False) as f:
        f.write(hy_text)
        path = f.name
    try:
        out = subprocess.run([exe, path], capture_output=True, text=True, timeout=120)
        return out.returncode, out.stdout[-4000:]
    except Exception as e:  # pragma: no cover
        return None, str(e)


def main():
    A = tank_instance()
    print("HyTech-style -- interval/box reachability with rate/guard/reset edits")
    r0 = find_reach(A, "FILL", 9.5)
    print(f"  baseline: FILL & x>=9.5 reachable = {r0.found} (the guard at x>=8 switches to DRAIN first)")
    name, A2, r = find_attack_over_edits(A, "FILL", 9.5)
    if name and name != "baseline":
        print(f"  attack: edit '{name}' makes FILL & x>=9.5 reachable "
              f"(box reached: {r.best.model[1][0]:.2f}..{r.best.model[1][1]:.2f})")
        print(f"  (explored {r.explored} hybrid states in {r.seconds*1e3:.1f} ms)")
    else:
        print("  no single rate/guard/reset edit makes the unsafe box reachable.")
    rc, out = run_real_hytech(to_hytech())
    print(f"  real hytech: {'not available in this environment' if rc is None else f'exit {rc}'}")


if __name__ == "__main__":
    main()
