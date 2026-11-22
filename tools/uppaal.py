"""
tools/uppaal.py -- a native instance in the style of UPPAAL.

Model: a network of timed automata -- locations with invariants (clock constraints bounding how
long control may stay), edges with guards (clock constraints) and resets, synchronized over
channels -- UPPAAL's own description: "modeling, validation and verification of real-time systems
modeled as networks of timed automata" (Wikipedia, https://en.wikipedia.org/wiki/Uppaal_Model_Checker).
Unlike tools/hytech.py's variable-rate hybrid automaton, every clock here advances at the fixed
rate 1 (the standard Alur-Dill timed-automaton model UPPAAL implements), and two automata can
synchronize on a shared channel action.
Property: a reachability/safety query (UPPAAL's TCTL fragment includes plain reachability and
safety formulas) -- here, "the Alarm location is never reached."
Adversary: UPPAAL has no adversary built in; the natural edit, as with HyTech, is to a guard or
invariant threshold -- here, the task's own deadline, or the watchdog's timeout.
Verifier: UPPAAL's real engine explores *zones* (symbolic regions of clock-value space) on the fly;
no DBM/zone library is available here, so this instance discretizes time into fixed steps and
tracks exact clock values directly (exact, not an over-approximation, because every constant in
this instance is a multiple of the step size -- the standard "digitization" argument for timed
automata, not UPPAAL's own zone algorithm).
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

DT = 0.5


@dataclass(frozen=True)
class Params:
    min_duration: float
    max_duration: float     # the task's own deadline guard/invariant
    timeout: float           # the watchdog's timeout guard/invariant


def invariant_ok(loc, val, p: Params) -> bool:
    if loc == "Running":
        return val <= p.max_duration
    if loc == "Watching":
        return val <= p.timeout
    return True


def neighbors(state, p: Params):
    tl, wl, x, y = state
    if invariant_ok(tl, x + DT, p) and invariant_ok(wl, y + DT, p):
        yield (tl, wl, round(x + DT, 6), round(y + DT, 6))         # delay (both clocks tick at rate 1)
    if tl == "Idle" and wl == "Idle":                               # sync on channel 'start'
        yield ("Running", "Watching", 0.0, 0.0)
    if tl == "Running" and wl == "Watching" and x >= p.min_duration:  # sync on channel 'finish'
        yield ("Idle", "Idle", x, y)
    if wl == "Watching" and y >= p.timeout:                          # local: the watchdog times out
        yield (tl, "Alarm", x, y)


def find_attack(init: tuple, p: Params, budget=20000):
    """phi: Alarm is never reached. max_steps=None asks the core not to bound search depth at all
    (there is no natural fixed step bound here -- it depends on the step size DT and the
    parameters); `budget` is what actually stops the search."""
    verifier = lambda s, _phi: s[1] != "Alarm"
    mt = ModelType("timed-automata state", neighbors=lambda s: neighbors(s, p), key=lambda s: s)
    synth = make_attack_synth(verifier, mt, max_steps=None)
    return synth(init, object(), budget=budget, limit=1)


def rate_edit_style_candidates(p: Params, deltas=(1.0, 2.0, 4.0)):
    """The two edits UPPAAL's own table row describes: a guard/invariant threshold change (on
    either side of the race -- the task's deadline, or the watchdog's timeout)."""
    for d in deltas:
        yield f"max_duration += {d}", Params(p.min_duration, p.max_duration + d, p.timeout)
    for d in deltas:
        yield f"timeout -= {d}", Params(p.min_duration, p.max_duration, max(p.min_duration, p.timeout - d))


def to_uppaal_xml() -> str:
    return """<?xml version="1.0" encoding="utf-8"?>
<nta>
  <template><name>Task</name>
    <location id="Idle"/><location id="Running"><label kind="invariant">x &lt;= MAX</label></location>
    <init ref="Idle"/>
    <transition><source ref="Idle"/><target ref="Running"/><label kind="synchronisation">start!</label>
      <label kind="assignment">x = 0</label></transition>
    <transition><source ref="Running"/><target ref="Idle"/><label kind="synchronisation">finish!</label>
      <label kind="guard">x &gt;= MIN</label></transition>
  </template>
  <template><name>Watchdog</name>
    <location id="Idle"/><location id="Watching"><label kind="invariant">y &lt;= TIMEOUT</label></location>
    <location id="Alarm"/>
    <init ref="Idle"/>
    <transition><source ref="Idle"/><target ref="Watching"/><label kind="synchronisation">start?</label>
      <label kind="assignment">y = 0</label></transition>
    <transition><source ref="Watching"/><target ref="Idle"/><label kind="synchronisation">finish?</label></transition>
    <transition><source ref="Watching"/><target ref="Alarm"/><label kind="guard">y &gt;= TIMEOUT</label></transition>
  </template>
  <system>Task, Watchdog</system>
  <queries><query><formula>A[] not Watchdog.Alarm</formula></query></queries>
</nta>
"""


def run_real_uppaal(xml_text: str):
    exe = shutil.which("verifyta")
    if not exe:
        return None, "verifyta (UPPAAL's command-line verifier) not found on PATH"
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml_text)
        path = f.name
    try:
        out = subprocess.run([exe, path], capture_output=True, text=True, timeout=120)
        return out.returncode, out.stdout[-4000:]
    except Exception as e:  # pragma: no cover
        return None, str(e)


def main():
    p = Params(min_duration=1.0, max_duration=4.0, timeout=6.0)
    init = ("Idle", "Idle", 0.0, 0.0)
    print("UPPAAL-style -- timed-automata reachability (rate-1 clocks, channel sync)")
    r0 = find_attack(init, p)
    print(f"  baseline (max_duration={p.max_duration}, timeout={p.timeout}): Alarm reachable = {r0.found}")
    found = None
    for name, p2 in rate_edit_style_candidates(p):
        r = find_attack(init, p2, budget=20000)
        if r.found:
            found = (name, p2, r)
            break
    if found:
        name, p2, r = found
        print(f"  attack: edit '{name}' makes Alarm reachable after {len(r.best.path) - 1} steps "
              f"(explored {r.explored} states in {r.seconds * 1e3:.1f} ms, unbounded depth, budget-limited search)")
    else:
        print("  no single guard/timeout edit from the tried deltas reaches Alarm.")
    rc, out = run_real_uppaal(to_uppaal_xml())
    print(f"  real uppaal (verifyta): {'not available in this environment' if rc is None else f'exit {rc}'}")


if __name__ == "__main__":
    main()
