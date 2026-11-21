"""
papers/vonhippel.py -- M. von Hippel, C. Vick, S. Tripakis, C. Nita-Rotaru, "Automated Attacker
Synthesis for Distributed Protocols" (arXiv:2004.01220).

Model: a composition  P || Q_0 || ... || Q_n  of Kripke-structure processes with inputs and
outputs, composed asynchronously by rendezvous on shared actions.
Property: an LTL formula phi, with  P || Q || ... |= phi  before the attack.
Adversary ~> : replaces a vulnerable component Q_i by a deterministic process A_i over the *same*
interface (same inputs/outputs), giving  P || Q || ... ~> P || A || ....
Attack: an A with  P || A || ... |/= phi  (an exists-attacker if it succeeds on at least one run).

The paper's algorithm (Sec. 4) is: replace every Q_i by DAISY(Q_i), the maximally nondeterministic
process with that same interface (able to emit or accept any of Q_i's declared actions at every
step); model-check P || DAISY(Q) against phi (the paper uses SPIN); then PROJECT the resulting
counterexample's DAISY actions onto each Q_i's interface to obtain the concrete attacker A_i.

That projection step is exactly a search over which of DAISY's free choices to resolve, at each
step, into a violating run -- which is the same search attack_synth already does for the LTL demo
in demos.py, just with a protocol-specific "neighbors" generator instead of a free trace mutator.
So this instance reuses attack_synth directly: a "model" is a finite trace prefix, phi is checked
as a bounded-response safety monitor over that prefix (an under-approximation of general LTL, but
exactly what the paper's handshake-style properties need), and model_type.neighbors enumerates the
actions DAISY(Q) could add next, synchronized with P's own deterministic transition.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_synth import Atom, ModelType, make_attack_synth  # noqa: E402


@dataclass(frozen=True)
class Process:
    """A deterministic Kripke-structure process with inputs and outputs: `trans[state][action] ->
    next_state`. `alphabet` is every action it can take in *some* state (its interface)."""
    states: tuple
    init: str
    trans: dict                  # {(state, action): next_state}
    alphabet: frozenset

    def step(self, state, action):
        return self.trans.get((state, action))


def daisy_choices(Q: Process, _state) -> frozenset:
    """DAISY(Q): able to offer any action in Q's interface at every step, regardless of Q's own
    state -- the maximally nondeterministic process with the same alphabet (Def. in Sec. 4)."""
    return Q.alphabet


@dataclass(frozen=True)
class BoundedResponse:
    """phi = G(req -> F<=k ack): every `req` is followed by an `ack` within k actions. This is the
    fragment of LTL the paper's handshake-style properties need; it is checked prefix-monotonically
    (a prefix "violates" phi as soon as some req is more than k steps old with no ack since)."""
    req: str
    ack: str
    k: int


def monitor_violated(trace: tuple, phi: BoundedResponse) -> bool:
    last_req = None
    for i, a in enumerate(trace):
        if a == phi.req:
            last_req = i
        elif a == phi.ack:
            last_req = None
        if last_req is not None and i - last_req >= phi.k:
            return True
    return False


def verifier(trace: tuple, phi: BoundedResponse) -> bool:
    return not monitor_violated(trace, phi)


def make_trace_model_type(P: Process, Q: Process, shared: frozenset) -> ModelType:
    """A 'model' is (p_state, trace-so-far). An edit extends the trace by one action that DAISY(Q)
    could offer AND that P can take from its current state (synchronous rendezvous on the shared
    alphabet; actions outside `shared` are P-only and always available if P itself enables them)."""
    def neighbors(cur):
        p_state, trace = cur
        offers = daisy_choices(Q, None) | (P.alphabet - shared)   # DAISY's offers, plus P's own-only actions
        for a in offers:
            nxt = P.step(p_state, a)
            if nxt is not None:
                yield (nxt, trace + (a,))
    return ModelType("P || DAISY(Q) trace", neighbors=neighbors, key=lambda c: c[1])


def synthesize_attacker(P: Process, Q: Process, shared: frozenset, phi: BoundedResponse, horizon: int, budget=20000):
    """Returns (SynthResult, attacker_trace | None). `attacker_trace` is the projection of the
    counterexample onto Q's interface -- the finite-string attacker A_i the paper's KORG tool would
    synthesize for this witness (a trivial one-path DFA replaying exactly these actions)."""
    mt = make_trace_model_type(P, Q, shared)
    vphi = lambda cur, f: verifier(cur[1], f)
    synth = make_attack_synth(vphi, mt, max_steps=horizon)
    r = synth((P.init, ()), phi, budget=budget, limit=1)
    if not r.found:
        return r, None
    full_trace = r.best.model[1]
    attacker_trace = tuple(a for a in full_trace if a in Q.alphabet)   # project onto Q's interface
    return r, attacker_trace


# --------------------------------------------------------------------------- an illustrative instance
def handshake_instance():
    """A simplified request/acknowledge handshake in the spirit of the paper's TCP connection-setup
    case study (not the literal TCP automaton, which the source available here does not give):
    a client P sends `req`, expects `ack` within k=3 actions, and may retry once on `timeout`."""
    P = Process(
        states=("idle", "wait1", "wait2", "done"),
        init="idle",
        trans={
            ("idle", "req"): "wait1",
            ("wait1", "ack"): "done",
            ("wait1", "timeout"): "wait2",
            ("wait1", "nop"): "wait1",
            ("wait2", "ack"): "done",
            ("wait2", "nop"): "wait2",
            ("done", "nop"): "done",
        },
        alphabet=frozenset({"req", "ack", "timeout", "nop"}),
    )
    # Q is the real responder: nominally always acks promptly. The attacker replaces it with
    # DAISY(Q), which may also legally emit timeout/nop (both in Q's declared interface) instead.
    Q = Process(states=("resp",), init="resp",
               trans={("resp", "ack"): "resp", ("resp", "timeout"): "resp", ("resp", "nop"): "resp"},
               alphabet=frozenset({"ack", "timeout", "nop"}))
    shared = frozenset({"ack", "timeout", "nop"})
    phi = BoundedResponse(req="req", ack="ack", k=3)
    return P, Q, shared, phi


def main():
    P, Q, shared, phi = handshake_instance()
    print("von Hippel et al. -- DAISY-replacement attacker synthesis")
    print(f"  phi: every '{phi.req}' is followed by '{phi.ack}' within {phi.k} actions")
    r, attacker = synthesize_attacker(P, Q, shared, phi, horizon=6)
    print(f"  explored {r.explored} trace extensions in {r.seconds*1e3:.1f} ms")
    if r.found:
        full = r.best.model[1]
        print(f"  violating run of P || DAISY(Q): {full}")
        print(f"  synthesized exists-attacker A (DAISY's choices projected onto Q's interface): {attacker}")
        print("  i.e. A always chooses 'timeout'/'nop' and never 'ack' -- a silent responder that never acknowledges.")
    else:
        print("  no attacker found within the horizon: P || DAISY(Q) |= phi for every resolution tried.")


if __name__ == "__main__":
    main()
