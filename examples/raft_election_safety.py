"""
examples/raft_election_safety.py -- a generality stress test: attack synthesis against a
distributed-consensus safety property, in the style of the Raft protocol (Ongaro & Ousterhout, "In
Search of an Understandable Consensus Algorithm", USENIX ATC 2014).

Model: a 3-node cluster within a single term. Each node grants a vote to one or more candidates
this term: `grants = {node: frozenset of candidates it has voted for}`. Correct Raft behavior keeps
each set a singleton (a node votes once per term); a candidate with grants from a majority (>=2 of
3 nodes, counting its own self-vote) is the term's leader.
Property: Raft's own Election Safety property, stated in the paper as "at most one leader can be
elected in a given term" -- here, at most one candidate has majority support.
Adversary: Raft's own paper explains *why* a node must durably persist `votedFor` to stable storage
"before responding to RPCs" -- precisely "to prevent it from voting twice in the same term" after a
crash and restart. This instance's adversary models exactly the failure that requirement exists to
prevent: a node's persisted vote record is lost (e.g. an untouched storage fault), so when a second
candidate requests its vote later in the same term (as can happen after a network partition causes
an election timeout, per the same paper), the node grants it too -- without un-granting the first.
Modeling this as an *additional* grant (not a replacement) is what makes split-brain reachable here
at all: Raft's safety proof rests on any two majorities of an odd-sized cluster overlapping in at
least one node, so if votes instead partitioned the cluster (every node supports exactly one
candidate), 3 single-valued votes could never produce two simultaneous majorities of 2. The real
vulnerability is a node's vote failing to be exclusive, which is exactly what "persist votedFor
before replying" prevents.
Verifier: direct majority counting over `grants` -- exact, since the cluster is small and finite.

Like examples/adversarial_ml.py, this file is deliberately outside tools/ (Example 4) and papers/
(Example 5): distributed consensus is a model class neither names, offered as further evidence for
Section 6's generality claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_synth import Logic, ModelType, ThreatModel  # noqa: E402

NODES = ("A", "B", "C")
MAJORITY = 2   # out of 3

BASELINE = {"A": frozenset({"A"}), "B": frozenset({"A"}), "C": frozenset({"A"})}   # A is leader


def to_dict(frozen) -> dict:
    return dict(frozen)


def votes(grants: dict, candidate: str) -> int:
    return sum(1 for n, cands in grants.items() if candidate in cands)


def leaders(grants: dict):
    return [c for c in NODES if votes(grants, c) >= MAJORITY]


def election_safety_holds(grants: dict, _phi=None) -> bool:
    """Raft's Election Safety: at most one leader per term."""
    return len(leaders(grants)) <= 1


def double_vote_neighbors(frozen_grants):
    """The adversary's single edit: node n's lost persisted vote record lets it additionally grant
    a vote to candidate c (without retracting its earlier vote), while c simultaneously declares
    candidacy by granting itself a vote (an election timeout, per the paper, triggering c to start
    an election in this same term) -- one combined, single-edit realization of the scenario the
    paper's own persistence requirement exists to prevent."""
    grants = to_dict(frozen_grants)
    for n in NODES:
        for c in NODES:
            if c in grants[n]:
                continue   # n already granted c; not a new double-vote
            new_grants = dict(grants)
            new_grants[n] = grants[n] | {c}
            if c not in new_grants[c]:
                new_grants[c] = new_grants[c] | {c}   # c declares candidacy (self-vote)
            yield frozenset((node, cands) for node, cands in new_grants.items())


def margin_measure(frozen) -> float:
    """Severity weighting: 1.0 when exactly one leader (fully safe), decreasing as additional
    simultaneous leaders appear (a genuine split-brain is the worst case this 3-node cluster can
    reach: 2 leaders at once, since a third would need >=6 total votes from only 3 nodes)."""
    grants = to_dict(frozen)
    n_leaders = len(leaders(grants))
    if n_leaders == 0:
        return 1.0   # no leader yet is not a safety violation, just an ongoing election
    return 1.0 - (n_leaders - 1) / (len(NODES) - 1)


def build_threat_model() -> ThreatModel:
    verifier = lambda frozen, _phi: election_safety_holds(to_dict(frozen))
    mt = ModelType("Raft cluster vote-grant set", neighbors=double_vote_neighbors, key=lambda s: s)
    logic = Logic("Raft Election Safety (single term)", verifier)
    baseline = frozenset((n, cands) for n, cands in BASELINE.items())
    return ThreatModel(logic, m=baseline, phi=object(), model_type=mt, measure=margin_measure)


def main():
    print("Raft-style -- Election Safety under a lost/forgotten persisted vote")
    tm = build_threat_model()
    print(f"  baseline grants: {to_dict(tm.m)}  leaders: {leaders(to_dict(tm.m))}  "
          f"(Election Safety holds: {election_safety_holds(to_dict(tm.m))})")

    result = tm.synthesize(budget=200, limit=1)
    assert result.found, "expected a single double-vote edit to produce a second leader"
    a = result.best
    new_grants = to_dict(a.model)
    # Genuineness check, mirroring tests/test_conformance.py's TestAttackIsGenuine.
    assert tm.logic.verifier(tm.m, tm.phi)                         # m |= phi (precondition)
    assert not tm.logic.verifier(a.model, tm.phi)                  # m' |/= phi
    assert a.model in set(tm.candidate_space())                    # m ~> m'
    print(f"  attack found (single edit): {new_grants}")
    print(f"    leaders: {leaders(new_grants)}  "
          f"(Election Safety violated: split-brain with {len(leaders(new_grants))} simultaneous leaders)")
    print(f"    severity sigma = {a.sigma:.3f}  (explored {result.explored} candidate states "
          f"in {result.seconds * 1e3:.1f} ms)")
    print("  (no dual-goal demonstration here: phi is an opaque sentinel object, not a formula in "
          "the Atom/Not/Op AST, so structural_dual(phi) -- Definition 4 -- does not apply to it; "
          "see demos.py's LTL/CTL/etc. demos, or tools/margrave.py, for instances where phi is in "
          "that AST and goal=\"dual\" is meaningful.)")


if __name__ == "__main__":
    main()
