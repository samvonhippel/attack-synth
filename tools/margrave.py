"""
tools/margrave.py -- a native instance in the style of the Margrave firewall-policy analyzer
(Nelson, Barratt, Dougherty, Fisler & Krishnamurthi, "The Margrave Tool for Firewall Analysis",
USENIX LISA 2010, https://static.usenix.org/event/lisa10/tech/full_papers/Nelson.pdf).

Model: Margrave's own description -- "a policy as a mapping from requests to decisions," with
rules checked in order and the policy ending in `(RComb FAC)` ("first applicable"): the first rule
whose conditions match a request determines its decision. Each rule gets an R_matches formula
(does the rule's condition hold) and an R_applies formula (R_matches, and no earlier rule matched).
This instance models exactly that: an ordered tuple of (action, src-pattern, dst-pattern) rules,
evaluated first-match, always ending in an explicit catch-all (Margrave's own worked examples all
end in "deny any"/"permit any" -- the paper does not define what happens with no catch-all, so this
instance does not allow a ruleset without one).
Property: a fixed request's decision must be "deny" (Margrave's own EXPLORE queries ask exactly
this kind of question: "can a blacklisted address get through?").
Adversary: Margrave's own "change impact" analysis is defined as "scenarios illustrating packets
whose decision or applicable rule changes in the face of edits" -- i.e. the edits ARE the ~>
relation Margrave itself reasons about. This instance's single edits are exactly the firewall
misconfiguration vectors Margrave's own paper discusses rules in terms of: insert a rule, delete a
rule, flip a rule's action, or swap two adjacent rules (a reordering bug -- order is semantically
load-bearing under first-applicable matching, unlike every other tool in this repository).
Verifier: direct first-match evaluation over the rule tuple (no SAT solving -- Margrave compiles
requests/decisions to first-order relational formulas and solves with Kodkod/Alloy's solver
over a bounded universe; this instance evaluates ground requests directly since there is no need
to enumerate an unbounded request space for a single fixed query).

Unlike every other file in tools/, this instance is built through the paper-literal
Logic/ThreatModel/DualThreatModel/severity layer in attack_synth.py (rather than calling
make_attack_synth directly), precisely to demonstrate that layer -- not just the lower-level
engine -- extends to a model class invented after that layer was written.
"""
from __future__ import annotations

import itertools
import shutil
import subprocess
import sys
import tempfile
from typing import NamedTuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_synth import Atom, DualThreatModel, Logic, ModelType, Not, ThreatModel  # noqa: E402

SRC_CLASSES = ("blacklisted", "normal")
DST_CLASSES = ("public", "admin")
ALL_REQUESTS = tuple(itertools.product(SRC_CLASSES, DST_CLASSES))
REQUEST = ("blacklisted", "admin")           # the fixed query this threat model is about
INTENDED = {                                  # the "should be" decision for every request, used
    ("blacklisted", "public"): "deny",        # only by the severity measure below -- not by the
    ("blacklisted", "admin"): "deny",         # verifier, which only ever checks REQUEST
    ("normal", "public"): "permit",
    ("normal", "admin"): "permit",
}


class Rule(NamedTuple):
    action: str      # "permit" | "deny"
    src: str         # a member of SRC_CLASSES, or "*"
    dst: str         # a member of DST_CLASSES, or "*"


def decide(ruleset: tuple, req: tuple) -> str:
    """Margrave's (RComb FAC): the first rule whose conditions match wins."""
    src, dst = req
    for r in ruleset:
        if r.src in (src, "*") and r.dst in (dst, "*"):
            return r.action
    raise ValueError("no catch-all rule -- every ruleset in this instance must end in one")


def is_catchall(ruleset: tuple) -> bool:
    return bool(ruleset) and ruleset[-1].src == "*" and ruleset[-1].dst == "*"


BASELINE = (
    Rule("deny", "blacklisted", "*"),
    Rule("permit", "normal", "public"),
    Rule("permit", "normal", "admin"),
    Rule("deny", "*", "*"),
)

CANDIDATE_LIBRARY = (
    Rule("permit", "blacklisted", "admin"),
    Rule("permit", "blacklisted", "public"),
    Rule("permit", "blacklisted", "*"),
    Rule("permit", "*", "admin"),
)


def ruleset_edits(ruleset: tuple):
    """The four single-edit vectors: insert, delete, flip a rule's action, swap adjacent rules --
    exactly the edits Margrave's own change-impact analysis is designed to catch the effect of."""
    n = len(ruleset)
    for i in range(n + 1):
        for cand in CANDIDATE_LIBRARY:
            yield ruleset[:i] + (cand,) + ruleset[i:]
    for i in range(n):
        if n > 1:
            yield ruleset[:i] + ruleset[i + 1:]
    for i in range(n):
        flipped = "deny" if ruleset[i].action == "permit" else "permit"
        yield ruleset[:i] + (ruleset[i]._replace(action=flipped),) + ruleset[i + 1:]
    for i in range(n - 1):
        yield ruleset[:i] + (ruleset[i + 1], ruleset[i]) + ruleset[i + 1:]


def margrave_neighbors(ruleset: tuple):
    return (r for r in ruleset_edits(ruleset) if is_catchall(r))


def coverage(ruleset: tuple) -> float:
    """Severity measure mu: the fraction of ALL_REQUESTS whose decision still matches INTENDED --
    the same "which packets does this change affect" question Margrave's change-impact query asks,
    just phrased as a measure instead of a difference query."""
    return sum(decide(ruleset, req) == INTENDED[req] for req in ALL_REQUESTS) / len(ALL_REQUESTS)


def build_threat_model() -> ThreatModel:
    """phi = Atom("deny"): REQUEST's decision must be "deny". Built through the literal
    Logic/ThreatModel API (attack_synth.py's paper-conformance layer), not make_attack_synth
    directly -- see the module docstring."""
    verifier = lambda ruleset, phi: (
        decide(ruleset, REQUEST) == "deny" if isinstance(phi, Atom)
        else decide(ruleset, REQUEST) != "deny"   # phi = Not(Atom("deny"))
    )
    logic = Logic("Margrave ACL (first-applicable)", verifier)
    mt = ModelType("firewall ruleset", neighbors=margrave_neighbors, key=lambda r: r)
    return ThreatModel(logic, m=BASELINE, phi=Atom("deny"), model_type=mt,
                        edit_relation=None, measure=coverage, max_steps=1)


def to_ios_acl(ruleset: tuple) -> str:
    lines = ["! Cisco IOS-style ACL, closest real syntax to Margrave's own worked example"]
    for i, r in enumerate(ruleset, 1):
        action = "permit" if r.action == "permit" else "deny"
        lines.append(f"access-list 101 {action} ip {r.src if r.src != '*' else 'any'} "
                     f"{r.dst if r.dst != '*' else 'any'}")
    return "\n".join(lines) + "\n"


def run_real_margrave(policy_text: str):
    exe = shutil.which("margrave")
    if not exe:
        return None, ("margrave (the real analyzer; a Racket program, no standard CLI binary "
                       "name ships on PATH) not found/available in this environment")
    with tempfile.NamedTemporaryFile("w", suffix=".ios", delete=False) as f:
        f.write(policy_text)
        path = f.name
    try:
        out = subprocess.run([exe, path], capture_output=True, text=True, timeout=120)
        return out.returncode, out.stdout[-4000:]
    except Exception as e:  # pragma: no cover
        return None, str(e)


def main():
    print("Margrave-style -- ordered first-applicable firewall policy analysis")
    tm = build_threat_model()
    print(f"  baseline: decide(REQUEST={REQUEST}) = {decide(tm.m, REQUEST)!r}  "
          f"(coverage of intended decisions: {coverage(tm.m):.2f})")

    result = tm.synthesize(budget=2000, limit=1)
    if result.found:
        a = result.best
        mprime = a.model
        print(f"  attack found: edited ruleset makes decide(REQUEST) = {decide(mprime, REQUEST)!r}")
        print(f"    edited policy: {list(mprime)}")
        print(f"    severity sigma = {a.sigma:.2f} (coverage dropped from "
              f"{coverage(tm.m):.2f} to {coverage(mprime):.2f})")
        # genuineness check, mirroring tests/test_conformance.py's TestAttackIsGenuine:
        assert tm.logic.verifier(tm.m, tm.phi)             # m |= phi (precondition)
        assert not tm.logic.verifier(mprime, tm.phi)        # m' |/= phi
        assert mprime in set(margrave_neighbors(tm.m)) or any(
            mprime in set(margrave_neighbors(step)) for step in (tm.m,))  # m ~> m'
    else:
        print(f"  no single-edit attack found within budget (explored {result.explored})")

    dual = tm.dual()
    dr = dual.realized(budget=2000, limit=1)
    print(f"  dual threat model T^d: phi^d = {dual.phi_d!r} "
          f"(for a single literal invariant, 'violate' and 'dual' coincide, unlike the "
          f"compound-formula demos in demos.py) -- realized: {dr.found}")

    rc, out = run_real_margrave(to_ios_acl(tm.m))
    print(f"  real margrave: {'not available in this environment' if rc is None else f'exit {rc}'}")


if __name__ == "__main__":
    main()
