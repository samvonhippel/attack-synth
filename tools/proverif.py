"""
tools/proverif.py -- a native instance in the style of ProVerif.

Model: a set of Horn clauses `head :- body` over ground terms, whose least fixpoint is the set of
facts the attacker predicate att(.) can derive -- this *is* how ProVerif itself works internally
("protocol-level constraints" are abstracted away so that clause resolution covers unboundedly
many sessions at once, per Example 4), unlike Tamarin's ground, per-trace constraint search in
tools/tamarin.py.
Property: a secrecy query -- att(secret) is not derivable.
Adversary: "fixed by the tool (Dolev-Yao)" (Example 4) -- so here the edit is on the *protocol*
clauses instead (an implementation mistake), not on the attacker: one clause is dropped (modeling
a service that forgets to check the sender's key before re-encrypting under the recipient's public
key, a classic ProVerif-style "oracle" flaw), and this reveals the secret via the fixed Dolev-Yao
rules alone.
Verifier: naive forward-chaining resolution to a fixpoint, which is sound for secrecy (as ProVerif
is) but, like ProVerif's abstraction, can be a false attack on a richer protocol than this one.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_synth import ModelType, make_attack_synth  # noqa: E402

Term = tuple


def dy_clauses():
    """The fixed Dolev-Yao clauses that matter for *this* query: unpairing (if a pair is ever
    derived) and nothing else. (Pairing/encryption-synthesis rules are correctly part of a full
    Dolev-Yao closure, but this instance never needs to build a new ciphertext to decide secrecy,
    and including them would recompute an unbounded products-of-known-facts closure every
    iteration for no benefit here -- so they are omitted, exactly as tools/tamarin.py omits them
    for the same reason.)"""
    def unpair(facts):
        for f in facts:
            if isinstance(f, tuple) and f and f[0] == "pair":
                yield f[1]
                yield f[2]
    return [unpair]


def decrypt_clause(fixed=True):
    """`att(m) :- att(aenc(m, pk(k))), att(sk(k))` -- the honest decryption rule every correct
    implementation needs; `fixed=False` instead installs the flawed oracle clause
    `att(m) :- att(aenc(m, pk(k)))` (no key check at all), which is the dropped-check bug."""
    def honest(facts):
        for f in facts:
            if isinstance(f, tuple) and f[:1] == ("aenc",) and ("sk", f[2][1]) in facts:
                yield f[1]

    def flawed(facts):
        for f in facts:
            if isinstance(f, tuple) and f[:1] == ("aenc",):
                yield f[1]
    return honest if fixed else flawed


def closure(initial: frozenset, extra_rules) -> frozenset:
    facts = set(initial)
    rules = dy_clauses() + list(extra_rules)
    changed = True
    while changed:
        changed = False
        new = set()
        for rule in rules:
            new |= set(rule(facts))
        if new - facts:
            facts |= new
            changed = True
    return frozenset(facts)


def secrecy_holds(initial, secret, decrypt_rule) -> bool:
    return secret not in closure(initial, [decrypt_rule])


def find_attack(initial, secret, budget=1000):
    """phi: secrecy holds under the honest decryption clause. An attack is the one-clause protocol
    edit (dropping the key check) that lets att(secret) be derived."""
    verifier = lambda fixed, _phi: secrecy_holds(initial, secret, decrypt_clause(fixed))
    mt = ModelType("protocol clause set", neighbors=lambda fixed: ((not fixed,) if fixed else ()),
                  key=lambda fixed: fixed)
    synth = make_attack_synth(verifier, mt, max_steps=1)
    return synth(True, object(), budget=budget, limit=1)


# --------------------------------------------------------------------------- an illustrative instance
def pke_instance():
    k = ("key", "recipient")
    secret = "secret"
    ciphertext = ("aenc", secret, ("pk", k))
    initial = frozenset({ciphertext, ("pk", k)})      # attacker sees the ciphertext and the public key, not sk(k)
    return initial, secret


def to_pv() -> str:
    return """free c: channel.
type key.
free sk: key [private].
free secret: bitstring [private].

fun pk(key): key.
fun aenc(bitstring, key): bitstring.
reduc forall m: bitstring, k: key; adec(aenc(m, pk(k)), k) = m.

(* the flawed oracle the adversary's clause-edit represents would instead publish adec
   without checking the key: reduc forall m: bitstring, k: key; leak(aenc(m, pk(k))) = m. *)

query attacker(secret).

process out(c, aenc(secret, pk(sk)))
"""


def run_real_proverif(pv_text: str):
    exe = shutil.which("proverif")
    if not exe:
        return None, "proverif not found on PATH"
    with tempfile.NamedTemporaryFile("w", suffix=".pv", delete=False) as f:
        f.write(pv_text)
        path = f.name
    try:
        out = subprocess.run([exe, path], capture_output=True, text=True, timeout=120)
        return out.returncode, out.stdout[-4000:]
    except Exception as e:  # pragma: no cover
        return None, str(e)


def main():
    initial, secret = pke_instance()
    print("ProVerif-style -- Horn-clause resolution over a protocol clause edit")
    print(f"  secrecy with the honest decryption clause: {secrecy_holds(initial, secret, decrypt_clause(True))}")
    r = find_attack(initial, secret)
    if r.found:
        print("  attack: dropping the key-check in the decryption clause (a 'leak' oracle) lets "
              "att(secret) be derived by the fixed Dolev-Yao rules alone")
        print(f"  (explored {r.explored} clause-set edits in {r.seconds*1e3:.1f} ms)")
    else:
        print("  no clause edit found that leaks the secret.")
    rc, out = run_real_proverif(to_pv())
    print(f"  real proverif: {'not available in this environment' if rc is None else f'exit {rc}'}")


if __name__ == "__main__":
    main()
