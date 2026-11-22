"""
tools/tamarin.py -- a native instance in the style of the Tamarin Prover.

Model: ground multiset-rewriting facts (Tamarin's state is a multiset of facts; here, a frozenset
of ground fact tuples -- a *ground* restriction, since there is no unification/constraint solver
available, only concrete values). Facts are closed under a set of Dolev-Yao deduction rules
(pairing/unpairing, symmetric encryption/decryption), i.e. "the attacker's knowledge."
Property (lemma): secrecy -- the attacker's knowledge never contains the ground term `secret`.
Adversary: "adding or changing a rewrite rule" (Example 4's own words) -- here, granting the
attacker one additional starting fact, modeling a single compromised credential.
Verifier: the deduction-closure fixpoint stands in for Tamarin's constraint solver (an exhaustive
symbolic search for a trace violating the lemma); on ground facts, forward-chaining to a fixpoint
decides exactly the same reachability question.

`to_spthy()` renders the instance as a real `.spthy` theory; `run_real_tamarin()` shells out to the
installed `tamarin-prover` binary if one is found (none is, in this environment).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_synth import ModelType, make_attack_synth  # noqa: E402

Fact = tuple   # ("enc", msg, key), ("pair", a, b), ("key", name), ("msg", name), "secret", ...


def deduction_closure(known: frozenset) -> frozenset:
    """Dolev-Yao *analysis* closure, to a fixpoint: unpairing, and decryption when the key is
    known. (Synthesis rules -- the attacker can always pair or encrypt things it knows -- are
    omitted: they only ever build bigger terms, and can never make an already-known atomic secret
    newly appear, so they are irrelevant to a secrecy query and would otherwise blow the ground
    closure up combinatorially.)"""
    known = set(known)
    changed = True
    while changed:
        changed = False
        new = set()
        for f in known:
            if isinstance(f, tuple) and f[0] == "pair":
                new.add(f[1]); new.add(f[2])
            if isinstance(f, tuple) and f[0] == "enc":
                _, m, k = f
                if k in known:
                    new.add(m)
        if new - known:
            known |= new
            changed = True
    return frozenset(known)


def secrecy_holds(known: frozenset, secret) -> bool:
    return secret not in deduction_closure(known)


def leak_candidates(initial: frozenset, compromisable: tuple):
    for cred in compromisable:
        if cred not in initial:
            yield initial | {cred}


def find_attack(initial: frozenset, secret, compromisable: tuple, budget=2000):
    """phi: secrecy holds. An attack is a rule-set edit granting one extra starting fact (a
    compromised credential) after which the deduction closure contains `secret`."""
    verifier = lambda known, _phi: secrecy_holds(known, secret)
    mt = ModelType("attacker knowledge", neighbors=lambda m: leak_candidates(m, compromisable), key=lambda m: m)
    synth = make_attack_synth(verifier, mt, max_steps=1)
    return synth(initial, object(), budget=budget, limit=1)


# --------------------------------------------------------------------------- an illustrative instance
def account_mutation_instance():
    """The paper's own AccountMutation-style theory, recast for secrecy: a server sends a
    session key `sk` to a client, symmetrically encrypted under a long-term key `ltk`. The
    attacker starts knowing only the ciphertext and public constants; the lemma is that `sk`
    stays secret unless `ltk` is separately compromised."""
    ltk, sk, secret_msg = ("key", "ltk"), ("key", "sk"), "secret"
    ciphertext = ("enc", sk, ltk)
    ciphertext2 = ("enc", secret_msg, sk)
    initial = frozenset({ciphertext, ciphertext2, ("const", "0"), ("const", "1")})
    return initial, secret_msg, (ltk, sk)


def to_spthy(secret_name="secret") -> str:
    return f"""theory AccountMutation
begin

builtins: symmetric-encryption

rule Server_Send:
  [ Fr(~ltk), Fr(~sk), Fr(~{secret_name}) ]
  --[ Setup() ]->
  [ !Ltk(~ltk), Out(senc(~sk, ~ltk)), Out(senc(~{secret_name}, ~sk)) ]

rule Compromise_Ltk:
  [ !Ltk(ltk) ]
  --[ Compromised(ltk) ]->
  [ Out(ltk) ]

lemma secrecy:
  " not (Ex #i. K({secret_name}) @ i) "

end
"""


def run_real_tamarin(spthy_text: str):
    exe = shutil.which("tamarin-prover")
    if not exe:
        return None, "tamarin-prover not found on PATH"
    with tempfile.NamedTemporaryFile("w", suffix=".spthy", delete=False) as f:
        f.write(spthy_text)
        path = f.name
    try:
        out = subprocess.run([exe, "--prove", path], capture_output=True, text=True, timeout=120)
        return out.returncode, out.stdout[-4000:]
    except Exception as e:  # pragma: no cover
        return None, str(e)


def main():
    initial, secret_msg, compromisable = account_mutation_instance()
    print("Tamarin-style -- Dolev-Yao deduction closure over a rule-set edit")
    print(f"  base knowledge: {sorted(map(str, initial))}")
    print(f"  secrecy of {secret_msg!r} before any edit: {secrecy_holds(initial, secret_msg)}")
    r = find_attack(initial, secret_msg, compromisable)
    if r.found:
        leaked = sorted(map(str, r.best.model - initial))
        print(f"  attack: compromising {leaked} (adding a rule granting it) leaks the secret")
        print(f"  (explored {r.explored} candidate rule-set edits in {r.seconds*1e3:.1f} ms)")
    else:
        print("  no single-credential compromise leaks the secret.")
    rc, out = run_real_tamarin(to_spthy())
    print(f"  real tamarin-prover: {'not available in this environment' if rc is None else f'exit {rc}'}")


if __name__ == "__main__":
    main()
