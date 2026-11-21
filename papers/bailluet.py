"""
papers/bailluet.py -- N. Bailluet, E. Fleury, I. Puaut, E. Rohou, "Nothing is Unreachable:
Automated Synthesis of Robust Code-Reuse Gadget Chains for Arbitrary Exploitation Primitives"
(USENIX Security 2025; tool: ARCANIST).

Model: a machine state s = <p, u> (bit-vector registers and byte memory), split into an
attacker-predictable part p and an attacker-unpredictable part u (ASLR offsets, stack canaries,
...). A gadget is a transition g(s, s') = pre(s) and post(s, s').
Adversary ~> : the attacker overwrites the code pointer and places controlled data in memory
(predictable p), then moves only through gadget transitions (NX on, CFI off).
Attack (robust n-gadget satisfiability, Def. 1): a chain of n gadgets and an assignment to p such
that the goal specification S holds after the chain, FOR ALL values of u.

This is exactly attack_synth's `robust=True` quantifier: `model_type.perturb(s)` re-samples the
unpredictable bytes of a candidate state, and an attack must hold for all of them. The paper
removes the forall with a taint encoding and solves one SMT query over bit-vectors and arrays with
Boolector; without an SMT solver here, this instead checks the universal quantifier by explicit
sampling of u's small domain -- exact for the toy machine below (u ranges over one byte), but only
a sound *approximation* on a larger domain.
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_synth import ModelType, make_attack_synth  # noqa: E402

MASK = 0xFF   # 8-bit toy registers/memory, for a fast, exhaustively-checkable unpredictable byte


@dataclass(frozen=True)
class State:
    regs: tuple          # (rax, rbx, rsp) as bytes, masked
    mem: tuple           # memory, indexed 0..len-1, as bytes
    canary_idx: int      # which memory cell holds the attacker-unpredictable value (the "u")


def mk(state: State, **changes) -> State:
    regs = changes.pop("regs", state.regs)
    mem = changes.pop("mem", state.mem)
    return State(regs, mem, state.canary_idx)


@dataclass(frozen=True)
class Gadget:
    name: str
    pre: Callable[[State], bool]
    post: Callable[[State], State]


def gadget_library(target_addr: int) -> tuple:
    """A small x86-ish gadget set: pop a byte off the (simulated) stack into rax, move rax into
    rbx, write rax to memory at the address in rbx ("mov [rbx], rax" -- an arbitrary-write
    primitive), and add an immediate to rax."""
    def pop_rax(s):
        return Gadget("pop rax; ret",
                     pre=lambda s: s.regs[2] < len(s.mem),
                     post=lambda s: mk(s, regs=(s.mem[s.regs[2]], s.regs[1], (s.regs[2] + 1) & MASK)))
    def mov_rbx_rax(s):
        return Gadget("mov rbx, rax; ret", pre=lambda s: True,
                     post=lambda s: mk(s, regs=(s.regs[0], s.regs[0], s.regs[2])))
    def write_rbx_rax(s):
        return Gadget("mov [rbx], rax; ret",
                     pre=lambda s: s.regs[1] < len(s.mem),
                     post=lambda s: mk(s, mem=s.mem[:s.regs[1]] + (s.regs[0] & MASK,) + s.mem[s.regs[1] + 1:]))
    def add_imm(k):
        return Gadget(f"add rax, {k}; ret", pre=lambda s: True,
                     post=lambda s: mk(s, regs=((s.regs[0] + k) & MASK, s.regs[1], s.regs[2])))
    return (pop_rax(None), mov_rbx_rax(None), write_rbx_rax(None), add_imm(1), add_imm(0x40))


def neighbors(s: State, gadgets: tuple):
    for g in gadgets:
        if g.pre(s):
            yield g.post(s)


def perturb_unpredictable(s: State, domain=range(0, 256, 37)) -> tuple:
    """Re-samples the one attacker-unpredictable memory byte (e.g. a canary/offset at a fixed
    index) over a coarse grid of its domain, holding everything else fixed -- the paper's
    `forall u` quantifier, checked by sampling rather than an SMT solver."""
    i = s.canary_idx
    return tuple(mk(s, mem=s.mem[:i] + (v,) + s.mem[i + 1:]) for v in domain)


def goal_write(target_addr: int, value: int):
    return lambda s: len(s.mem) > target_addr and s.mem[target_addr] == value


def synthesize_chain(s0: State, gadgets: tuple, goal: Callable[[State], bool], max_len=4, budget=20000, robust=True):
    """phi = 'goal not yet reached'; an attack is any gadget-chain state for which goal holds
    (robustly, for every sampled u, when robust=True)."""
    verifier = lambda s, _phi: not goal(s)
    mt = ModelType("machine state", neighbors=lambda s: neighbors(s, gadgets),
                  key=lambda s: (s.regs, s.mem), perturb=lambda s: perturb_unpredictable(s))
    synth = make_attack_synth(verifier, mt, max_steps=max_len)
    return synth(s0, object(), budget=budget, limit=1, robust=robust)


# --------------------------------------------------------------------------- an illustrative instance
def toy_instance():
    mem_len, target_addr, canary_idx = 16, 12, 10
    # the attacker controls memory[0:8] (the simulated stack: bytes to pop) and the initial
    # registers; memory[canary_idx] is unpredictable (e.g. a randomized cookie elsewhere in
    # memory that the chain must not depend on).
    stack = (target_addr, 0x41) + (0,) * (mem_len - 2)   # mem[0]=target address, mem[1]=value to write
    s0 = State(regs=(0, 0, 0), mem=stack, canary_idx=canary_idx)
    gadgets = gadget_library(target_addr)
    goal = goal_write(target_addr, 0x41)
    return s0, gadgets, goal


def main():
    s0, gadgets, goal = toy_instance()
    print("Bailluet et al. -- robust gadget-chain synthesis")
    print(f"  gadgets available: {[g.name for g in gadgets]}")
    print(f"  goal: memory[12] == 0x41 (an arbitrary-write primitive), for all sampled values of the unpredictable byte")
    r = synthesize_chain(s0, gadgets, goal, max_len=4, robust=True)
    print(f"  explored {r.explored} chain states in {r.seconds*1e3:.1f} ms (budgeted search)")
    if r.found:
        b = r.best
        names = []
        cur = s0
        for nxt in b.path[1:]:
            names.append(next(g.name for g in gadgets if g.pre(cur) and g.post(cur) == nxt))
            cur = nxt
        print(f"  robust {len(names)}-gadget chain found: " + " ; ".join(names))
        print(f"  final state memory[12] = {hex(b.model.mem[12])}")
    else:
        print("  no robust chain found within the length bound and budget.")
    r2 = synthesize_chain(s0, gadgets, goal, max_len=4, robust=False)
    print(f"  (without the robustness/forall-u requirement: found={r2.found}, same chain exists non-robustly too: "
          f"{r2.found == r.found})")


if __name__ == "__main__":
    main()
