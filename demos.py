"""
demos.py -- plug real instances from the paper's logic table (Example 1) and duality table
(Example 2) into make_attack_synth.   Run:   python3 demos.py [-v] [--only LTL,CTL] [--steps K]

Every demo is a *runnable* system, not a stub:
  LTL      finite-trace LTL evaluator over execution traces
  CTL      fixpoint CTL model checker over Kripke structures
  DEL      public-announcement epistemic logic evaluator
  PCTL     DTMC bounded-reachability probability computation
  STL      sampled-signal robustness-free STL evaluator
  TM       a real Turing machine (binary increment) simulated step by step
  CPU      a toy cache simulator with a prime+probe attacker (micro-architectural side channel)
  Hoare    real Python source mutated through the ast module, checked as a Hoare triple
"""
from __future__ import annotations

import argparse
import ast
import copy
import itertools
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from attack_synth import (And, Atom, ModelType, Not, Op, Or, Implies, make_attack_synth, neg, op,
                          show, structural_dual)


@dataclass
class Demo:
    name: str
    logic: str
    m: Any
    phi: Any
    synth: Any
    describe: Callable[[Any, Any], str]
    dual: Any = None            # the dual goal exactly as listed in the paper's Example 2 table
    dual_fixed: Any = None      # a dual goal that is semantically the negation of phi (if the table's isn't)
    steps: int = 1
    phi_text: str = ""


# =============================================================================== LTL (finite traces)
PROPS = ("Valid", "Secure")
SUBSETS = [frozenset(), frozenset({"Valid"}), frozenset({"Secure"}), frozenset({"Valid", "Secure"})]


def ltl_sat(tr, phi, i=0):
    n = len(tr)
    if isinstance(phi, Atom):
        return phi.name in tr[i]
    if isinstance(phi, Not):
        return not ltl_sat(tr, phi.f, i)
    a = phi.args
    s = lambda f, j: ltl_sat(tr, f, j)
    if phi.name == "and":
        return all(s(x, i) for x in a)
    if phi.name == "or":
        return any(s(x, i) for x in a)
    if phi.name == "implies":
        return (not s(a[0], i)) or s(a[1], i)
    if phi.name == "X":
        return i + 1 < n and s(a[0], i + 1)
    if phi.name == "G":
        return all(s(a[0], j) for j in range(i, n))
    if phi.name == "F":
        return any(s(a[0], j) for j in range(i, n))
    if phi.name == "U":
        return any(s(a[1], j) and all(s(a[0], k) for k in range(i, j)) for j in range(i, n))
    raise ValueError(phi.name)


def ltl_universe(m):
    for n in (len(m) - 1, len(m), len(m) + 1):
        if n >= 1:
            yield from itertools.product(SUBSETS, repeat=n)


def ltl_edit(x, y):
    """Trace operator: flip one proposition at one position, delete one step, or duplicate one step."""
    if len(x) == len(y):
        diff = [(a, b) for a, b in zip(x, y) if a != b]
        return len(diff) == 1 and len(diff[0][0] ^ diff[0][1]) == 1
    if len(y) == len(x) - 1:
        return any(x[:i] + x[i + 1:] == y for i in range(len(x)))
    if len(y) == len(x) + 1:
        return any(x[:i + 1] + x[i:] == y for i in range(len(x)))
    return False


def demo_ltl():
    V, S = Atom("Valid"), Atom("Secure")
    phi = op("G", Implies(V, op("F", S)))
    dual = op("F", And(V, op("G", Not(S))))
    m = tuple(frozenset(s) for s in [{"Valid"}, {"Valid", "Secure"}, {"Secure"}, set()])
    mu = lambda t: sum(ltl_sat(t, phi, i) for i in range(len(t))) / len(t)   # share of good suffixes
    fmt = lambda t: " ".join("{" + ",".join(sorted(s)) + "}" for s in t)
    synth = make_attack_synth(ltl_sat, ModelType("LTL trace", ltl_universe, key=lambda t: t), ltl_edit, measure=mu)
    return Demo("LTL", "Linear Temporal Logic", m, phi, synth,
                lambda a, b: f"{fmt(a)}  ->  {fmt(b)}", dual=dual, dual_fixed=None)


# =============================================================================== CTL (Kripke structures)
@dataclass(frozen=True)
class Kripke:
    states: tuple
    edges: frozenset
    labels: frozenset      # {(state, prop)}
    init: str


def ctl_sat_set(K, phi):
    S = set(K.states)
    succ = lambda s: {t for (u, t) in K.edges if u == s}
    EX = lambda Z: {s for s in S if succ(s) & Z}
    AX = lambda Z: {s for s in S if succ(s) <= Z}
    if isinstance(phi, Atom):
        return {s for s in S if (s, phi.name) in K.labels}
    if isinstance(phi, Not):
        return S - ctl_sat_set(K, phi.f)
    a = [ctl_sat_set(K, x) for x in phi.args]
    n = phi.name
    if n == "and":
        return set.intersection(*a)
    if n == "or":
        return set.union(*a)
    if n == "implies":
        return (S - a[0]) | a[1]
    if n == "EX":
        return EX(a[0])
    if n == "AX":
        return AX(a[0])
    if n in ("EF", "EU"):                       # lfp
        goal, hold = (a[0], S) if n == "EF" else (a[1], a[0])
        Z = set(goal)
        while True:
            Z2 = Z | (hold & EX(Z))
            if Z2 == Z:
                return Z
            Z = Z2
    if n in ("AF", "AU"):
        goal, hold = (a[0], S) if n == "AF" else (a[1], a[0])
        Z = set(goal)
        while True:
            Z2 = Z | (hold & {s for s in S if succ(s) and succ(s) <= Z})
            if Z2 == Z:
                return Z
            Z = Z2
    if n == "EG":                               # gfp
        Z = set(a[0])
        while True:
            Z2 = a[0] & EX(Z)
            if Z2 == Z:
                return Z
            Z = Z2
    if n == "AG":
        Z = set(a[0])
        while True:
            Z2 = a[0] & AX(Z)
            if Z2 == Z:
                return Z
            Z = Z2
    raise ValueError(n)


def ctl_sat(K, phi):
    return K.init in ctl_sat_set(K, phi)


def ctl_universe(K):
    for u, v in itertools.product(K.states, repeat=2):         # toggle every transition
        yield Kripke(K.states, K.edges ^ {(u, v)}, K.labels, K.init)
    for s in K.states:                                          # toggle every label
        yield Kripke(K.states, K.edges, K.labels ^ {(s, "normal")}, K.init)


def ctl_edit(x, y):
    """Adversary may only ADD one transition (state labelling is out of reach)."""
    return x.labels == y.labels and x.states == y.states and x.edges < y.edges and len(y.edges - x.edges) == 1


def reachable(K):
    seen, todo = {K.init}, [K.init]
    while todo:
        s = todo.pop()
        for (u, t) in K.edges:
            if u == s and t not in seen:
                seen.add(t); todo.append(t)
    return seen


def demo_ctl():
    states = ("idle", "login", "session", "locked", "fault")
    edges = frozenset({("idle", "login"), ("login", "session"), ("session", "idle"), ("login", "locked"),
                       ("locked", "idle"), ("fault", "fault")})
    labels = frozenset((s, "normal") for s in states if s != "fault")
    K = Kripke(states, edges, labels, "idle")
    phi = op("AG", Atom("normal"))
    mu = lambda k: (lambda R: sum((s, "normal") in k.labels for s in R) / len(R))(reachable(k))
    synth = make_attack_synth(ctl_sat, ModelType("Kripke structure", ctl_universe,
                                                 key=lambda k: (k.edges, k.labels)), ctl_edit, measure=mu)
    d = lambda a, b: "add transition " + ", ".join(f"{u}->{v}" for u, v in sorted(b.edges - a.edges))
    return Demo("CTL", "Computation Tree Logic", K, phi, synth, d,
                dual=op("AG", Not(Atom("normal"))),           # as written in the Example 2 table
                dual_fixed=op("EF", Not(Atom("normal"))))


# =============================================================================== DEL (public announcements)
@dataclass(frozen=True)
class Epi:
    worlds: frozenset
    val: frozenset          # {(world, prop)}
    acc: frozenset          # {(agent, w, v)}
    point: str


def del_sat(M, phi, w=None):
    w = M.point if w is None else w
    if isinstance(phi, Atom):
        return (w, phi.name) in M.val
    if isinstance(phi, Not):
        return not del_sat(M, phi.f, w)
    n = phi.name
    if n == "and":
        return all(del_sat(M, x, w) for x in phi.args)
    if n == "or":
        return any(del_sat(M, x, w) for x in phi.args)
    if n.startswith("K:"):
        ag = n[2:]
        return all(del_sat(M, phi.args[0], v) for (a, u, v) in M.acc if a == ag and u == w)
    if n.startswith("hatK:"):
        ag = n[5:]
        return any(del_sat(M, phi.args[0], v) for (a, u, v) in M.acc if a == ag and u == w)
    raise ValueError(n)


def announce(M, chi):
    """Public announcement !chi: only possible if chi is true at the actual world."""
    if not del_sat(M, chi, M.point):
        return None
    keep = frozenset(w for w in M.worlds if del_sat(M, chi, w))
    return Epi(keep, frozenset(x for x in M.val if x[0] in keep),
               frozenset(x for x in M.acc if x[1] in keep and x[2] in keep), M.point)


def demo_del():
    obs, sec = Atom("obs"), Atom("secret")
    worlds = frozenset({"w1", "w2"})
    val = frozenset({("w1", "secret"), ("w1", "obs")})
    acc = frozenset((a, u, v) for a in ("adv",) for u in worlds for v in worlds)
    M = Epi(worlds, val, acc, "w1")
    K = lambda f: op("K:adv", f)
    secret_safe = And(Not(K(sec)), Not(K(Not(sec))))            # adversary knows neither value of the secret
    phi = secret_safe
    dual = Or(K(sec), K(Not(sec)))                              # table style: duals of ¬K / ∧
    ALL = [obs, Not(obs), sec, Not(sec)]
    ADV = [obs, Not(obs)]                                       # the adversary can only announce observables
    uni = lambda m: [x for x in (announce(m, c) for c in ALL) if x is not None]
    rel = lambda x, y: y != x and any(announce(x, c) == y for c in ADV)
    synth = make_attack_synth(del_sat, ModelType("epistemic model", uni), rel, measure=lambda m: len(m.worlds))
    return Demo("DEL", "Dynamic Epistemic Logic", M, phi, synth,
                lambda a, b: f"announce an observable: worlds {sorted(a.worlds)} -> {sorted(b.worlds)}",
                dual=dual)


# =============================================================================== PCTL (DTMC)
@dataclass(frozen=True)
class DTMC:
    P: tuple
    labels: tuple          # per-state frozenset of props
    init: int = 0


def reach_prob(M, target, bound):
    x = [1.0 if s in target else 0.0 for s in range(len(M.P))]
    for _ in range(bound if bound is not None else 100000):
        y = [1.0 if s in target else sum(p * v for p, v in zip(M.P[s], x)) for s in range(len(M.P))]
        if bound is None and max(abs(a - b) for a, b in zip(x, y)) < 1e-13:
            x = y
            break
        x = y
    return x[M.init]


def pctl_sat(M, phi):
    mt = re.fullmatch(r"P(>=|<=|>|<)([0-9.]+)", phi.name)
    cmp_, p = mt.group(1), float(mt.group(2))
    path = phi.args[0]
    kind, _, bound = re.fullmatch(r"([FG])(<=(\d+))?", path.name).group(1, 2, 3)
    bound = int(bound) if bound else None
    holds = {s for s in range(len(M.P)) if path.args[0].name in M.labels[s]}
    if kind == "F":
        pr = reach_prob(M, holds, bound)
    else:
        pr = 1.0 - reach_prob(M, set(range(len(M.P))) - holds, bound)
    return {">=": pr >= p - 1e-12, ">": pr > p + 1e-12, "<=": pr <= p + 1e-12, "<": pr < p - 1e-12}[cmp_]


DELTAS = (1e-4, 1e-3, 1e-2, 5e-2)
BUDGET = 0.02                         # adversary may move at most this much probability mass in one row


def dtmc_universe(M):
    n = len(M.P)
    for i in range(n):
        for j in range(n):
            for d in DELTAS:
                row = list(M.P[i])
                k = max((t for t in range(n) if t != j), key=lambda t: row[t])
                if row[k] < d:
                    continue
                row[k] = round(row[k] - d, 9)
                row[j] = round(row[j] + d, 9)
                yield DTMC(M.P[:i] + (tuple(row),) + M.P[i + 1:], M.labels, M.init)


def dtmc_edit(x, y):
    diff = [i for i in range(len(x.P)) if x.P[i] != y.P[i]]
    if len(diff) != 1 or x.labels != y.labels:
        return False
    i = diff[0]
    l1 = sum(abs(a - b) for a, b in zip(x.P[i], y.P[i]))
    return l1 <= 2 * BUDGET + 1e-12 and abs(sum(y.P[i]) - 1) < 1e-9


def demo_pctl():
    P = ((0.5, 0.5, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    labels = (frozenset({"ok"}), frozenset({"ok"}), frozenset({"fail"}))
    M = DTMC(P, labels)
    phi = Op("P>=0.99", (op("G<=50", Atom("ok")),))
    mu = lambda m: 1.0 - reach_prob(m, {2}, 50)
    synth = make_attack_synth(pctl_sat, ModelType("DTMC", dtmc_universe, key=lambda d: d.P), dtmc_edit, measure=mu)
    d = lambda a, b: "; ".join(f"row {i}: {a.P[i]} -> {b.P[i]}" for i in range(len(a.P)) if a.P[i] != b.P[i])
    return Demo("PCTL", "Probabilistic CTL", M, phi, synth, d,
                dual=Op("P<0.99", (op("F<=50", Atom("fail")),)),      # as in the Example 2 table
                dual_fixed=Op("P>0.01", (op("F<=50", Atom("fail")),)))


# =============================================================================== STL (sampled signals)
def stl_sat(sig, phi, t=0):
    if isinstance(phi, Atom):
        mt = re.fullmatch(r"x(>=|<=|>|<)(-?[0-9.]+)", phi.name)
        c = float(mt.group(2))
        return {">=": sig[t] >= c, "<=": sig[t] <= c, ">": sig[t] > c, "<": sig[t] < c}[mt.group(1)]
    if isinstance(phi, Not):
        return not stl_sat(sig, phi.f, t)
    mt = re.fullmatch(r"([GF])\[(\d+),(\d+)\]", phi.name)
    kind, a, b = mt.group(1), int(mt.group(2)), int(mt.group(3))
    rng = range(t + a, min(t + b, len(sig) - 1) + 1)
    return (all if kind == "G" else any)(stl_sat(sig, phi.args[0], j) for j in rng)


WINDOW, EPS_STEPS = range(2, 10), (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75)


def stl_universe(sig):
    for i in range(len(sig)):
        for d in EPS_STEPS:
            yield sig[:i] + (round(sig[i] + d, 3),) + sig[i + 1:]


def stl_edit(x, y):
    diff = [i for i in range(len(x)) if x[i] != y[i]]
    return len(diff) == 1 and diff[0] in WINDOW and abs(x[diff[0]] - y[diff[0]]) <= 0.75 + 1e-9


def demo_stl():
    sig = (3.0, 2.8, 2.5, 2.2, 2.0, 1.8, 1.6, 1.5, 1.7, 2.0, 2.4, 2.8)
    phi = op("G[0,11]", Atom("x>=1.0"))
    dual = op("F[0,11]", Atom("x<1.0"))
    mu = lambda s: sum(v >= 1.0 for v in s) / len(s)
    synth = make_attack_synth(stl_sat, ModelType("signal", stl_universe, key=lambda s: s), stl_edit, measure=mu)
    d = lambda a, b: next(f"sample {i}: {a[i]} -> {b[i]}" for i in range(len(a)) if a[i] != b[i])
    return Demo("STL", "Signal Temporal Logic", sig, phi, synth, d, dual=dual)


# =============================================================================== Turing machine
TM_STATES, TM_SYMS, TM_MOVES = ("scan", "carry", "acc", "rej"), ("0", "1", "_"), ("L", "R", "S")
INC = (  # binary increment: scan right to the end, then propagate the carry leftwards
    (("scan", "0"), ("scan", "0", "R")), (("scan", "1"), ("scan", "1", "R")), (("scan", "_"), ("carry", "_", "L")),
    (("carry", "1"), ("carry", "0", "L")), (("carry", "0"), ("acc", "1", "S")), (("carry", "_"), ("acc", "1", "S")),
)


def run_tm(rules, word, max_steps=200):
    table, tape, st, head = dict(rules), dict(enumerate(word)), "scan", 0
    for _ in range(max_steps):
        if st in ("acc", "rej"):
            break
        r = table.get((st, tape.get(head, "_")))
        if r is None:
            return "stuck", None
        st, w, mv = r
        tape[head] = w
        head += {"L": -1, "R": 1, "S": 0}[mv]
    if st != "acc":
        return st, None
    return st, "".join(tape[i] for i in sorted(tape) if tape[i] != "_")


@dataclass(frozen=True)
class TMSpec:
    max_len: int = 4

    def inputs(self):
        for n in range(1, self.max_len + 1):
            for bits in itertools.product("01", repeat=n):
                w = "".join(bits)
                if w == "0" or not w.startswith("0"):       # canonical binary numerals only
                    yield w


def tm_ok(rules, w):
    st, out = run_tm(rules, w)
    return st == "acc" and out == bin(int(w, 2) + 1)[2:]


def tm_sat(rules, spec):
    return all(tm_ok(rules, w) for w in spec.inputs())


def tm_universe(rules):
    for i, (k, _) in enumerate(rules):
        for new in itertools.product(TM_STATES, TM_SYMS, TM_MOVES):
            yield rules[:i] + ((k, new),) + rules[i + 1:]


def tm_edit(x, y):
    """Transition-rule mutation: exactly one rule's right-hand side is rewritten."""
    return [k for (k, v), (_, v2) in zip(x, y) if v != v2].__len__() == 1


def demo_tm():
    spec = TMSpec(4)
    ins = list(spec.inputs())
    mu = lambda r: sum(tm_ok(r, w) for w in ins) / len(ins)
    synth = make_attack_synth(tm_sat, ModelType("Turing machine", tm_universe, key=lambda r: r), tm_edit, measure=mu)
    d = lambda a, b: next(f"rule {k} : {v} -> {v2}" for (k, v), (_, v2) in zip(a, b) if v != v2)
    return Demo("TM", "Turing Machine", INC, spec, synth, d, phi_text="increments every binary input of length <= 4")


# =============================================================================== micro-architectural side channel
@dataclass(frozen=True)
class CPU:
    partitioned: bool = True
    flush_on_switch: bool = True
    prefetch: bool = False


def probe(cpu, secret, sets=8):
    """Prime+probe on a direct-mapped cache. Attacker owns sets 0-3; the victim does table[secret]."""
    cache = {s: None for s in range(sets)}
    for s in range(4):                                   # prime
        cache[s] = "A"
    vic = [secret] + ([secret + 1] if cpu.prefetch else [])
    for line in vic:                                     # victim run
        cache[(line % 4) + 4 if cpu.partitioned else line % sets] = "V"
    if cpu.flush_on_switch:                              # context-switch flush
        cache = {s: None for s in cache}
    return tuple(cache[s] == "A" for s in range(4))      # probe: True = hit


def cpu_sat(cpu, _spec):
    """Non-interference: the attacker's observation is independent of the secret."""
    return len({probe(cpu, s) for s in range(4)}) == 1


def cpu_universe(cpu):
    for f in ("partitioned", "flush_on_switch", "prefetch"):
        yield CPU(**{**cpu.__dict__, f: not getattr(cpu, f)})


def cpu_edit(x, y):
    return sum(a != b for a, b in zip(x.__dict__.values(), y.__dict__.values())) == 1


def demo_cpu():
    mu = lambda c: sum(probe(c, s) == probe(c, 0) for s in range(4)) / 4
    synth = make_attack_synth(cpu_sat, ModelType("CPU config", cpu_universe, key=lambda c: c), cpu_edit, measure=mu)
    d = lambda a, b: ", ".join(f"{f}: {getattr(a, f)} -> {getattr(b, f)}" for f in a.__dict__ if getattr(a, f) != getattr(b, f))
    return Demo("CPU", "Micro-architectural side channel", CPU(), "noninterference", synth, d, steps=2,
                phi_text="probe trace independent of secret")


# =============================================================================== Hoare logic on real Python source
CMPS = {ast.LtE: (ast.Lt, ast.Gt, ast.GtE, ast.Eq, ast.NotEq), ast.Lt: (ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq),
        ast.GtE: (ast.Gt, ast.Lt, ast.LtE, ast.Eq, ast.NotEq), ast.Gt: (ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq)}


def mutants(src, kinds=("cmp", "arith", "const", "guard")):
    """All single-edit mutants of a Python function (the universe of source edits)."""
    tree = ast.parse(src)
    out = set()
    for idx, node in enumerate(ast.walk(tree)):
        def fresh():
            t = copy.deepcopy(tree)
            return t, list(ast.walk(t))[idx], t
        if "cmp" in kinds and isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMPS:
            for new in CMPS[type(node.ops[0])]:
                t, n, root = fresh(); n.ops[0] = new(); out.add(ast.unparse(root))
        if "arith" in kinds and isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            t, n, root = fresh(); n.op = ast.Add() if isinstance(n.op, ast.Sub) else ast.Sub(); out.add(ast.unparse(root))
        if "const" in kinds and isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            for d in (-1, 1):
                t, n, root = fresh(); n.value += d; out.add(ast.unparse(root))
        if "guard" in kinds and isinstance(node, ast.If):
            t, n, root = fresh()
            for par in ast.walk(root):
                if hasattr(par, "body") and isinstance(par.body, list) and n in par.body:
                    i = par.body.index(n); par.body[i:i + 1] = n.body; out.add(ast.unparse(root)); break
    return out


@dataclass(frozen=True)
class HoareSpec:
    func: str
    pre: Callable
    post: Callable
    domain: tuple

    def points(self):
        return [a for a in self.domain if self.pre(*a)]


def hoare_ok(src, spec, args):
    ns = {}
    exec(compile(src, "<mutant>", "exec"), ns)            # our own generated code
    try:
        return bool(spec.post(args, ns[spec.func](*args)))
    except Exception:
        return False


def hoare_sat(src, spec):
    return all(hoare_ok(src, spec, a) for a in spec.points())


def demo_hoare():
    src = ast.unparse(ast.parse(
        "def withdraw(balance, x):\n"
        "    if x <= balance:\n"
        "        balance = balance - x\n"
        "    return balance\n"))
    spec = HoareSpec("withdraw", pre=lambda b, x: b >= 0 and x >= 0, post=lambda a, r: r >= 0,
                     domain=tuple(itertools.product(range(0, 6), range(0, 7))))
    uni = lambda s: mutants(s)                                              # all mutants, incl. guard removal
    rel = lambda a, b: b in mutants(a, kinds=("cmp", "arith", "const"))     # adversary can't delete code
    mu = lambda s: sum(hoare_ok(s, spec, a) for a in spec.points()) / len(spec.points())
    synth = make_attack_synth(hoare_sat, ModelType("Python source", uni, key=lambda s: s), rel, measure=mu)
    d = lambda a, b: " | ".join(f"{x.strip()}  =>  {y.strip()}" for x, y in zip(a.splitlines(), b.splitlines()) if x != y)
    return Demo("Hoare", "Hoare logic (Tamarin example lifted to source)", src, spec, synth, d,
                phi_text="{balance>=0 and x>=0} withdraw {result>=0}")


DEMOS = [demo_ltl, demo_ctl, demo_del, demo_pctl, demo_stl, demo_tm, demo_cpu, demo_hoare]


# =============================================================================== runner
def run(demo: Demo, steps: Optional[int], verbose: bool):
    k = steps or demo.steps
    r = demo.synth(demo.m, demo.phi, max_steps=k)
    row = {"name": demo.name, "k": k, "explored": r.explored, "attacks": len(r.attacks), "ms": r.seconds * 1e3,
           "dual": "-", "fixed": "-", "best": "none"}
    if r.found:
        b = r.best
        row["best"] = f"σ={b.sigma:.2f}  {demo.describe(b.path[0], b.model)}"
    keys = lambda res: {demo.synth.model_type.key(a.model) for a in res.attacks}
    for field_, attr in (("dual", demo.dual), ("fixed", demo.dual_fixed)):
        if attr is not None:
            rd = demo.synth(demo.m, demo.phi, goal="dual", dual_phi=attr, max_steps=k)
            row[field_] = "agrees" if keys(rd) == keys(r) else f"DIFFERS ({len(rd.attacks)} vs {len(r.attacks)})"
    if verbose:
        print(f"\n== {demo.name}: {demo.logic}")
        if isinstance(demo.phi, (Atom, Not, Op)):
            print("   φ       :", show(demo.phi), "\n   structural dual (Def.):", show(structural_dual(demo.phi)))
        else:
            print("   φ       :", demo.phi_text)
        for a in r.attacks[:5]:
            print(f"   attack σ={a.sigma:.2f}: {demo.describe(a.path[0], a.model)}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--steps", type=int, default=None, help="override the number of composed edits")
    args = ap.parse_args()
    only = {x.strip().lower() for x in args.only.split(",") if x.strip()}
    rows = []
    for mk in DEMOS:
        d = mk()
        if only and d.name.lower() not in only:
            continue
        rows.append(run(d, args.steps, args.verbose))
        if d.steps > 1 and not args.steps:                      # also show the weaker adversary
            rows.append(run(d, 1, False))
    print(f"\n{'demo':7}{'k':>2} {'cands':>6} {'attacks':>8} {'ms':>7}  {'dual(table)':17}{'dual(fixed)':13} best attack")
    for r in rows:
        print(f"{r['name']:7}{r['k']:>2} {r['explored']:>6} {r['attacks']:>8} {r['ms']:>7.1f}  "
              f"{r['dual']:17}{r['fixed']:13} {r['best']}")


if __name__ == "__main__":
    main()
