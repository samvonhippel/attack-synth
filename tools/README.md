# tools/

Each file is a native instance in the style of one tool in `paper/threat_model_framework.tex`'s
Example 4 (the model-checker comparison table), built from that tool's own model class and
adversary notion, and designed to be *distinct* in technique from the `papers/` instances even
where the underlying mechanism (fixpoint, search, value iteration) is shared -- see each file's
module docstring for exactly which parts are the real technique and which are simplifications
forced by having no SMT solver, BDD package, or the real binary installed.

Run one: `python3 tools/tamarin.py`
Run all: `python3 tools/run_all.py` (or `python3 -m tools.run_all` from the repo root)

| File | Tool | Model / adversary | What's distinct from the `papers/` instances |
|---|---|---|---|
| `tamarin.py` | Tamarin | ground Dolev-Yao knowledge closure; adversary edits the rule set (one compromised credential) | a *ground* restriction of MSR, where attack_synth's `neighbors` offers one new starting fact per candidate credential |
| `spin.py` | SPIN | shared-variable mutual exclusion; adversary adds a rogue interleaved process | true multi-process interleaving (any enabled process may step), vs. the synchronized two-party rendezvous in `papers/vonhippel.py` |
| `proverif.py` | ProVerif | Horn-clause resolution for secrecy; adversary edits a protocol clause (a dropped key check) | forward-chaining over *clauses*, not ground facts -- the fixed point reasons about the rule shape itself, not one session's trace |
| `nuxmv.py` | nuXmv | Integer-valued traffic-light controller; one transition-relation OR variable-assignment edit allowed along the whole run | the *single-mutation-along-a-path* adversary (tracked in the state itself), vs. the CTL demo's whole-structure edit in `demos.py` |
| `prism.py` | PRISM | an MDP (nondeterministic + probabilistic) with Pmin value iteration, plus epsilon-ball perturbation | genuine nondeterminism (`min` over actions each state), which `papers/oakley.py`'s pure DTMC instance has none of |
| `hytech.py` | HyTech | a two-location linear hybrid automaton (fill/drain tank) with urgent guarded edges; adversary edits a rate, guard, or reset | continuous-flow reachability via interval/box over-approximation -- the only instance in this repository with continuous dynamics |

Every one of these tools is native to `make_attack_synth`: the adversary's edit is encoded as a
`ModelType.neighbors` generator (or, for `tamarin.py`/`proverif.py`, a one-shot set of candidate
rule-set edits), and the verifier is each tool's own core algorithm re-implemented explicitly
(deduction closure, interleaving search, clause resolution, explicit-state reachability, value
iteration, interval reachability) rather than symbolically.

Each file also includes `to_<format>()`, rendering the instance in the tool's real native input
language (`.spthy`, Promela, `.pv`, `.smv`, `.prism`, `.hy`), and `run_real_<tool>()`, which shells
out to the real binary via `shutil.which` if one happens to be installed. None of the six are
installed in this environment, so every `run_real_*` call reports that plainly rather than
pretending to have run the real tool.
