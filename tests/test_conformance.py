"""tests/test_conformance.py -- checkable proof that attack_synth.py's Logic/ThreatModel/
DualThreatModel/MeasureSpace/severity layer (attack_synth.py's "paper conformance" section)
actually implements the paper's own Definitions, not just names them.

Standard library only (unittest). Run:
    python3 -m unittest tests.test_conformance -v          (from the repo root)
    python3 tests/test_conformance.py -v                    (directly)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attack_synth import (And, Atom, DualThreatModel, Implies, Logic, MeasureSpace, ModelType,
                          Not, Op, ThreatModel, make_attack_synth, neg, op, severity,
                          structural_dual)
from demos import demo_ltl, ltl_edit, ltl_sat, ltl_universe


class TestStructuralDualInvolution(unittest.TestCase):
    """Definition: the structural dual (.)^d. The paper's construction is only meaningful if
    applying it twice returns the original formula (an involution) -- check this on formulas of
    every shape the AST supports: atom, negated atom, negated compound, and nested compounds."""

    def _check(self, f):
        self.assertEqual(structural_dual(structural_dual(f)), f)

    def test_atom(self):
        self._check(Atom("p"))

    def test_negated_atom(self):
        self._check(Not(Atom("p")))

    def test_binary_op(self):
        self._check(And(Atom("p"), Atom("q")))

    def test_negated_compound(self):
        self._check(neg(And(Atom("p"), Atom("q"))))

    def test_nested(self):
        self._check(op("G", Implies(Atom("Valid"), op("F", Atom("Secure")))))

    def test_known_values(self):
        # (p)^d = not p
        self.assertEqual(structural_dual(Atom("p")), Not(Atom("p")))
        # (not p)^d = p
        self.assertEqual(structural_dual(Not(Atom("p"))), Atom("p"))
        # (p and q)^d = not (not p and not q)  ==  De Morgan's "or"
        d = structural_dual(And(Atom("p"), Atom("q")))
        self.assertEqual(d, neg(Op("and", (Not(Atom("p")), Not(Atom("q"))))))


class TestThreatModelBaselineRequirement(unittest.TestCase):
    """Definition: a threat model T = (m, phi, ~>) presupposes m |= phi. Constructing one whose
    baseline does not satisfy phi must fail -- not silently search anyway."""

    def test_valid_baseline_accepted(self):
        verifier = lambda m, _phi: m > 0
        mt = ModelType("int", universe=lambda m: (m - 1, m + 1), key=lambda m: m)
        logic = Logic("toy", verifier)
        tm = ThreatModel(logic, m=5, phi="positive", model_type=mt, edit_relation=lambda a, b: True)
        self.assertEqual(tm.m, 5)

    def test_invalid_baseline_rejected(self):
        verifier = lambda m, _phi: m > 0
        mt = ModelType("int", universe=lambda m: (m - 1, m + 1), key=lambda m: m)
        logic = Logic("toy", verifier)
        with self.assertRaises(ValueError):
            ThreatModel(logic, m=-5, phi="positive", model_type=mt, edit_relation=lambda a, b: True)


class TestAttackIsGenuine(unittest.TestCase):
    """Definition: attack = (m ~> m') and (m' |/= phi). A found attack's model must both be
    reachable via the edit relation from the baseline and genuinely fail the verifier -- this is
    checked directly against the raw verifier/edit_relation, not re-derived from the search."""

    def setUp(self):
        verifier = lambda m, _phi: m % 2 == 0           # phi: "m is even"
        edit = lambda a, b: abs(a - b) == 1              # ~> : +/-1
        mt = ModelType("int", universe=lambda m: (m - 1, m + 1), key=lambda m: m)
        self.logic = Logic("parity", verifier)
        self.tm = ThreatModel(self.logic, m=10, phi="even", model_type=mt, edit_relation=edit)

    def test_attack_found_and_genuine(self):
        result = self.tm.synthesize()
        self.assertTrue(result.found)
        mprime = result.best.model
        # (m ~> m')
        self.assertTrue(abs(self.tm.m - mprime) == 1)
        # (m' |/= phi)
        self.assertFalse(self.logic.verifier(mprime, self.tm.phi))


class TestDualThreatModel(unittest.TestCase):
    """Definition: T^d = (m, (phi)^d, ~>); a realized dual attack is m' in C(m) with m' |= (phi)^d.
    Also checks that DualThreatModel does NOT require m |= (phi)^d at construction (unlike
    ThreatModel, which does require m |= phi) -- a baseline that already fails the dual property
    must still be a legal dual threat model."""

    def test_dual_phi_is_structural_dual(self):
        verifier = lambda s, phi: isinstance(phi, Atom) and phi.name in s
        mt = ModelType("set", universe=lambda s: (s | {"x"}, s - {"x"}), key=lambda s: frozenset(s))
        logic = Logic("membership", verifier)
        tm = ThreatModel(logic, m=frozenset({"ok"}), phi=Atom("ok"), model_type=mt,
                          edit_relation=lambda a, b: True)
        d = tm.dual()
        self.assertIsInstance(d, DualThreatModel)
        self.assertEqual(d.phi_d, structural_dual(Atom("ok")))

    def test_dual_construction_does_not_require_m_sat_phi_d(self):
        # phi = Atom("ok"); phi^d = Not(Atom("ok")).  m |= phi (True in "ok"), so m does NOT
        # satisfy phi^d.  Building the dual threat model must not raise despite that.
        verifier = lambda s, phi: (phi.name in s) if isinstance(phi, Atom) else (phi.f.name not in s)
        mt = ModelType("set", universe=lambda s: (s | {"x"}, s - {"x"}), key=lambda s: frozenset(s))
        logic = Logic("membership", verifier)
        tm = ThreatModel(logic, m=frozenset({"ok"}), phi=Atom("ok"), model_type=mt,
                          edit_relation=lambda a, b: True)
        d = tm.dual()                      # must not raise
        self.assertFalse(logic.verifier(tm.m, d.phi_d))

    def test_realized_dual_attack(self):
        # phi: "even"; phi^d = Not(phi) in spirit --> here verifier is boolean-direct (no AST), so
        # emulate phi/phi^d as two plain predicates and pass dual_phi explicitly via .realized().
        verifier = lambda m, predicate: predicate(m)
        is_even = lambda m: m % 2 == 0
        is_odd = lambda m: m % 2 == 1
        mt = ModelType("int", universe=lambda m: (m - 1, m + 1), key=lambda m: m)
        logic = Logic("parity", verifier)
        tm = ThreatModel(logic, m=10, phi=is_even, model_type=mt, edit_relation=lambda a, b: abs(a - b) == 1)
        d = DualThreatModel(tm, phi_d=is_odd)
        result = d.realized()
        self.assertTrue(result.found)
        self.assertTrue(is_odd(result.best.model))


class TestSeverity(unittest.TestCase):
    """Definition: sigma = 1 - mu(M')/mu(M), clamped to [0, 1], with the paper's three named
    cases: sigma == 0 (nominal), 0 < sigma < 1 (partial), sigma == 1 (full compromise)."""

    def test_zero_when_no_mass_lost(self):
        mu = lambda m: 10.0
        self.assertEqual(severity(mu, "a", "b"), 0.0)

    def test_one_when_all_mass_lost(self):
        mu = lambda m: {"a": 10.0, "b": 0.0}[m]
        self.assertEqual(severity(mu, "a", "b"), 1.0)

    def test_partial(self):
        mu = lambda m: {"a": 10.0, "b": 4.0}[m]
        self.assertAlmostEqual(severity(mu, "a", "b"), 0.6)

    def test_clamped_not_negative(self):
        # mu(m') > mu(m) would make the raw ratio negative; severity must clamp to 0.
        mu = lambda m: {"a": 10.0, "b": 20.0}[m]
        self.assertEqual(severity(mu, "a", "b"), 0.0)

    def test_zero_baseline_measure(self):
        mu = lambda m: 0.0
        self.assertEqual(severity(mu, "a", "b"), 0.0)

    def test_accepts_measure_space(self):
        space = MeasureSpace(mu=lambda m: {"a": 10.0, "b": 4.0}[m])
        self.assertAlmostEqual(severity(space, "a", "b"), 0.6)

    def test_sigma_delegates_to_severity(self):
        mu = lambda m: {"a": 10.0, "b": 4.0}[m]
        mt = ModelType("x", universe=lambda m: ("b",) if m == "a" else (), key=lambda m: m)
        synth = make_attack_synth(lambda m, _phi: m == "a", mt, lambda a, b: True, measure=mu)
        self.assertAlmostEqual(synth.sigma("a", "b"), severity(mu, "a", "b"))

    def test_no_warning_or_exception_when_mu_increases(self):
        # A non-attack candidate examined mid-search (e.g. papers/oakley.py's epsilon-ball
        # sampling) can legitimately have mu(m') > mu(m); severity() must fold this to 0.0
        # quietly, not warn or raise -- see severity()'s docstring for why this is routine.
        import warnings
        mu = lambda m: {"a": 10.0, "b": 20.0}[m]
        with warnings.catch_warnings():
            warnings.simplefilter("error")   # any warning here would fail the test
            self.assertEqual(severity(mu, "a", "b"), 0.0)


class TestCandidateSpace(unittest.TestCase):
    """Definition 6's C(m) = {m' in M | m ~> m'}, exposed directly as
    AttackSynthesizer.candidate_space / ThreatModel.candidate_space, should match exactly the
    set of candidates the search itself would expand at depth 1 -- i.e. everything ModelType
    offers that also passes edit_relation, and nothing else."""

    def test_matches_manual_filtering(self):
        verifier = lambda m, _phi: m % 2 == 0
        edit = lambda a, b: abs(a - b) == 1
        mt = ModelType("int", universe=lambda m: (m - 2, m - 1, m + 1, m + 2), key=lambda m: m)
        synth = make_attack_synth(verifier, mt, edit)
        expected = {c for c in mt.universe(10) if edit(10, c)}
        self.assertEqual(set(synth.candidate_space(10)), expected)
        self.assertEqual(expected, {9, 11})   # sanity: edit_relation correctly excludes +/-2

    def test_threat_model_and_dual_agree(self):
        # phi must be in the Atom/Not/Op AST here so structural_dual (Definition 4) applies.
        verifier = lambda m, phi: (m % 2 == 0) if isinstance(phi, Atom) else (m % 2 == 1)
        edit = lambda a, b: abs(a - b) == 1
        mt = ModelType("int", universe=lambda m: (m - 1, m + 1), key=lambda m: m)
        logic = Logic("parity", verifier)
        tm = ThreatModel(logic, m=10, phi=Atom("even"), model_type=mt, edit_relation=edit)
        self.assertEqual(set(tm.candidate_space()), set(tm.dual().candidate_space()))


class TestEquivalenceWithMakeAttackSynth(unittest.TestCase):
    """The new ThreatModel layer must be provably just a relabeling of make_attack_synth, not a
    second implementation that could silently drift from it: reuse an existing, already-verified
    demo's own verifier/model_type/edit_relation/measure/m/phi (the LTL demo from demos.py) to
    build a ThreatModel, and require byte-identical SynthResult.attacks to calling
    make_attack_synth directly with the exact same inputs."""

    def test_ltl_demo_equivalence(self):
        demo = demo_ltl()
        # Rebuild the same (logic, model_type, edit_relation, measure) the LTL demo itself uses.
        mu = lambda t: sum(ltl_sat(t, demo.phi, i) for i in range(len(t))) / len(t)
        mt = ModelType("LTL trace", ltl_universe, key=lambda t: t)
        logic = Logic("LTL", ltl_sat)
        tm = ThreatModel(logic, m=demo.m, phi=demo.phi, model_type=mt, edit_relation=ltl_edit, measure=mu)

        direct = make_attack_synth(ltl_sat, mt, ltl_edit, measure=mu)
        r_direct = direct(demo.m, demo.phi)
        r_tm = tm.synthesize()

        self.assertEqual(r_direct.found, r_tm.found)
        self.assertEqual([a.model for a in r_direct.attacks], [a.model for a in r_tm.attacks])
        self.assertEqual([a.sigma for a in r_direct.attacks], [a.sigma for a in r_tm.attacks])
        self.assertEqual(r_direct.explored, r_tm.explored)

    def test_ltl_demo_dual_equivalence(self):
        demo = demo_ltl()
        mu = lambda t: sum(ltl_sat(t, demo.phi, i) for i in range(len(t))) / len(t)
        mt = ModelType("LTL trace", ltl_universe, key=lambda t: t)
        logic = Logic("LTL", ltl_sat)
        tm = ThreatModel(logic, m=demo.m, phi=demo.phi, model_type=mt, edit_relation=ltl_edit, measure=mu)

        direct = make_attack_synth(ltl_sat, mt, ltl_edit, measure=mu)
        r_direct = direct(demo.m, demo.phi, goal="dual")
        r_dual = tm.dual().realized()

        self.assertEqual(r_direct.found, r_dual.found)
        self.assertEqual([a.model for a in r_direct.attacks], [a.model for a in r_dual.attacks])


class TestMargraveIsGenuineConformance(unittest.TestCase):
    """tools/margrave.py is the one instance built through the literal Logic/ThreatModel API in a
    domain (ordered first-applicable rule matching) invented after that API was written -- the
    concrete test that the paper-conformance layer isn't just special-cased to the demos it was
    designed alongside. Re-derive its ThreatModel here and check the same three things
    TestAttackIsGenuine checks for a hand-rolled toy model."""

    def test_margrave_attack_is_genuine(self):
        from tools import margrave
        tm = margrave.build_threat_model()
        self.assertTrue(tm.logic.verifier(tm.m, tm.phi))          # precondition: m |= phi
        result = tm.synthesize(budget=2000, limit=1)
        self.assertTrue(result.found)
        mprime = result.best.model
        self.assertFalse(tm.logic.verifier(mprime, tm.phi))       # m' |/= phi
        self.assertIn(mprime, set(margrave.margrave_neighbors(tm.m)))  # m ~> m'

    def test_margrave_dual_realized(self):
        from tools import margrave
        tm = margrave.build_threat_model()
        d = tm.dual()
        self.assertEqual(d.phi_d, Not(Atom("deny")))
        result = d.realized(budget=2000, limit=1)
        self.assertTrue(result.found)
        self.assertTrue(tm.logic.verifier(result.best.model, d.phi_d))

    def test_margrave_severity_in_bounds(self):
        from tools import margrave
        tm = margrave.build_threat_model()
        result = tm.synthesize(budget=2000, limit=1)
        self.assertTrue(result.found)
        self.assertIsNotNone(result.best.sigma)
        self.assertGreaterEqual(result.best.sigma, 0.0)
        self.assertLessEqual(result.best.sigma, 1.0)


if __name__ == "__main__":
    unittest.main()
