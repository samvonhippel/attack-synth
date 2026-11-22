"""
examples/adversarial_ml.py -- a generality stress test: attack synthesis against a trained
classifier's robustness, in the style of the standard Lp-norm bounded-perturbation threat model
for adversarial examples (Goodfellow, Shlens & Szegedy, "Explaining and Harnessing Adversarial
Examples", ICLR 2015; Madry et al., "Towards Deep Learning Models Resistant to Adversarial
Attacks", ICLR 2018). This is not a re-implementation of either paper's own algorithm (FGSM's
gradient step, or PGD's iterated projected gradient ascent) -- there is no autodiff here -- but the
*threat model itself* (a fixed classifier, an input the model classifies correctly, and an
adversary confined to an L-infinity epsilon-ball around that input) is exactly theirs, searched
here by grid enumeration instead of gradient ascent.

Model: a fixed linear classifier (w, b) over R^2, and a single labeled input point x with its true
label. Nothing about the classifier is adversary-controlled -- only the input.
Property: the classifier correctly labels x (the "robustness at a point" property both cited
papers define their threat model around).
Adversary: an L-infinity epsilon-ball around x -- the standard bounded-perturbation adversary of
both cited papers. This instance's neighbors() enumerates a fixed grid over that ball rather than
following a gradient, since attack_synth.py's engine only needs a candidate generator, not a
gradient.
Verifier: direct evaluation of sign(w . x + b) against the true label -- exact, not approximate,
since the classifier itself is exactly linear.

This file is deliberately outside tools/ (Example 4) and papers/ (Example 5): it demonstrates the
engine on a model class (a continuous-input machine-learning classifier) that the paper's own
comparison examples do not name, as evidence for Section 6's generality claim beyond its own
twenty-one counted instances.
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_synth import ModelType, make_attack_synth  # noqa: E402


@dataclass(frozen=True)
class Classifier:
    w: tuple   # (w1, w2)
    b: float

    def margin(self, x: tuple) -> float:
        """w . x + b -- positive means predicted label +1, negative means predicted label -1."""
        return self.w[0] * x[0] + self.w[1] * x[1] + self.b

    def predict(self, x: tuple) -> int:
        return 1 if self.margin(x) >= 0 else -1


CLF = Classifier(w=(1.0, 1.0), b=-1.0)        # decision boundary: x1 + x2 = 1
TRUE_LABEL = 1
X0 = (0.8, 0.8)                                # margin = 0.6 > 0: correctly classified, confidently


def confidence(x: tuple) -> float:
    """A [0, 1]-valued measure mu(x): a squashed margin, used only as this instance's severity
    weighting (not part of the classifier itself) -- 0.5 exactly at the decision boundary."""
    import math
    return 1.0 / (1.0 + math.exp(-CLF.margin(x)))


def epsilon_ball_neighbors(x: tuple, eps=0.5, steps=5):
    """The standard L-infinity epsilon-ball adversary: every grid point within +/- eps of x on
    each coordinate (a discretization of the continuous ball both cited papers define)."""
    deltas = [eps * k / steps for k in range(-steps, steps + 1)]
    for dx, dy in itertools.product(deltas, deltas):
        if dx == 0.0 and dy == 0.0:
            continue
        yield (round(x[0] + dx, 6), round(x[1] + dy, 6))


def find_adversarial_example(x0: tuple, true_label: int, eps=0.5, steps=5, budget=5000):
    """phi: the classifier predicts true_label at the (perturbed) point. An attack is any point
    within the epsilon-ball whose prediction flips -- an adversarial example."""
    verifier = lambda x, _phi: CLF.predict(x) == true_label
    mt = ModelType("point in R^2", neighbors=lambda x: epsilon_ball_neighbors(x, eps, steps),
                   key=lambda x: x)
    synth = make_attack_synth(verifier, mt, measure=confidence, max_steps=1)
    return synth(x0, object(), budget=budget, limit=1)


def main():
    print("Adversarial-ML-style -- L-infinity epsilon-ball bounded-perturbation attack search")
    print(f"  classifier: sign({CLF.w[0]}*x1 + {CLF.w[1]}*x2 + {CLF.b})   boundary: x1+x2=1")
    print(f"  baseline x0={X0}  margin={CLF.margin(X0):.3f}  "
          f"predict={CLF.predict(X0)} (true label {TRUE_LABEL})  confidence={confidence(X0):.3f}")

    result = find_adversarial_example(X0, TRUE_LABEL, eps=0.5, steps=5)
    assert result.found, "expected an adversarial example within this epsilon-ball"
    a = result.best
    xprime = a.model
    # Genuineness check, mirroring tests/test_conformance.py's TestAttackIsGenuine: xprime is
    # within the declared epsilon-ball (m ~> m') and genuinely flips the prediction (m' |/= phi).
    assert max(abs(xprime[0] - X0[0]), abs(xprime[1] - X0[1])) <= 0.5 + 1e-9
    assert CLF.predict(xprime) != TRUE_LABEL
    print(f"  adversarial example found: x'={xprime}  margin={CLF.margin(xprime):.3f}  "
          f"predict={CLF.predict(xprime)} (flipped)  confidence={confidence(xprime):.3f}")
    print(f"    L-infinity perturbation: {max(abs(xprime[0]-X0[0]), abs(xprime[1]-X0[1])):.3f}  "
          f"(epsilon budget: 0.5)")
    print(f"    severity sigma = {a.sigma:.3f}  (explored {result.explored} grid points "
          f"in {result.seconds * 1e3:.1f} ms)")

    # Robustness check in the *other* direction: confirm a smaller ball around a confidently
    # classified point contains no adversarial example, i.e. the search correctly reports "no
    # attack" rather than always finding one regardless of budget.
    safe = find_adversarial_example((0.9, 0.9), 1, eps=0.05, steps=5)
    print(f"  smaller ball (eps=0.05) around (0.9, 0.9): adversarial example found = {safe.found} "
          f"(expected False -- margin there is {CLF.margin((0.9, 0.9)):.3f}, "
          f"ball radius can't flip the sign)")
    assert not safe.found


if __name__ == "__main__":
    main()
