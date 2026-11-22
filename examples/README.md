# examples/

Two further instances, each in a model class that neither `tools/` (Example 4) nor `papers/`
(Example 5) names -- added as further evidence for Section 6's generality claim, beyond the 21
instances the paper itself counts. Unlike `tools/` and `papers/`, these are not tied to a specific
cited tool or paper's own definitions; each is grounded in a well-established threat model from its
field (cited in the file's own docstring) rather than one specific piece of software.

| File | Domain | Model / adversary |
|---|---|---|
| `adversarial_ml.py` | adversarial machine learning | a fixed linear classifier and a labeled point; adversary is the standard L-infinity epsilon-ball bounded perturbation (Goodfellow et al. 2015; Madry et al. 2018), searched by grid enumeration instead of gradient ascent |
| `raft_election_safety.py` | distributed consensus | a 3-node Raft cluster within one term; adversary is a lost/forgotten persisted vote record letting one node vote for two candidates in the same term -- exactly the failure Raft's own paper (Ongaro & Ousterhout, 2014) says persisting `votedFor` exists to prevent |

Run either directly: `python3 examples/adversarial_ml.py`, `python3 examples/raft_election_safety.py`.

Both use the same unmodified `attack_synth.py` engine as every other instance in this repository --
no change to `ModelType` or the search procedure was needed for either.
