"""examples/run_all.py -- run both generality-stress-test instances in sequence. Standard library
only (adversarial_ml.py uses only math/itertools). Run as `python3 examples/run_all.py` (from
anywhere) or `python3 -m examples.run_all` (from the repo root)."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NAMES = ["adversarial_ml", "raft_election_safety"]

if __name__ == "__main__":
    only = set(sys.argv[1:]) or set(NAMES)
    for n in NAMES:
        if n not in only:
            continue
        print(f"\n{'=' * 70}\n{n}\n{'=' * 70}")
        importlib.import_module(f"examples.{n}").main()
