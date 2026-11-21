"""papers/run_all.py -- run all four paper instances in sequence. Standard library only.
Run as `python3 papers/run_all.py` (from anywhere) or `python3 -m papers.run_all` (from the repo root)."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NAMES = ["oakley", "vonhippel", "bailluet", "wardega"]

if __name__ == "__main__":
    only = set(sys.argv[1:]) or set(NAMES)
    for n in NAMES:
        if n not in only:
            continue
        print(f"\n{'=' * 70}\n{n}\n{'=' * 70}")
        importlib.import_module(f"papers.{n}").main()
