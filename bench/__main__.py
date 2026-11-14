"""python -m bench  ==  bench/run_all.sh: the full Phase 7 measurement run (see bench/run_all.py, bench/README.md)."""
import sys

from .run_all import main

sys.exit(main())
