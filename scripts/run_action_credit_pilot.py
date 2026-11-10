#!/usr/bin/env python3
"""Inspect a prepared action-credit bootstrap; actual train() is programmatic."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from openwebrl.action_credit_bootstrap import main

if __name__=='__main__':
    main()
