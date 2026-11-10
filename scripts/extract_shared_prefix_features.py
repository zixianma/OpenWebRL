#!/usr/bin/env python3
"""Replay eight frozen prompts once inside the approved shared-prefix allocation."""

import argparse
import importlib.util
from pathlib import Path


PILOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-action-chunk-outcome-20261017-v2')
MODEL = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = PILOT / 'source/run_action_chunk_outcome_pilot.py'
    spec = importlib.util.spec_from_file_location('frozen_action_chunk_replay', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = module.load_rows(args.panels)
    if len(rows) != 8:
        raise ValueError('Expected the eight frozen external states')
    module.replay(rows, MODEL, args.output, batch_size=8)


if __name__ == '__main__':
    main()
