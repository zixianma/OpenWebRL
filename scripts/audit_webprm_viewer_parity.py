#!/usr/bin/env python3
"""Replay pinned viewer validation on saved pilot proposals; zero API/browser calls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.webprm_arbiter_harness import load_arbiter_harness
from openwebrl.webprm_viewer_parity import load_viewer_helpers


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(source, campaign):
    helpers = load_viewer_helpers(source)
    harness = load_arbiter_harness(source)
    totals, rows, evidence = Counter(), [], {}
    for path in sorted(Path(campaign).glob('cell-*/episodes/*/*/proposal-*.json')):
        proposal = json.loads(path.read_text())
        evidence[str(path)] = sha(path)
        totals['saved_proposals'] += 1
        if proposal['terminal']:
            totals['detector_overrides_not_policy_filtered'] += 1
            continue
        folder = path.parent
        rollout_path = folder / 'rollout.json'
        rollout = json.loads(rollout_path.read_text()) if rollout_path.exists() else []
        if rollout_path.exists(): evidence[str(rollout_path)] = sha(rollout_path)
        turn = int(path.stem.split('-')[-1])
        record = next((r for r in rollout if r['turn'] == turn), {})
        blocks = harness.parse_policy_response(proposal['raw_response']).candidates
        available = helpers.extract_elements(proposal['context']['text_observation'])
        accepted = []
        for slot, block in enumerate(blocks[:proposal['requested_raw_slots']]):
            totals['parsed_policy_slots'] += 1
            passed = helpers.validate_candidate(block, available)
            current = next((c for c in proposal['candidates'] if c['raw_slot'] == slot), None)
            if current: totals['adapter_accepted_slots'] += 1
            if passed: totals['viewer_accepted_slots'] += 1
            if current and not passed:
                totals['adapter_slots_rejected_by_viewer'] += 1
                selected = record.get('selected_index')
                selected_here = (isinstance(selected, int) and
                    proposal['candidates'][selected]['raw_slot'] == slot)
                execution_error = record.get('execution_error') if selected_here else None
                if selected_here: totals['selected_slots_rejected_by_viewer'] += 1
                if execution_error: totals['errored_selected_slots_rejected_by_viewer'] += 1
                rows.append(dict(proposal=str(path), turn=turn, raw_slot=slot,
                    action=block.action, selected=selected_here, execution_error=execution_error))
            if passed: accepted.append(slot)
        if not accepted:
            totals['policy_proposals_with_no_viewer_accepted_slot'] += 1
    return dict(schema_version=1, status='offline_source_differential_complete',
        scope='Closed 20-episode development pilot; validation counterfactual only',
        source=helpers.manifest, totals=dict(totals), differences=rows,
        evidence_sha256=evidence, provider_calls=0, browser_calls=0,
        historical_outcomes_changed=False, counterfactual_success_rate=None,
        limitation='Rejection does not prove recovery; future actions and outcomes are unknown')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    allowed = [Path('/gpfs/projects/krishna/zixianma/OpenWebRL'),
               Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'), Path('/tmp')]
    if not any(root in output.parents for root in allowed):
        raise ValueError('Output must be beneath an approved writable root')
    result = audit(args.source, args.campaign)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['totals'], sort_keys=True))


if __name__ == '__main__': main()
