"""Bounded CPU checks: preserve reward math, batch identity and auxiliary rows."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from openwebrl.arm_gate_recovery import calibration_guard, replay_auxiliary, auxiliary_payload_limit, SCALE_MAX


class Guard(unittest.TestCase):
    def test_artifact_limit_tracks_gate_without_changing_rewards(self):
        self.assertEqual(auxiliary_payload_limit({}),2*1024**3)
        self.assertEqual(auxiliary_payload_limit({'candidate_gate':'distinct5'}),2*1024**3)
        for rule in ('response_index','action_class'):
            self.assertEqual(auxiliary_payload_limit({'candidate_gate':'min2','credit_assignment':rule}),4*1024**3)
        with self.assertRaises(ValueError):auxiliary_payload_limit({'candidate_gate':'unknown'})

    def fixture(self, ratio=.10167208094879619):
        report = dict(config=dict(candidate_gate='min2',credit_assignment='response_index',
            beta=.5,scored_fraction=.2,gate_scale_guard='pretraining_coverage_audit_v1'),
            variants={'0.5':dict(bonus_to_outcome_rms=ratio)})
        decision = dict(passed=False,checks=dict(labels=True,tasks=True,coverage=True,scale=False),beta=.5,q=.2)
        return report,decision

    def test_fixed_coefficient_and_other_checks_are_preserved(self):
        report,decision = self.fixture()
        original = deepcopy(decision)
        result = calibration_guard(report,decision)
        self.assertTrue(result['passed'])
        self.assertEqual(decision,original)
        self.assertEqual((result['beta'],result['q']),(.5,.2))
        self.assertTrue(result['scale_guard']['above_old_upper'])
        self.assertAlmostEqual(SCALE_MAX,.10*math.sqrt(25181/13471))
        decision['checks']['labels'] = False
        self.assertFalse(calibration_guard(report,decision)['passed'])

    def test_old_runs_untouched_invalid_and_extreme_cases_stop(self):
        report,decision = self.fixture()
        report['config'].pop('gate_scale_guard')
        self.assertIs(calibration_guard(report,decision),decision)
        for ratio in (None,float('nan'),float('inf'),.039,.14):
            report,decision = self.fixture(ratio)
            self.assertFalse(calibration_guard(report,decision)['passed'])
        for key,value in [('beta',.35),('scored_fraction',.3),('candidate_gate','distinct5'),
                          ('credit_assignment','wrong'),('gate_scale_guard','typo')]:
            report,decision = self.fixture();report['config'][key] = value
            with self.assertRaises(ValueError):calibration_guard(report,decision)


class SavedAuxiliary(unittest.TestCase):
    def fixture(self, tmp):
        root = Path(tmp)
        old = root/'old/iterations/0000';old.mkdir(parents=True)
        out = root/'new';out.mkdir()
        config = dict(rollout_id=0,policy_id='original:rollout0',checkpoint='original-sft',
            candidate_gate='min2',credit_assignment='response_index',output=str(out))
        manifest = {k:config[k] for k in ('rollout_id','policy_id','checkpoint','candidate_gate','credit_assignment')}
        manifest.update(beta=.5,q=.2,expected_windows=2,mixed_groups=48,tensor_file='failure_auxiliary.pt',
            total_failure_rows=400,failure_groups=8,records=[dict(advantage=.4)],coefficient=1/6)
        data = json.dumps(manifest).encode()
        (old/'failure_auxiliary.json').write_bytes(data)
        (old/'failure_auxiliary.pt').write_bytes(b'lossless-test-fixture')
        previous = dict(rows=48,batch_rows=48,admitted=8,admitted_distinct_tasks=6,
            admitted_distinct_action_counts={'2':8},task_counts={'task':48},label_reasons={'admitted':8},
            outcome_rms=.9,unit_bonus_rms=.18,additive_failure_groups=8,
            additive_failure_rows=400,additive_failure_labels=68,outcome_groups_preserved=48)
        (old/'calibration.json').write_text(json.dumps(previous))
        config['gate_replay_origin'] = dict(source=str(root/'old'),
            auxiliary_manifest_sha256=hashlib.sha256(data).hexdigest(),
            auxiliary_bytes=(old/'failure_auxiliary.pt').stat().st_size)
        report = {k:v for k,v in previous.items() if not k.startswith('additive_')}
        samples = [SimpleNamespace(group_index=i) for i in range(48)]
        return SimpleNamespace(global_batch_size=24), samples, dict(config=config), report

    def test_exact_auxiliary_reused_and_denominator_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,samples,current,report = self.fixture(tmp)
            self.assertTrue(replay_auxiliary(args,samples,current,report))
            out = Path(current['config']['output'])
            self.assertTrue((out/'failure_auxiliary.pt').is_symlink())
            manifest = json.loads((out/'failure_auxiliary.json').read_text())
            self.assertEqual(manifest['total_failure_rows'],400)
            self.assertEqual(manifest['failure_groups'],8)
            self.assertEqual(report['additive_failure_labels'],68)
            self.assertEqual(report['outcome_groups_preserved'],48)

    def test_live_integer_histogram_matches_json_saved_string_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,samples,current,report = self.fixture(tmp)
            # Counter keys are integers before serialization; JSON reloads
            # object keys as strings. Production replay must compare counts.
            report['admitted_distinct_action_counts'] = {2:8}
            self.assertTrue(replay_auxiliary(args,samples,current,report))

    def test_changed_or_ambiguous_histograms_are_rejected(self):
        for histogram in ({2:7},{2:8,3:1},{2:8,'2':8},{1:8},{2:8.0}):
            with self.subTest(histogram=histogram),tempfile.TemporaryDirectory() as tmp:
                args,samples,current,report = self.fixture(tmp)
                report['admitted_distinct_action_counts'] = histogram
                with self.assertRaises(ValueError):
                    replay_auxiliary(args,samples,current,report)

    def test_changes_and_later_batches_cannot_replay_silently(self):
        for kind in ('group','scale','labels','credit','manifest','size','windows'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                args,samples,current,report = self.fixture(tmp)
                if kind=='group':samples[-1].group_index=0
                if kind=='scale':report['outcome_rms']=.8
                if kind=='labels':report['admitted']=7
                if kind=='credit':current['config']['credit_assignment']='action_class'
                if kind=='manifest':current['config']['gate_replay_origin']['auxiliary_manifest_sha256']='bad'
                if kind=='size':current['config']['gate_replay_origin']['auxiliary_bytes']=0
                if kind=='windows':args.global_batch_size=48
                with self.assertRaises(ValueError):replay_auxiliary(args,samples,current,report)
        with tempfile.TemporaryDirectory() as tmp:
            args,samples,current,report = self.fixture(tmp)
            current['config']['rollout_id']=1
            self.assertFalse(replay_auxiliary(args,samples,current,report))


if __name__ == '__main__':unittest.main()
