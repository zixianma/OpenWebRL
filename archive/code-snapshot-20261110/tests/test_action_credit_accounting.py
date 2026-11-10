"""Actual sacct-format fixtures; subprocess transport is mocked, no Slurm calls."""
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from openwebrl.action_credit_accounting import AllocationAccountingProvider
from openwebrl.action_credit_acceptance import audit_allocation
from openwebrl.action_credit_native_controller import _hash, _json, _write
from openwebrl.action_credit_state import CampaignManifest
from test_action_credit_state import manifest_data, sha


class AccountingTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='action-credit-accounting-',dir='/tmp')
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        data=manifest_data(); data['budget_caps']['gpu_seconds']=8*86400
        self.manifest=CampaignManifest.from_dict(data)
        self.ledger=dict(manifest_sha256=self.manifest.sha256,approved_gpu_seconds_cap=8*86400,
            attempts=[dict(job_id=job) for job in ('100','101','102')])
        self.raw='100|FAILED|120|cpu=64,gres/gpu=8,mem=960G|\n101|RUNNING|60|cpu=64,gres/gpu=8|\n102|PENDING|0||\n'

    def provider(self,**changes):
        index=len(list(self.root.glob('ledger-*.json')))
        ref=_write(self.root/f'ledger-{index}.json',dict(self.ledger,**changes))
        return AllocationAccountingProvider(manifest=self.manifest,attempt_ledger_ref=ref,output_root=self.root/'snapshots')

    def query(self,provider,raw=None,returncode=0):
        with patch('openwebrl.action_credit_accounting.subprocess.run',return_value=subprocess.CompletedProcess(
                [],returncode,self.raw if raw is None else raw,'fixture stderr')) as run:
            ref=provider()
        return ref,run

    def test_every_attempt_is_charged_and_exact_parent_query_is_preserved(self):
        provider=self.provider(); before=time.time()
        ref,run=self.query(provider)
        result=_json(ref)
        self.assertGreaterEqual(result['observed_unix'],before)
        self.assertEqual(result['gpu_seconds_used'],8*(120+60))
        self.assertEqual([row['state'] for row in result['attempts']],['FAILED','RUNNING','PENDING'])
        self.assertEqual(Path(result['scheduler_records'][0]['path']).read_text(),self.raw)
        command=run.call_args.args[0]
        self.assertEqual(command,['sacct','-n','-P','-X','--jobs=100,101,102','--format=JobIDRaw,State,ElapsedRaw,AllocTRES'])
        self.assertEqual(run.call_args.kwargs['timeout'],30)
        self.assertEqual(audit_allocation(ref,self.manifest)['running_gpus'],8)
        # sacct -P omits the final delimiter; accept the optional -p delimiter
        # without accidentally removing a pending row's empty fourth field.
        no_trailing='\n'.join(line[:-1] for line in self.raw.splitlines())+'\n'
        second,_=self.query(provider,no_trailing)
        self.assertEqual(_json(second)['gpu_seconds_used'],result['gpu_seconds_used'])

    def test_source_manifest_cap_and_duplicate_job_ids_rejected_before_query(self):
        for changes in (dict(manifest_sha256=sha('other campaign')),dict(approved_gpu_seconds_cap=8*86401),
                        dict(attempts=[dict(job_id='100'),dict(job_id='100')]),dict(attempts=[dict(job_id='100.batch')])):
            with self.subTest(changes=changes),patch('openwebrl.action_credit_accounting.subprocess.run') as run:
                with self.assertRaises(ValueError): self.provider(**changes)
                run.assert_not_called()

    def test_missing_duplicate_wrong_gpu_and_nonzero_pending_records_fail_closed(self):
        raws=(self.raw.splitlines()[0]+'\n',self.raw+self.raw.splitlines()[0]+'\n',
              self.raw.replace('gres/gpu=8','gres/gpu=4',1),self.raw.replace('PENDING|0','PENDING|9'),
              self.raw.replace('FAILED|120','FAILED|999999'))
        provider=self.provider()
        for raw in raws:
            with self.subTest(raw=raw),self.assertRaises(ValueError): self.query(provider,raw)

    def test_command_failure_retains_raw_evidence_without_admission(self):
        provider=self.provider()
        with self.assertRaisesRegex(RuntimeError,'nonzero'):
            self.query(provider,returncode=1)
        self.assertEqual(len(list((self.root/'snapshots').glob('*/query-result.json'))),1)
        self.assertFalse(list((self.root/'snapshots').glob('*/accounting.json')))
        with patch('openwebrl.action_credit_accounting.subprocess.run',side_effect=subprocess.TimeoutExpired('sacct',30,output=b'partial')):
            with self.assertRaisesRegex(RuntimeError,'query failed'): provider()
        self.assertEqual(len(list((self.root/'snapshots').glob('*/query-failure.json'))),1)

    def test_observation_precedes_query_and_stale_snapshot_cannot_pass(self):
        provider=self.provider(); started=time.time()
        with patch('openwebrl.action_credit_accounting.time.time',side_effect=[started,started+301]), \
                patch('openwebrl.action_credit_accounting.subprocess.run',return_value=subprocess.CompletedProcess([],0,self.raw,'')):
            with self.assertRaisesRegex(ValueError,'stale'): provider()
        path=next((self.root/'snapshots').glob('*/accounting.json'))
        saved=_json(dict(path=str(path),sha256=_hash(path)))
        self.assertEqual(saved['observed_unix'],started)

    def test_repeat_snapshot_keeps_prior_failure_cost_and_does_not_reset_cap(self):
        provider=self.provider(); first,_=self.query(provider)
        second,_=self.query(provider,self.raw.replace('RUNNING|60','RUNNING|80'))
        a,b=_json(first),_json(second)
        self.assertNotEqual(first['path'],second['path'])
        self.assertEqual(b['gpu_seconds_used']-a['gpu_seconds_used'],8*20)
        self.assertEqual(b['gpu_seconds_used'],8*(120+80))
        self.assertEqual(b['approved_gpu_seconds_cap'],8*86400)

    def test_actual_never_started_cancellation_is_retained_at_zero_cost(self):
        self.ledger['attempts']=[dict(job_id=job) for job in ('355629','101','102')]
        provider=self.provider()
        raw='355629|CANCELLED by 1385309|0||\n101|RUNNING|60|gres/gpu=8|\n102|PENDING|0||\n'
        ref,_=self.query(provider,raw)
        result=_json(ref)
        self.assertEqual(result['gpu_seconds_used'],8*60)
        self.assertEqual(result['attempts'][0],dict(job_id='355629',state='CANCELLED',elapsed_seconds=0,gpus=8))
        self.assertEqual(audit_allocation(ref,self.manifest)['attempt_ids'],['101','102','355629'])
        for bad in (raw.replace('1385309|0||','1385309|1||'),
                    raw.replace('1385309|0||','1385309|0|cpu=64|'),
                    raw.replace('1385309|0||','1385309|0|gres/gpu=4|')):
            with self.subTest(raw=bad),self.assertRaises(ValueError): self.query(provider,bad)


if __name__=='__main__': unittest.main()
