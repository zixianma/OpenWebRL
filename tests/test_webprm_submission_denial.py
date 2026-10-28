import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from openwebrl import webprm_submission_denial as d


class DenialTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'v3/campaign';self.root.mkdir(parents=True)
        self.nonce='f'*24;self.intent=self.root/('submission-intent-'+self.nonce+'.json');self.sidecar=self.root/('submission-denial-'+self.nonce+'.json')
        batch=self.root/'hosting.sbatch';batch.write_text('#!/bin/bash\n#SBATCH --job-name=webprm-host-validate\n')
        d.h.put(self.intent,dict(nonce=self.nonce,batch_script=str(batch),argv=['sbatch','--hold','--comment=webprm-gitlab-host:'+self.nonce]))
        earlier=time.time()-180;os.utime(self.intent,(earlier,earlier))
        self.started=self.intent.stat().st_mtime
        attest=self.root/'attestation.json';d.h.put(attest,dict(status='approved_submissions_blocked_by_host_account_budget',no_new_job_ids=True,rejected_packages=['v3']))
        self.proof=self.root/'proof.json'
        def iso(seconds):return datetime.fromtimestamp(seconds,timezone.utc).isoformat()
        pairs=[]
        for number,offset in enumerate((1,65)):
            pair={}
            for kind in ('squeue','sacct'):
                argv=['squeue','-u','zixianma','-h','-o','%i|%V|%j|%k'] if kind=='squeue' else ['sacct','-u','zixianma','-S',iso(self.started-120),'-E','now','-X','-P','-n','-o','JobIDRaw,Submit,Start,End,State,JobName,Comment%150']
                path=self.root/f'{number}-{kind}.json'
                d.h.put(path,dict(argv=argv,tz='UTC',started_utc=iso(self.started+offset),completed_utc=iso(self.started+offset+1),returncode=0,stdout='',stderr=''))
                pair[kind]=dict(path=str(path),sha256=d.h.sha(path))
            pairs.append(pair)
        self.evidence=dict(schema='webprm_nonjob_submission_reconciliation_v1',intent=self.ref(self.intent),nonce=self.nonce,
            comment='webprm-gitlab-host:'+self.nonce,intent_mtime_ns=self.intent.stat().st_mtime_ns,
            root_denial_attestation=self.ref(attest),observation_pairs=pairs,matched_job_ids=[])
        self.refresh()

    def ref(self,path):return dict(path=str(path),sha256=d.h.sha(path))

    def refresh(self):
        self.proof.write_text(json.dumps(self.evidence))
        self.sidecar.write_text(json.dumps(dict(schema='webprm_submission_denial_v1',status='scheduler_rejected_no_job_independently_reconciled',intent=self.ref(self.intent),reconciliation=self.ref(self.proof),job_id=None)))

    def test_strict_nonjob_denial_preserves_original_intent_and_charges_no_invented_allocation(self):
        raw=self.intent.read_bytes();result=d.verified_denial(self.root,self.intent,self.sidecar)
        self.assertEqual(result['allocations_created'],0);self.assertIsNone(result['job_id'])
        self.assertEqual(raw,self.intent.read_bytes())

    def test_nonce_match_or_matching_job_name_time_is_not_nonjob_evidence(self):
        for stdout in ('42|2026-01-01|unrelated|'+self.nonce,
                       '42|'+datetime.fromtimestamp(self.started,timezone.utc).isoformat()+'|webprm-host-validate|(null)'):
            path=Path(self.evidence['observation_pairs'][0]['squeue']['path']);row=json.loads(path.read_text());row['stdout']=stdout
            path.write_text(json.dumps(row));self.evidence['observation_pairs'][0]['squeue']=self.ref(path);self.refresh()
            with self.assertRaises(d.h.Closed):d.verified_denial(self.root,self.intent,self.sidecar)

    def test_missing_pair_or_unfiltered_wrong_user_or_failing_scheduler_blocks(self):
        pristine=copy.deepcopy(self.evidence)
        original_row=json.loads(Path(pristine['observation_pairs'][1]['squeue']['path']).read_text())
        for variant in ('pair','user','returncode','stderr'):
            self.evidence=copy.deepcopy(pristine)
            if variant=='pair':self.evidence['observation_pairs'].pop()
            else:
                path=Path(self.evidence['observation_pairs'][1]['squeue']['path']);row=copy.deepcopy(original_row)
                if variant=='user':row['argv'][2]='other'
                if variant=='returncode':row['returncode']=1
                if variant=='stderr':row['stderr']='controller unavailable'
                path.write_text(json.dumps(row));self.evidence['observation_pairs'][1]['squeue']=self.ref(path)
            self.refresh()
            with self.subTest(variant=variant),self.assertRaises(d.h.Closed):d.verified_denial(self.root,self.intent,self.sidecar)

    def test_short_gap_tampered_source_or_fabricated_job_id_blocks(self):
        self.evidence['matched_job_ids']=['42'];self.refresh()
        with self.assertRaises(d.h.Closed):d.verified_denial(self.root,self.intent,self.sidecar)
        self.evidence['matched_job_ids']=[]
        path=Path(self.evidence['observation_pairs'][1]['squeue']['path']);row=json.loads(path.read_text());row['started_utc']=datetime.fromtimestamp(self.started+30,timezone.utc).isoformat()
        path.write_text(json.dumps(row));self.evidence['observation_pairs'][1]['squeue']=self.ref(path);self.refresh()
        with self.assertRaises(d.h.Closed):d.verified_denial(self.root,self.intent,self.sidecar)
        self.intent.write_text(self.intent.read_text()+' ')
        with self.assertRaises(d.h.Closed):d.verified_denial(self.root,self.intent,self.sidecar)


if __name__=='__main__':unittest.main()
