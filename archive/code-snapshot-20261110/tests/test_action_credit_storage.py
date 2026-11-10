"""Offline real-lock/quota-format tests; never query GPFS or alter live ledgers."""
from concurrent.futures import ThreadPoolExecutor
import copy
import getpass
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from openwebrl import action_credit_storage as storage
from openwebrl.action_credit_native_controller import _json, _write
from openwebrl.action_credit_state import CampaignStore, CampaignManifest
from test_action_credit_state import manifest_data


class StorageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='action-credit-storage-',dir='/tmp')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.shared=self.root/'shared'; self.shared.mkdir()
        shutil.copyfile(storage.LEGACY_ROOT/'storage_reservations.py',self.shared/'storage_reservations.py')
        self.patch=patch.object(storage,'LEGACY_ROOT',self.shared); self.patch.start(); self.addCleanup(self.patch.stop)
        (self.shared/'storage-reservations.lock').touch()
        self.legacy=dict(token='old-token',lineage='batch56',job_id='123',reserved_bytes=storage.TIB)
        self.initial=dict(active={'batch56':self.legacy},history=[{'event':'old'}])
        (self.shared/'storage-reservations.json').write_text(json.dumps(self.initial))
        _write(self.shared/'storage-plan.json',dict(approved=True,preparation_verified=True,
            verified_sufficient=True,active_workers_excluded=True,minimum_shared_free_bytes=2*storage.TIB,
            maximum_future_stage_per_lineage_bytes=dict(batch56=storage.TIB,uniform8=storage.TIB),
            full300_evaluation_reservation_bytes=storage.TIB))
        self.headroom=10*storage.TIB; self.owned_bytes=1024; self.calls=[]

    def provider(self,name='pilot',growth=storage.TIB,**changes):
        owned=self.root/name; owned.mkdir()
        data=manifest_data(); data['campaign_id']=name
        store=CampaignStore.create(owned/'ledger',CampaignManifest.from_dict(data))
        account=_write(owned/'accounting.json',dict(campaign_id=name,cleanup_authorized=False,
            storage=dict(protective_ceiling_bytes=5*storage.TIB,global_free_floor_bytes=2*storage.TIB)))
        source=_write(owned/'observed-inventory.json',dict(description='CPU fixture measured inventory',bytes=growth))
        measure=_write(owned/'measurements.json',dict(method='CPU fixture derived retained inventory',
            measurements={'remaining':growth},source_artifacts=[source]))
        term=dict(measurement_ref=measure,measurement_key='remaining',remaining_units=1,stress_numerator=1,stress_denominator=1)
        forecast=dict(schema_version=1,manifest_sha256=store.manifest.sha256,coordination_root=str(self.shared),
            resource_accounting_ref=account,owned_roots=[str(owned)],scope=storage.SCOPE,
            stages={s:[copy.deepcopy(term)] for s in storage.STAGES})
        forecast.update(changes)
        ref=_write(owned/'forecast.json',forecast)
        return store,storage.StorageAdmissionProvider(forecast_ref=ref,output_root=owned/'admissions',job_id='456',coordination_root=self.shared)

    def transport(self,command,**kwargs):
        self.calls.append(command)
        if command[0]=='du':
            raw=''.join(str(self.owned_bytes)+'\t'+p+'\n' for p in command[command.index('--')+1:])
        else:
            self.assertEqual(command,['/usr/lpp/mmfs/bin/mmlsquota','-u',getpass.getuser(),'-Y','mmfs1:scrubbed'])
            limit=30*storage.TIB//1024; used=(30*storage.TIB-self.headroom)//1024
            raw=('mmlsquota:user:HEADER:name:filesetname:blockUsage:blockQuota:blockLimit:blockInDoubt:blockGrace\n'
                 f'mmlsquota:user:0:{getpass.getuser()}:scrubbed:{used}:{limit}:{limit+1000000}:0:7 days\n')
        return subprocess.CompletedProcess(command,0,raw,'')

    def read(self):return json.loads((self.shared/'storage-reservations.json').read_text())

    def test_shared_legacy_entries_preserved_and_seen_by_original_reserver(self):
        store,provider=self.provider()
        with patch.object(storage.subprocess,'run',side_effect=self.transport):
            ref=provider(store,'native-startup'); second=provider(store,'initial')
        evidence=_json(ref); ledger=self.read()
        self.assertEqual(ledger['active']['batch56'],self.legacy)
        self.assertEqual(evidence['required_headroom_bytes'],4*storage.TIB)
        self.assertEqual(len(ledger['active']),2)
        self.assertEqual(evidence['reservation']['token'],_json(second)['reservation']['token'])
        self.assertEqual(ledger['history'][0],self.initial['history'][0])
        # Actual old function counts the new namespaced record, without modification.
        import sys
        with patch.object(sys,'path',[str(Path(__file__).parents[1]/'scripts'),*sys.path]):
            spec=importlib.util.spec_from_file_location('storage_legacy_fixture',self.shared/'storage_reservations.py')
            module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        plan=json.loads((self.shared/'storage-plan.json').read_text())
        requested,minimum=module.required_bytes(plan,ledger['active'],'uniform8','training',2)
        self.assertEqual(requested,storage.TIB); self.assertEqual(minimum,5*storage.TIB)

    def test_concurrent_pilots_cannot_both_admit_against_same_remaining_space(self):
        first=self.provider('a'); second=self.provider('b'); self.headroom=4*storage.TIB
        def admit(pair):
            try: pair[1](pair[0],'native-startup'); return True
            except ValueError as exc:
                self.assertIn('headroom',str(exc)); return False
        with patch.object(storage.subprocess,'run',side_effect=self.transport),ThreadPoolExecutor(max_workers=2) as pool:
            outcomes=list(pool.map(admit,[first,second]))
        self.assertEqual(sorted(outcomes),[False,True]); self.assertEqual(len(self.read()['active']),2)

    def test_floor_and_actual_usage_and_forecast_cap_fail_closed(self):
        store,provider=self.provider(); self.headroom=4*storage.TIB-1024
        with patch.object(storage.subprocess,'run',side_effect=self.transport):
            with self.assertRaisesRegex(ValueError,'headroom'):provider(store,'initial')
        self.assertEqual(self.read(),self.initial)
        self.headroom=10*storage.TIB; self.owned_bytes=5*storage.TIB
        with patch.object(storage.subprocess,'run',side_effect=self.transport):
            with self.assertRaisesRegex(ValueError,'ceiling'):provider(store,'initial')
        bigstore,bigprovider=self.provider('too-big',growth=6*storage.TIB)
        with patch.object(storage.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'ceiling'):bigprovider(bigstore,'initial')
            run.assert_not_called()

    def test_stale_unknown_and_failed_quota_grant_no_reservation(self):
        store,provider=self.provider()
        original=provider._query
        def stale(command,folder,label):
            text,stamp,ref=original(command,folder,label)
            return text,stamp-120 if label=='quota' else stamp,ref
        with patch.object(storage.subprocess,'run',side_effect=self.transport),patch.object(provider,'_query',side_effect=stale):
            with self.assertRaisesRegex(ValueError,'stale'):provider(store,'initial')
        for raw in ('unknown quota',):
            def malformed(command,**kwargs):
                return self.transport(command,**kwargs) if command[0]=='du' else subprocess.CompletedProcess(command,0,raw,'')
            with patch.object(storage.subprocess,'run',side_effect=malformed),self.assertRaises(ValueError):provider(store,'initial')
        with patch.object(storage.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','quota unavailable')):
            with self.assertRaisesRegex(ValueError,'query failed'):provider(store,'initial')
        self.assertEqual(self.read(),self.initial)

    def test_existing_reservation_never_expires_or_changes_forecast_on_retry(self):
        store,provider=self.provider()
        with patch.object(storage.subprocess,'run',side_effect=self.transport):provider(store,'initial')
        ledger=self.read(); key='action-credit:pilot'; ledger['active'][key]['epoch']=1
        (self.shared/'storage-reservations.json').write_text(json.dumps(ledger))
        provider.job_id='789'
        with patch.object(storage.subprocess,'run',side_effect=self.transport):ref=provider(store,'optimizer-update-0')
        self.assertEqual(_json(ref)['reservation']['job_ids'],['456','789'])
        changed=_json(provider.forecast_ref); changed['stages']['initial'][0]['remaining_units']=2
        provider.forecast_ref=_write(self.root/'changed-forecast.json',changed)
        with patch.object(storage.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'reconciliation'):provider(store,'initial')
            run.assert_not_called()

    def test_unmeasured_stage_missing_sources_or_changed_legacy_code_rejected(self):
        store,provider=self.provider(stages={})
        with self.assertRaisesRegex(ValueError,'Every bootstrap stage'):provider(store,'initial')
        store,provider=self.provider('other')
        with self.assertRaisesRegex(ValueError,'Unknown storage stage'):provider(store,'future')
        (self.shared/'storage_reservations.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'compatibility review'):provider(store,'initial')
        with self.assertRaisesRegex(ValueError,'global coordination'):
            storage.StorageAdmissionProvider(forecast_ref=provider.forecast_ref,output_root=self.root/'bad',job_id='456',coordination_root=self.root)


if __name__=='__main__':unittest.main()
