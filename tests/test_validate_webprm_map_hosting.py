"""Offline gates for the prospective owned fresh-schema Map reconstruction."""
import copy
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tarfile
import time
import unittest
import types
from unittest.mock import patch
from scripts import validate_webprm_map_hosting as owner
from scripts.prepare_webprm_map_validation import make_recipe
from tests.test_webprm_cms_hosting import snapshot


def phase(label):
    pair=dict(positive=dict(valid=True,score=1),negative=dict(valid=True,score=0))
    return dict(instance_id=label,base_identity='a'*64,database_before='b'*64,database_marker_before=0,database_marker_after=1,
        closure=dict(remaining_owned_processes={}),backend_pass=dict(tiles=True,geocode=True,routes=True),
        browser=dict(released_map_tasks=26,home=dict(status=200),initial_cookie_count=0,storage_state_sha256=label*64,
            evaluator={name:copy.deepcopy(pair) for name in ('task90_url_match','task87_program_html','task159_program_html')},
            ui_routes=dict(Foot='native route',Bicycle='native route')))


class MapOwnerTests(unittest.TestCase):
    def test_failed_ui_keeps_raw_native_evidence_before_rejection(self):
        result=phase('a');result['browser']['ui_routes']['Foot']='error: preserved diagnostic'
        r=types.SimpleNamespace(backends=lambda value:None,browser=lambda value,**kwargs:None)
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            out=Path(folder)
            with self.assertRaises(owner.h.Closed):owner.record_controls(r,result,out)
            self.assertEqual(owner.h.read(out/'native-controls-raw.json')['browser']['ui_routes']['Foot'],'error: preserved diagnostic')
            self.assertEqual(owner.h.read(out/'native-controls-verdict.json')['status'],'failed')

    def test_browser_exception_preserves_prior_backend_and_error(self):
        def browser(value,**kwargs):raise RuntimeError('original browser error')
        r=types.SimpleNamespace(backends=lambda value:None,browser=browser)
        result=phase('a')
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            out=Path(folder)
            with self.assertRaisesRegex(RuntimeError,'original browser error'):owner.record_controls(r,result,out)
            self.assertEqual(owner.h.read(out/'native-controls-verdict.json')['status'],'browser_failed')
            self.assertTrue((out/'backend-controls-raw.json').exists())

    def test_complete_reset_retains_reconstruction_caveat(self):
        gate=owner.reset_pass(phase('a'),phase('b'))
        self.assertTrue(gate['fresh_reset_verified'])
        self.assertFalse(gate['original_state_equivalence_verified'])
        self.assertFalse(gate['browse_equivalence_supported'])

    def test_missing_native_backend_invalid_evaluator_or_route_fails(self):
        for variant in ('backend','native','route','count'):
            result=phase('a')
            if variant=='backend':result['backend_pass']['geocode']=False
            if variant=='native':result['browser']['evaluator']['task87_program_html']['positive']['valid']=False
            if variant=='route':result['browser']['ui_routes']['Foot']='error: timeout'
            if variant=='count':result['browser']['released_map_tasks']=25
            with self.subTest(variant=variant),self.assertRaises(owner.h.Closed):owner.controls_pass(result)

    def test_reset_rejects_mutation_leak_same_state_session_or_open_process(self):
        for variant in ('marker','state','session','open','schema','initial_cookie'):
            a,b=phase('a'),phase('b')
            if variant=='marker':b['database_marker_before']=1
            if variant=='state':b['instance_id']='a'
            if variant=='session':b['browser']['storage_state_sha256']='a'*64
            if variant=='open':b['closure']['remaining_owned_processes']={'23':100}
            if variant=='schema':b['database_before']='c'*64
            if variant=='initial_cookie':b['browser']['initial_cookie_count']=1
            with self.subTest(variant=variant),self.assertRaises(owner.h.Closed):owner.reset_pass(a,b)

    def test_missing_approval_stops_before_scheduler_and_execution(self):
        p=dict(protocol=owner.PROTOCOL,service_plan=dict(sha256='a'*64))
        with patch.object(owner.subprocess,'check_output') as command,patch.object(owner.h,'own') as execute:
            with self.assertRaises(owner.h.Closed):owner.approved(p,{})
            command.assert_not_called();execute.assert_not_called()

    def test_exact_resource_caps(self):
        self.assertEqual(owner.RESOURCES['gpus'],0)
        self.assertEqual(owner.RESOURCES['cpus'],8)
        self.assertEqual(owner.RESOURCES['memory_gib'],128)
        self.assertEqual(owner.RESOURCES['total_seconds'],14400)
        self.assertEqual(owner.LIMITS['api_usd'],0)

    def test_actual_map_scheduler_rejects_other_cpu_memory_gpu_or_time(self):
        profile=owner.profile(dict(service_plan=dict(sha256='a'*64)))
        raw=snapshot(cpus=8,memory=128*1024,minutes=240,end=14500).replace('cpu=4,node=1,mem=16G','cpu=8,node=1,mem=128G')
        self.assertEqual(owner.h.scheduler(raw,'42',requested_seconds=14400,now=110,profile=profile)['work_deadline'],13900)
        for wrong in (raw.replace('mem=128G','mem=128G,gres/gpu=1'),snapshot(),raw.replace('"number": 131072','"number": 65536')):
            with self.assertRaises(owner.h.Closed):owner.h.scheduler(wrong,'42',requested_seconds=14400,now=110,profile=profile)

    def test_reduced_recipe_has_no_historical_owner_and_cli_dsn_is_one_shell_value(self):
        source=owner.N/'map-bringup-probe-v1/probe.py'
        compiled=make_recipe(source.read_text());namespace={'__file__':'/tmp/frozen-recipe.py'}
        exec(compile(compiled,str(source),'exec'),namespace)
        self.assertNotIn('main',namespace);self.assertNotIn('inside',namespace)
        for name in ('TILE_CTL','NOM_CTL','DB_CTL','WEB_CTL'):
            text=namespace[name]
            self.assertNotIn('rm -',text);self.assertNotIn('HOME=',text)
            result=subprocess.run(['bash','-n'],input=text,text=True,capture_output=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr)
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            executable=Path(folder)/'nominatim';executable.write_text('#!/bin/sh\nprintf "%s" "$NOMINATIM_DATABASE_DSN"\n');executable.chmod(0o700)
            env=dict(os.environ,PATH=folder+':'+os.environ['PATH'],BASH_ENV='/dev/null',ENV='/dev/null')
            for line in namespace['NOM_CTL'].splitlines():
                if 'NOMINATIM_DATABASE_DSN=' not in line:continue
                command=re.search(r'-c "([^"]+)"',line).group(1).replace('cd /nominatim && ','')
                result=subprocess.run(['bash','-c',command],env=env,capture_output=True,text=True,timeout=5)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(result.stdout,'pgsql:dbname=nominatim;user=nominatim')

    @unittest.skipUnless(hasattr(tarfile,'data_filter'),'Production Python 3.12 safe extraction required')
    def test_safe_selected_extraction_rejects_escape_links_and_byte_cap(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root=Path(folder);path=root/'input.tar'
            with tarfile.open(path,'w') as archive:
                item=tarfile.TarInfo('selected/escape');item.type=tarfile.SYMTYPE;item.linkname='/outside';archive.addfile(item)
            with self.assertRaises(tarfile.FilterError):owner.extract(path,root/'state','selected/',time.time()+10,1024)
            path=root/'large.tar'
            with tarfile.open(path,'w') as archive:
                item=tarfile.TarInfo('selected/file');item.size=2;archive.addfile(item,io.BytesIO(b'ab'))
            with self.assertRaises(owner.h.Closed):owner.extract(path,root/'bounded','selected/',time.time()+10,1)


if __name__=='__main__':unittest.main()
