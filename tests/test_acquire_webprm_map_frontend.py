import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('map_acquire',Path(__file__).resolve().parents[1]/'scripts/acquire_webprm_map_frontend.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class MapAcquisitionTests(unittest.TestCase):
    def scheduler(self):
        def n(value):return {'set':True,'infinite':False,'number':value}
        return {'errors':[],'jobs':[dict(job_id=123,user_id=os.getuid(),job_state=['RUNNING'],restart_cnt=0,
            account='zixianma',partition='gpu-h200',qos='normal',cpus=n(2),node_count=n(1),memory_per_node=n(4096),
            start_time=n(100),end_time=n(1903),time_limit=n(30),tres_alloc_str='cpu=2,node=1,mem=4G')]}

    def test_scheduler_grace_is_not_added_to_budget(self):
        x=m.allocation(self.scheduler(),'123',101)
        self.assertEqual(x['deadline'],1900)
        self.assertEqual(x['work_deadline'],1840)

    def test_scheduler_shorter_end_is_honored(self):
        d=self.scheduler();d['jobs'][0]['end_time']['number']=800
        self.assertEqual(m.allocation(d,'123',101)['deadline'],800)

    def test_unset_epoch_is_rejected(self):
        d=self.scheduler();d['jobs'][0]['start_time']['set']=False
        with self.assertRaises(m.Closed):m.allocation(d,'123',101)

    def test_wrong_gpu_allocation_rejected(self):
        d=self.scheduler();d['jobs'][0]['tres_alloc_str']+=',gres/gpu=1'
        with self.assertRaises(m.Closed):m.allocation(d,'123',101)

    def test_expired_reserve_rejected(self):
        with self.assertRaises(m.Closed):m.allocation(self.scheduler(),'123',1841)

    def test_extended_request_rejected(self):
        d=self.scheduler();d['jobs'][0]['time_limit']['number']=31
        with self.assertRaises(m.Closed):m.allocation(d,'123',101)

    def test_redirect_cannot_create_an_unmetered_request(self):
        with self.assertRaises(m.Closed):m.NoRedirect().redirect_request(None,None,302,'',{},'https://elsewhere/')

    def test_no_home_write_path(self):
        with self.assertRaises(m.Closed):m.checked('/gpfs/home/zixianma/map.json')

    def test_public_files_fit_cap_with_headers(self):
        self.assertEqual(sum(x['bytes'] for x in m.files()),1231929664)
        self.assertLess(sum(x['bytes']+65536 for x in m.files()),m.LIMITS['charged_download_bytes'])

    def response(self,body=b'abc'):
        class Response:
            status=200;url='https://zenodo.org/fixture'
            headers={'Content-Length':str(len(body)),'Content-Disposition':'attachment; filename=fixture.tar.gz'}
            def __init__(self):
                import io
                self.data=io.BytesIO(body)
            def read(self,size):return self.data.read(size)
            def __enter__(self):return self
            def __exit__(self,*args):pass
        return Response()

    def test_nonidentity_encoding_rejected(self):
        r=self.response();r.headers=dict(r.headers,**{'Content-Encoding':'gzip'})
        with self.assertRaises(m.Closed):m.headers_ok(r,dict(url=r.url,bytes=3,filename='fixture.tar.gz'))

    def test_complete_checksum_and_reservations_are_preserved(self):
        # Inject an isolated write root; this runs no network or scheduler call.
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            r=self.response();item=dict(url=r.url,bytes=3,filename='fixture.tar.gz',published_md5='900150983cd24fb0d6963f7d28e17f72')
            class Opener:
                def open(self,*a,**k):return r
            with patch.object(m,'checked',side_effect=lambda p:Path(p)):
                x=m.transfer(Path(tmp),item,0,time.time()+10,Opener())
            self.assertEqual(x['bytes'],3)
            self.assertTrue((Path(tmp)/'transfer-0-reservation.json').exists())
            self.assertTrue((Path(tmp)/'transfer-0-verified.json').exists())

    def test_checksum_failure_retains_body_and_full_reservation(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            r=self.response();item=dict(url=r.url,bytes=3,filename='fixture.tar.gz',published_md5='0'*32)
            class Opener:
                def open(self,*a,**k):return r
            with patch.object(m,'checked',side_effect=lambda p:Path(p)):
                with self.assertRaises(m.Closed):m.transfer(Path(tmp),item,0,time.time()+10,Opener())
            self.assertTrue((Path(tmp)/'fixture.tar.gz').exists())
            self.assertTrue((Path(tmp)/'transfer-0-reservation.json').exists())
            self.assertFalse((Path(tmp)/'transfer-0-verified.json').exists())


if __name__=='__main__':unittest.main()
