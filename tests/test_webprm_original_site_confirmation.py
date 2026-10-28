"""Exercise each frozen prospective site handler without network or Slurm."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE=Path(__file__).resolve().parents[1]/'scripts/acquire_webprm_original_sites.py'
spec=importlib.util.spec_from_file_location('confirmation_sites',SOURCE)
sites=importlib.util.module_from_spec(spec);spec.loader.exec_module(sites)

PROGRAM=r'''
import importlib.util,io,json,pathlib,socket,sys,time,urllib.parse
from unittest.mock import patch
root=pathlib.Path(sys.argv[1]);scenario=sys.argv[2]
spec=importlib.util.spec_from_file_location('worker',root/'acquisition.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
profile=owner.read(root/'proposal.json')['profile']
with patch.object(socket.socket,'connect',side_effect=AssertionError('No live network')):
 lib,_=owner.load_library(root,profile);transport=lib.confirmation_transport
 assert transport.ARCHIVE_BYTES==profile['archive_bytes']
 assert transport.cms.ARCHIVE_BYTES==profile['archive_bytes']
 assert transport.LIMITS==lib.LIMITS==transport.cms.LIMITS
 assert transport.cms.__file__.startswith(str(root/'dependencies'))
 attempt=root/scenario;attempt.mkdir()
 ledger=lib.TransferLedger(attempt,cap=profile['charged_download_bytes'],metadata_cap=16*2**20)
 def page(identity):
  fields=dict(id=identity,export='download',confirm='t',uuid='01234567-89ab-cdef-0123-456789abcdef')
  return ('<form id="download-form" action="https://drive.usercontent.google.com/download" method="get">'+''.join('<input type="hidden" name="%s" value="%s">'%x for x in fields.items())+'</form>').encode()
 class Response:
  def __init__(self,status,body,**headers):
   self.status=status;self.headers={'Content-Length':str(len(body)),**headers};self.body=io.BytesIO(body);self.reads=0
  def read(self,n):self.reads+=1;return self.body.read(n)
  def close(self):pass
 def archive(body,start,end,size,status=206,name=None):
  h={'Content-Disposition':'attachment; filename='+(name or profile['filename']),'ETag':'"fixed"','Content-Length':str(end-start+1)}
  if status==206:h['Content-Range']=f'bytes {start}-{end}/{size}'
  return Response(status,body,**h)
 raw=page(profile['drive_id']);size=6;chunk=3;failure=None
 if scenario=='ranges':responses=[Response(200,raw),archive(b'abc',0,2,6),archive(b'def',3,5,6)]
 elif scenario=='full':responses=[Response(200,raw),archive(b'abcdef',0,5,6,200)]
 elif scenario=='foreign_id':
  responses=[Response(200,page('1kFocBsE7zPaqeWmG7rNQzTWrVXlrGOno'))];failure='confirmation'
 elif scenario=='quota':responses=[Response(200,raw),Response(429,b'Download quota exceeded')];failure='no retry'
 elif scenario=='repeated_confirmation':responses=[Response(200,raw),Response(200,raw)];failure='no retry'
 elif scenario=='wrong_filename':responses=[archive(b'abcdef',0,5,6,200,name='shopping_admin_final_0719.tar')];failure='filename'
 elif scenario=='full_real_size_reservation':
  size=profile['archive_bytes'];chunk=transport.CHUNK
  responses=[archive(b'',0,size-1,size,200,name='wrong.tar')];failure='filename'
 elif scenario=='full_over_cap_before_read':
  size=profile['archive_bytes'];chunk=transport.CHUNK;ledger.cap=transport.CHUNK+65536
  responses=[archive(b'',0,size-1,size,200)];failure='reservation'
 elif scenario=='range_entity_changed':
  second=archive(b'def',3,5,6);second.headers['ETag']='"changed"'
  responses=[archive(b'abc',0,2,6),second];failure='entity'
 else:raise AssertionError(scenario)
 pending=list(responses);requests=[]
 def fake(request,timeout):requests.append(request);return pending.pop(0)
 try:
  result=lib.download(attempt,ledger,time.time()+15,open_response=fake,size=size,chunk=chunk)
 except transport.GateClosed as error:
  assert failure and failure.lower() in str(error).lower(),str(error)
 else:
  assert failure is None
  assert result['official_drive_id']==profile['drive_id']
  assert (attempt/profile['filename']).read_bytes()==b'abcdef'
  assert ledger.data['metadata_received_bytes']==len(raw)
  assert ledger.data['charged_bytes']==len(requests)*65536+len(raw)+6
  assert urllib.parse.parse_qs(urllib.parse.urlsplit(requests[1].full_url).query)['id']==[profile['drive_id']]
 assert not pending
 assert ledger.data['prior_charged_bytes']==0 and ledger.data['prior_metadata_bytes']==0
 if scenario in ('wrong_filename','full_real_size_reservation','full_over_cap_before_read'):
  assert responses[-1].reads==0
 if scenario=='full_real_size_reservation':
  assert ledger.data['reservations'][0]['reserved_bytes']==size
  assert ledger.data['charged_bytes']==size+65536
 if scenario=='quota':assert ledger.data['metadata_received_bytes']==len(raw)+len(b'Download quota exceeded')
 print(json.dumps({'site':profile['filename'],'scenario':scenario,'network_calls':0}))
'''


class OriginalSiteConfirmationTests(unittest.TestCase):
    def test_frozen_site_identity_accounting_and_failure_boundaries(self):
        with tempfile.TemporaryDirectory(prefix='site-confirmation-',dir=sites.RUNTIME/'next-stage-preparation') as name:
            for site in sites.PROFILES:
                root=Path(name)/site;proposal=sites.prepare(root,site)
                self.assertEqual(proposal['confirmation']['fixed_drive_id'],sites.PROFILES[site]['drive_id'])
                for scenario in ('ranges','full','foreign_id','quota','repeated_confirmation','wrong_filename',
                                 'full_real_size_reservation','full_over_cap_before_read','range_entity_changed'):
                    with self.subTest(site=site,scenario=scenario):
                        result=subprocess.run([sys.executable,'-I','-B','-c',PROGRAM,str(root),scenario],
                                              capture_output=True,text=True,timeout=20)
                        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
