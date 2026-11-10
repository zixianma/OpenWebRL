"""Prevent duplicate paid batches after a failed handoff or recovered upload."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from label_arm_refresh import run,file_sha
from prepare_arm_forward_transfer import write,sha


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.request=self.root/'batch-001.jsonl';self.request.write_text('{}\n')
        (self.root/'index.jsonl').write_text('')
        self.digest=file_sha(self.request)
        self.manifest=dict(cost_ceiling_usd=10,files={p.name:dict(sha256=file_sha(p)) for p in (self.request,self.root/'index.jsonl')})
        write(self.root/'manifest.json',self.manifest)
        self.approval=self.root/'approval.json'
        write(self.approval,dict(teacher_approved=True,model='gpt-5.5',api_budget_usd=1000,requests_manifest_sha256=file_sha(self.root/'manifest.json')))
        write(self.root/'pilot-passed.json',dict(passed=True))
        self.client=SimpleNamespace(files=SimpleNamespace(create=Mock(return_value=SimpleNamespace(id='file-test',model_dump=lambda **_:dict(id='file-test')))),batches=SimpleNamespace(create=Mock(return_value=SimpleNamespace(model_dump=lambda **_:dict(id='batch-test',status='validating')))))
    def tearDown(self):self.temp.cleanup()
    def submit(self):
        with patch('openai.OpenAI',return_value=self.client),patch('dotenv.dotenv_values',return_value={'OPENAI_API_KEY':'synthetic-test-only'}):
            run('submit',self.root,self.approval)

    def test_unknown_batch_result_blocks_duplicate_upload_and_batch(self):
        self.client.batches.create.side_effect=RuntimeError('Unknown network outcome')
        with self.assertRaises(RuntimeError):self.submit()
        self.assertTrue((self.root/'batch-001.uploaded.json').exists())
        with self.assertRaisesRegex(ValueError,'Unresolved batch creation'):self.submit()
        self.client.files.create.assert_called_once();self.client.batches.create.assert_called_once()

    def test_reconciled_upload_is_reused(self):
        write(self.root/'batch-001.uploaded.json',dict(input_file_id='file-recovered',request_file_sha256=self.digest))
        self.submit();self.submit()
        self.client.files.create.assert_not_called();self.client.batches.create.assert_called_once()
        self.assertEqual(self.client.batches.create.call_args.kwargs['input_file_id'],'file-recovered')

    def test_streaming_hash_avoids_whole_file_read(self):
        with patch.object(Path,'read_bytes',side_effect=AssertionError('Whole file read forbidden')):
            self.assertEqual(file_sha(self.request),hashlib.sha256(b'{}\n').hexdigest())

if __name__=='__main__':unittest.main()
