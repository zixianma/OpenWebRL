import copy
import hashlib
import os
from pathlib import Path
import tempfile
import time
import unittest

from openwebrl.webprm_gitlab_recovery import verify_pristine_base,verify_independent_copy,Closed


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.base=self.root/'base';self.base.mkdir(mode=0o700)
        (self.base/'d').mkdir(mode=0o755);(self.base/'d/file').write_bytes(b'native');(self.base/'d/file').chmod(0o600)
        os.link(self.base/'d/file',self.base/'hard');os.symlink('/d/file',self.base/'link');os.mkfifo(self.base/'fifo',0o600)
        self.manifest=dict(destination=str(self.base),image_executed=False,source='third_party_digest_pinned_oci_candidate',
            entries={'d':dict(kind='dir',mode=0o755),'d/file':dict(kind='file',mode=0o600,bytes=6),
                'hard':dict(kind='hardlink',regular_target='d/file',mode=0o600,bytes=6),
                'link':dict(kind='symlink',linkname='/d/file'),'fifo':dict(kind='fifo',mode=0o600)},
            regular_file_sha256={'d/file':hashlib.sha256(b'native').hexdigest()})

    def verify(self):return verify_pristine_base(self.base,self.manifest,time.time()+10)

    def test_pristine_types_links_modes_and_bytes_verify_without_mutation(self):
        before={str(p.relative_to(self.base)):p.lstat() for p in self.base.rglob('*')}
        result=self.verify();self.assertTrue(result['verified']);self.assertEqual(result['entries'],5)
        after={str(p.relative_to(self.base)):p.lstat() for p in self.base.rglob('*')}
        self.assertEqual({k:(s.st_size,s.st_mtime_ns,s.st_ctime_ns) for k,s in before.items()},
                         {k:(s.st_size,s.st_mtime_ns,s.st_ctime_ns) for k,s in after.items()})

    def test_content_mode_and_extra_file_changes_fail_closed(self):
        (self.base/'d/file').write_bytes(b'tamper')
        with self.assertRaises(Closed):self.verify()
        (self.base/'d/file').write_bytes(b'native');(self.base/'d/file').chmod(0o644)
        with self.assertRaises(Closed):self.verify()
        (self.base/'d/file').chmod(0o600);(self.base/'extra').write_text('unexpected')
        with self.assertRaises(Closed):self.verify()

    def test_executed_or_wrong_destination_base_is_never_reused(self):
        for change in (dict(image_executed=True),dict(destination=str(self.root/'other'))):
            with self.subTest(change=change),self.assertRaises(Closed):
                verify_pristine_base(self.base,dict(self.manifest,**change),time.time()+10)

    def test_absent_old_base_is_diagnosed_before_inventory(self):
        missing=self.root/'absent'
        with self.assertRaisesRegex(Closed,'root is absent'):
            verify_pristine_base(missing,dict(self.manifest,destination=str(missing)),time.time()+10)

    def test_independent_copy_rejects_cross_base_hardlinks(self):
        state=self.root/'state';state.mkdir();(state/'d').mkdir();(state/'d/file').write_bytes(b'native');(state/'hard').write_bytes(b'native');os.symlink('/d/file',state/'link');os.mkfifo(state/'fifo',0o600)
        self.assertEqual(verify_independent_copy(self.base,state,self.manifest,time.time()+10)['shared_regular_inodes'],0)
        other=self.root/'aliased';other.mkdir();(other/'d').mkdir();os.link(self.base/'d/file',other/'d/file');(other/'hard').write_bytes(b'native');os.symlink('/d/file',other/'link');os.mkfifo(other/'fifo',0o600)
        with self.assertRaises(Closed):verify_independent_copy(self.base,other,self.manifest,time.time()+10)

