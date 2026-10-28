import gzip
import hashlib
import io
import os
from pathlib import Path
import tarfile
import tempfile
import time
import unittest

from openwebrl import webprm_cms_materializer as m


class MaterializerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.layers = []

    def layer(self, entries):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w') as archive:
            for entry in entries:
                name,kind,value=entry[:3]
                item = tarfile.TarInfo(name)
                mode=entry[3] if len(entry)==4 else None
                if kind == 'file':
                    item.size = len(value); item.mode = 0o644 if mode is None else mode
                    archive.addfile(item, io.BytesIO(value))
                else:
                    item.type = {'symlink':tarfile.SYMTYPE,'hardlink':tarfile.LNKTYPE,
                                 'dir':tarfile.DIRTYPE,'device':tarfile.CHRTYPE,'fifo':tarfile.FIFOTYPE}[kind]
                    item.linkname = value; item.mode = 0o755 if mode is None else mode
                    archive.addfile(item)
        raw = output.getvalue(); compressed = gzip.compress(raw)
        p = self.root / ('layer' + str(len(self.layers)))
        p.write_bytes(compressed)
        self.layers.append(dict(path=str(p),compressed_bytes=len(compressed),raw_bytes=len(raw),
                                compressed_sha256=hashlib.sha256(compressed).hexdigest(),
                                raw_diff_id='sha256:'+hashlib.sha256(raw).hexdigest()))

    def run_materialize(self, **kwargs):
        return m.materialize(self.layers,self.root/'image',self.root/'evidence',time.time()+10,**kwargs)

    def test_final_overlay_and_internal_links(self):
        self.layer([('etc/value','file',b'old'),('etc/gone','file',b'gone')])
        self.layer([('etc/value','file',b'new'),('etc/.wh.gone','file',b''),
                    ('etc/alias','hardlink','etc/value'),('config','symlink','/etc/value')])
        result = self.run_materialize()
        self.assertEqual((self.root/'image/etc/value').read_bytes(),b'new')
        self.assertFalse((self.root/'image/etc/gone').exists())
        self.assertEqual(os.readlink(self.root/'image/config'),'/etc/value')
        self.assertEqual((self.root/'image/etc/value').stat().st_ino,(self.root/'image/etc/alias').stat().st_ino)
        self.assertEqual(result['final_regular_bytes'],3)

    def test_directory_sticky_and_group_modes_preserved_regular_setid_cleared(self):
        self.layer([('tmp','dir','',0o1777),('data','dir','',0o2775),
                    ('setid','file',b'fixture',0o6755)])
        self.run_materialize()
        self.assertEqual((self.root/'image/tmp').stat().st_mode&0o7777,0o1777)
        self.assertEqual((self.root/'image/data').stat().st_mode&0o7777,0o2775)
        self.assertEqual((self.root/'image/setid').stat().st_mode&0o7777,0o755)

    def test_opaque_keeps_new_same_layer_file(self):
        self.layer([('etc/old','file',b'old')])
        self.layer([('etc/new','file',b'new'),('etc/.wh..wh..opq','file',b'')])
        self.run_materialize()
        self.assertEqual([p.name for p in (self.root/'image/etc').iterdir()],['new'])

    def test_traversal_rejected_before_destination_created(self):
        self.layer([('../escape','file',b'bad')])
        with self.assertRaisesRegex(m.Closed,'escapes'):
            self.run_materialize()
        self.assertFalse((self.root/'image').exists())

    def test_symlink_parent_writer_rejected(self):
        self.layer([('link','symlink','/tmp'),('link/outside','file',b'bad')])
        with self.assertRaisesRegex(m.Closed,'non-directory'):
            self.run_materialize()
        self.assertFalse((self.root/'image').exists())

    def test_link_escape_and_special_files_rejected(self):
        self.layer([('escape','symlink','../../outside')])
        with self.assertRaisesRegex(m.Closed,'escapes'):
            self.run_materialize()
        self.layers = []
        (self.root/'layer0').unlink()
        self.layer([('dev/bad','device','')])
        with self.assertRaisesRegex(m.Closed,'Special'):
            self.run_materialize()

    def test_opt_in_special_policy_creates_fifos_and_records_skipped_devices(self):
        self.layer([('sv/x/supervise/control','fifo','',0o600),('dev/null','device',''),('etc/v','file',b'1')])
        with self.assertRaisesRegex(m.Closed,'Special'):
            self.run_materialize()
        result=self.run_materialize(limits=m.Limits(special_members='fifo_and_skip_devices'))
        import stat
        self.assertTrue(stat.S_ISFIFO((self.root/'image/sv/x/supervise/control').lstat().st_mode))
        self.assertFalse(os.path.lexists(self.root/'image/dev/null'))
        manifest=__import__('json').loads(Path(result['manifest']['path']).read_text())
        self.assertEqual(manifest['skipped_device_members'],['dev/null'])
        self.assertEqual(manifest['created_fifos'],['sv/x/supervise/control'])

    def test_hash_cap_and_deadline_fail_before_materialization(self):
        self.layer([('file','file',b'contents')])
        with self.assertRaisesRegex(m.Closed,'apparent-byte'):
            self.run_materialize(limits=m.Limits(final_bytes=1))
        self.layers[0]['raw_diff_id']='sha256:'+'0'*64
        with self.assertRaisesRegex(m.Closed,'content/identity'):
            self.run_materialize()
        with self.assertRaisesRegex(m.Closed,'deadline'):
            m.overlay(self.layers,time.time()-1)
        self.assertFalse((self.root/'image').exists())

    def test_cyclic_links_and_missing_hardlinks_rejected(self):
        self.layer([('a','symlink','b'),('b','symlink','a')])
        with self.assertRaisesRegex(m.Closed,'cycle'):
            self.run_materialize()

    def test_historical_hardlink_never_silently_follows_new_contents(self):
        self.layer([('a','file',b'old'),('b','hardlink','a')])
        self.layer([('a','file',b'new')])
        with self.assertRaisesRegex(m.Closed,'overwritten target'):
            self.run_materialize()
        self.assertFalse((self.root/'image').exists())

    def test_existing_destination_never_modified(self):
        self.layer([('file','file',b'contents')])
        (self.root/'image').mkdir();(self.root/'image/sentinel').write_bytes(b'keep')
        with self.assertRaisesRegex(m.Closed,'Fresh'):
            self.run_materialize()
        self.assertEqual((self.root/'image/sentinel').read_bytes(),b'keep')
