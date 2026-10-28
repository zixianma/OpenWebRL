import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import validate_webprm_cms_hosting as owner


class CMSOwnerTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_run_rejects_before_materialization_or_process(self):
        p=self.root/'proposal.json';p.write_text('{}')
        a=self.root/'approval.json';a.write_text('{}')
        with patch.object(owner.materializer,'materialize') as materialize,patch.object(owner.subprocess,'check_output') as command:
            with self.assertRaises(owner.h.Closed):owner.run(p,a)
            materialize.assert_not_called();command.assert_not_called()

    def test_bootstrap_shell_syntax_and_no_initializer(self):
        result=subprocess.run(['/bin/sh','-n'],input=owner.BOOTSTRAP.encode(),capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for token in ('rm ','mysql_install_db','initdb','docker-entrypoint','curl ','wget '):
            self.assertNotIn(token,owner.BOOTSTRAP)
        self.assertIn('test -f /run/mysqld/.init',owner.BOOTSTRAP)
        self.assertIn('exec supervisord',owner.BOOTSTRAP)

    def test_private_database_options_escape_without_argv_password(self):
        raw=owner.private_db_options(dict(username='user',password='p"\\q',host='127.0.0.1',dbname='magentodb'))
        self.assertIn(b'password="p\\"\\\\q"',raw)
        for bad in ('line\nbreak','nul\0byte'):
            with self.assertRaises(owner.h.Closed):
                owner.private_db_options(dict(username='user',password=bad,host='127.0.0.1',dbname='magentodb'))

    def test_runtime_configuration_changes_preserve_original_nginx(self):
        state=self.root/'state';web=state/'etc/nginx/conf.d/default.conf';web.parent.mkdir(parents=True);web.write_bytes(b'original')
        source=self.root/'replacement';source.write_bytes(b'replacement')
        plan=dict(container_file_install_map={'/etc/nginx/conf.d/default.conf':dict(path=str(source),sha256=hashlib.sha256(b'replacement').hexdigest())},runtime_adaptations=[])
        previous=os.umask(0o077)
        try:owner.prepare_controls(state,self.root/'controls',plan)
        finally:os.umask(previous)
        self.assertEqual(web.read_bytes(),b'replacement')
        self.assertEqual(web.with_name('default.conf.preserved-webprm').read_bytes(),b'original')
        self.assertEqual((self.root/'controls/bootstrap.sh').stat().st_mode&0o777,0o644)
        # 357128: --writable cannot create bind destinations; both exist afterwards.
        for target in owner.MOUNT_POINTS:self.assertTrue((state/target.lstrip('/')).is_dir())
        made=[c['path'] for c in owner.h.read(self.root/'controls/runtime-adaptations.json')['changes'] if c.get('created_empty_mount_point')]
        self.assertEqual(sorted(made),['opt','opt/native-fakeroot','opt/webprm-cms'])

    def test_mount_point_symlink_is_rejected(self):
        state=self.root/'state2';(state/'etc/nginx/conf.d').mkdir(parents=True)
        (state/'etc/nginx/conf.d/default.conf').write_bytes(b'original')
        (state/'opt').symlink_to(self.root)
        source=self.root/'replacement2';source.write_bytes(b'replacement')
        plan=dict(container_file_install_map={'/etc/nginx/conf.d/default.conf':dict(path=str(source),sha256=hashlib.sha256(b'replacement').hexdigest())},runtime_adaptations=[])
        with self.assertRaisesRegex(owner.h.Closed,'plain directory'):owner.prepare_controls(state,self.root/'controls2',plan)

    def test_child_may_write_large_files_while_log_stays_capped(self):
        # 356545: an inherited 32 MiB RLIMIT_FSIZE killed cp -a on the 255 MB Magento log.
        big=self.root/'big.bin';log=self.root/'log'
        script=("import sys;open(sys.argv[1],'wb').write(b'x'*(40*2**20));"
                "sys.stdout.buffer.write(b'y'*(%d+4096))"%owner.LOG_BYTES)
        owner.command([owner.sys.executable,'-c',script,str(big)],log,time.time()+60)
        self.assertEqual(big.stat().st_size,40*2**20);self.assertEqual(log.stat().st_size,owner.LOG_BYTES)
        src=self.root/'src';src.mkdir();big.rename(src/'big.bin')
        owner.command(['cp','-a','--reflink=auto','--',str(src),str(self.root/'dst')],self.root/'copy.log',time.time()+60)
        self.assertEqual((self.root/'dst'/'big.bin').stat().st_size,40*2**20)

    def test_namespace_copy_reads_mode_zero_image_files_and_keeps_modes(self):
        # 356561: plain host cp -a could not read the image's mode-0 run/crond.reboot.
        base=self.root/'base';(base/'run').mkdir(parents=True);locked=base/'run'/'crond.reboot'
        locked.touch();locked.chmod(0);setid=base/'tool';setid.write_bytes(b'x');setid.chmod(0o4755)
        with self.assertRaisesRegex(owner.h.Closed,'Owned command failed'):
            owner.command(['cp','-a','--',str(base),str(self.root/'plain')],self.root/'plain.log',time.time()+60)
        owner.command(['unshare','--user','--map-root-user','cp','-a','--reflink=auto','--',str(base),str(self.root/'state')],
            self.root/'copy.log',time.time()+60)
        self.assertEqual((self.root/'state'/'run'/'crond.reboot').stat().st_mode&0o7777,0)
        self.assertEqual((self.root/'state'/'tool').stat().st_mode&0o7777,0o4755)
        self.assertEqual(Path(owner.__file__).read_text().count("['unshare','--user','--map-root-user','cp','-a'"),1)

    def test_failed_and_timed_out_commands_are_closed(self):
        with self.assertRaisesRegex(owner.h.Closed,'Owned command failed'):
            owner.command([owner.sys.executable,'-c','import sys;sys.exit(3)'],self.root/'fail.log',time.time()+30)
        with self.assertRaises(subprocess.TimeoutExpired):
            owner.command([owner.sys.executable,'-c','import time;time.sleep(30)'],self.root/'slow.log',time.time()+30,timeout=1)
        self.assertEqual(owner.command(['cat'],self.root/'echo.log',time.time()+30,input_bytes=b'hi'),b'hi')

    def test_forbidden_command_rejected_before_subprocess_or_log(self):
        with patch.object(owner.subprocess,'Popen') as run:
            with self.assertRaisesRegex(owner.h.Closed,'Forbidden'):
                owner.command(['rm','fixture-only-not-executed'],self.root/'output',time.time()+10)
            run.assert_not_called()
            self.assertFalse((self.root/'output').exists())

    def test_image_env_is_hash_bound_and_applied_inside_container(self):
        blobs=self.root/'blobs';blobs.mkdir()
        config=json.dumps(dict(config=dict(Env=['PATH=/usr/share/java/elasticsearch/bin:/usr/bin','ES_HOME=/es']))).encode()
        digest=hashlib.sha256(config).hexdigest();(blobs/digest).write_bytes(config)
        manifest=self.root/'manifest.json';manifest.write_text(json.dumps(dict(config=dict(digest='sha256:'+digest))))
        proposal=dict(manifest=owner.ref(manifest),blob_root=str(blobs))
        self.assertEqual(owner.image_env(proposal),['PATH=/usr/share/java/elasticsearch/bin:/usr/bin','ES_HOME=/es'])
        addon=self.root/'addon';addon.mkdir();(addon/'f').write_bytes(b'fixture')
        prefix=owner.container_prefix(dict(state='/s',controls='/c'),
            dict(proposal,fakeroot_addon=str(addon),fakeroot_source_sha256={'f':owner.h.sha(addon/'f')}))
        i=prefix.index('env')
        self.assertEqual(prefix[i+1:i+4],['PATH=/usr/share/java/elasticsearch/bin:/usr/bin','ES_HOME=/es','FAKEROOTDONTTRYCHOWN=1'])
        (blobs/digest).write_bytes(config+b' ')
        with self.assertRaisesRegex(owner.h.Closed,'config blob changed'):owner.image_env(proposal)
