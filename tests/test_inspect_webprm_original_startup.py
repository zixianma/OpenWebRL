import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time
import unittest

spec=importlib.util.spec_from_file_location('startup_scan',Path(__file__).resolve().parents[1]/'scripts/inspect_webprm_original_startup.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def tar(items):
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w') as archive:
        for name,raw in items:
            info=tarfile.TarInfo(name)
            if isinstance(raw,tuple):info.type=tarfile.SYMTYPE;info.linkname=raw[0];archive.addfile(info)
            else:info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    return data.getvalue()


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/tmp',prefix='original-startup-test-');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def archive(self,layers,domain='shopping_admin'):
        blobs=[tar(x) for x in layers]
        config=json.dumps(dict(os='linux',architecture='amd64',rootfs=dict(diff_ids=['sha256:'+sha(x) for x in blobs]),
            config=dict(Entrypoint=['/entrypoint.sh'],Cmd=['supervisord','-n'],Env=['DATABASE_PASSWORD=fixture-secret']))).encode()
        config_name=sha(config)+'.json';names=[str(i)+'/layer.tar' for i in range(len(blobs))]
        manifest=json.dumps([dict(RepoTags=[m.TAGS[domain]],Config=config_name,Layers=names)]).encode()
        archive=tar([('manifest.json',manifest),(config_name,config),*zip(names,blobs)])
        path=self.root/'original.tar';path.write_bytes(archive)
        proof=dict(status='offline_archive_identity_consistent',image_tag=m.TAGS[domain],archive_path=str(path),archive_bytes=len(archive),archive_sha256=sha(archive),
            manifest_sha256=sha(manifest),config_sha256=sha(config),layers=[dict(index=i,path=names[i],bytes=len(raw),sha256=sha(raw),diff_id='sha256:'+sha(raw)) for i,raw in enumerate(blobs)])
        return path,proof

    def scan(self,layers,domain='shopping_admin',**kwargs):
        path,proof=self.archive(layers,domain);return m.inspect(path,proof,domain,time.time()+10,**kwargs)

    def test_original_runtime_and_mutable_roots_are_observed_not_inferred(self):
        result,files=self.scan([[('entrypoint.sh',b'#!/bin/sh\nexec supervisord -n\n'),('var/lib/mysql/base.ibd',b'db'),('var/www/magento2/pub/media/base.jpg',b'media')]])
        self.assertEqual(result['startup_config']['Entrypoint'],['/entrypoint.sh'])
        self.assertEqual(result['mutable_root_candidates']['var/lib/mysql']['layer_regular_bytes'],2)
        self.assertEqual(result['mutable_root_candidates']['var/www/magento2/pub/media']['layer_regular_bytes'],5)
        self.assertFalse(result['executed_image']);self.assertFalse(result['ready_to_host'])
        self.assertEqual(result['environment_names'],['DATABASE_PASSWORD']);self.assertNotIn('fixture-secret',json.dumps(result))

    def test_later_original_layer_replaces_selected_startup_content(self):
        result,files=self.scan([[('entrypoint.sh',b'old')],[('entrypoint.sh',b'new')]])
        self.assertEqual(files['entrypoint.sh']['raw'],b'new');self.assertEqual(files['entrypoint.sh']['layer'],1)

    def test_whiteout_and_opaque_directory_preserve_same_layer_new_entries(self):
        result,files=self.scan([[('etc/supervisor.d/a.ini',b'old'),('entrypoint.sh',b'old')],
          [('etc/supervisor.d/new.ini',b'new'),('etc/supervisor.d/.wh..wh..opq',b''),('.wh.entrypoint.sh',b'')]])
        self.assertEqual(set(files),{'etc/supervisor.d/new.ini'})

    def test_root_opaque_whiteout_removes_prior_selected_files(self):
        result,files=self.scan([[('entrypoint.sh',b'old')],[('.wh..wh..opq',b''),('etc/os-release',b'ID=fixture')]])
        self.assertEqual(set(files),{'etc/os-release'})

    def test_symlink_is_recorded_without_following_or_copying_target(self):
        result,files=self.scan([[('entrypoint.sh',('/gpfs/home/zixianma/do-not-read',))]])
        self.assertEqual(files,{});self.assertIn('entrypoint.sh',result['unresolved_selected_links'])

    def test_wrong_domain_and_changed_layer_identity_are_rejected(self):
        path,proof=self.archive([[('entrypoint.sh',b'original')]])
        with self.assertRaises(m.GateClosed):m.inspect(path,proof,'shopping',time.time()+10)
        proof['layers'][0]['sha256']='0'*64
        with self.assertRaises(m.GateClosed):m.inspect(path,proof,'shopping_admin',time.time()+10)

    def test_large_or_binary_selected_members_stop_before_evidence_copy(self):
        for raw in (b'too-large',b'bad\0binary'):
            with self.subTest(raw=raw):
                path,proof=self.archive([[('entrypoint.sh',raw)]])
                with self.assertRaises(m.GateClosed):m.inspect(path,proof,'shopping_admin',time.time()+10,text_cap=4)

    def test_private_sensitive_startup_evidence_is_content_addressed(self):
        result,files=self.scan([[('var/www/magento2/app/etc/env.php',b'password=fixture-private')]])
        out=self.root/'evidence';m.save(out,result,files,dict(path='fixture',sha256='0'*64))
        path=next((out/'private-startup-files').iterdir())
        self.assertEqual(path.name,sha(b'password=fixture-private')+'.txt');self.assertEqual(path.stat().st_mode&0o777,0o600)
        self.assertNotIn('fixture-private',(out/'startup-inventory.json').read_text())

    def test_gitlab_mutable_repository_and_configuration_roots_remain_distinct(self):
        result,files=self.scan([[('etc/gitlab/gitlab.rb',b'external_url fixture'),('var/opt/gitlab/git-data/repositories/base',b'repo')]],domain='gitlab')
        self.assertEqual(result['mutable_root_candidates']['var/opt/gitlab']['layer_regular_bytes'],4)
        self.assertIn('etc/gitlab/gitlab.rb',files)


if __name__=='__main__':unittest.main()
