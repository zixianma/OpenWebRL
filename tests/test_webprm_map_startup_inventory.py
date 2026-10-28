import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time
import unittest

from openwebrl import webprm_map_startup_inventory as m


def tar_bytes(entries):
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w') as archive:
        for name,data,kind in entries:
            row=tarfile.TarInfo(name)
            if kind=='file':row.size=len(data);archive.addfile(row,io.BytesIO(data))
            else:
                row.type={'dir':tarfile.DIRTYPE,'symlink':tarfile.SYMTYPE,'fifo':tarfile.FIFOTYPE}[kind]
                if kind=='symlink':row.linkname=data
                archive.addfile(row)
    return out.getvalue()


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='map-inventory-fixture-',dir='/tmp')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def budget(self,**kw):return m.Budget(time.time()+20,**kw)

    def save(self,name,raw):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        return path,m.digest(raw)

    def test_actual_volume_versions_are_captured_without_extraction(self):
        prefix='mnt/projects/site/docker/volumes/'
        raw=tar_bytes([('.',None,'dir'),(prefix+'osm-data/_data/PG_VERSION',b'15\n','file'),
                       (prefix+'osm-tiles/_data/tile.png',b'png','file')])
        path,h=self.save('volume.tar',raw)
        result=m.inventory_dataset(path,h,'osm_tile_server.tar',self.budget())
        self.assertEqual(result['pg_versions']['osm-data/PG_VERSION']['version'],'15')
        self.assertEqual(set(result['role_totals']),{'osm-data','osm-tiles'})
        self.assertFalse(result['ready_to_host']);self.assertFalse(result['extracted'])
        self.assertFalse((self.root/'mnt').exists())

    def test_gzip_selected_text_and_private_evidence(self):
        raw=tar_bytes([('source/config/database.yml',b'database: original\n','file'),('source/db/seed.sql',b'seed','file')])
        path,h=self.save('source.tar.gz',gzip.compress(raw))
        result=m.inventory_dataset(path,h,'frontend_source',self.budget())
        out=self.root/'evidence';m.save_inventory(out,result)
        self.assertEqual(len(list((out/'private-selected').glob('*.txt'))),1)
        self.assertTrue(all((x.stat().st_mode&0o777)==0o600 for x in (out/'private-selected').iterdir()))
        self.assertEqual(result['initializer_candidates'][0]['path'],'source/db/seed.sql')

    def test_saved_selected_evidence_remains_hash_bound_for_version_gates(self):
        path,h=self.save('source.tar',tar_bytes([('db/PG_VERSION',b'15\n','file')]))
        result=m.inventory_dataset(path,h,'frontend_source',self.budget())
        out=self.root/'saved';m.save_inventory(out,result);receipt_hash=m.digest((out/'inventory.json').read_bytes())
        loaded=m.load_inventory(out,receipt_hash)
        self.assertEqual(loaded['captured']['db/PG_VERSION']['raw'],b'15\n')
        member=next((out/'private-selected').iterdir());member.write_bytes(b'14\n')
        with self.assertRaisesRegex(m.Closed,'content changed'):m.load_inventory(out,receipt_hash)

    def test_hash_caps_and_deadline_cannot_be_bypassed(self):
        path,h=self.save('source.tar',tar_bytes([('source/PG_VERSION',b'15\n','file')]))
        for kw in ({'max_selected_bytes':2},{'max_text_file_bytes':2},{'max_archive_bytes':100},{'max_regular_bytes':2}):
            with self.subTest(kw=kw),self.assertRaises(m.Closed):m.inventory_dataset(path,h,'frontend_source',self.budget(**kw))
        with self.assertRaises(m.Closed):m.inventory_dataset(path,'0'*64,'frontend_source',self.budget())
        for deadline in (float('nan'),float('inf'),time.time()-1):
            with self.assertRaises(m.Closed):m.Budget(deadline)

    def test_traversal_duplicate_sparse_or_special_members_fail(self):
        for name,kind in [('../','dir'),('/','dir'),('../PG_VERSION','file'),('x','fifo')]:
            path,h=self.save('unsafe.tar',tar_bytes([(name,b'15','file' if kind=='file' else kind)]))
            with self.subTest(name=name),self.assertRaises(ValueError):m.inventory_dataset(path,h,'frontend_source',self.budget())
        path,h=self.save('duplicate.tar',tar_bytes([('x/PG_VERSION',b'15','file'),('x/PG_VERSION',b'14','file')]))
        with self.assertRaises(m.Closed):m.inventory_dataset(path,h,'frontend_source',self.budget())

    def test_links_are_unresolved_and_protected_paths_rejected(self):
        path,h=self.save('link.tar',tar_bytes([('data/PG_VERSION','/outside/version','symlink')]))
        result=m.inventory_dataset(path,h,'frontend_source',self.budget())
        self.assertEqual(result['pg_versions'],{})
        self.assertEqual(result['unresolved_selected_links']['data/PG_VERSION']['target'],'/outside/version')
        with self.assertRaises(m.Closed):m.checked('/gpfs/home/zixianma/forbidden')
        alias=self.root/'alias';alias.symlink_to(path)
        with self.assertRaises(m.Closed):m.inventory_dataset(alias,h,'frontend_source',self.budget())

    def oci(self,layers,*,diff_override=None):
        root=self.root/('oci-'+str(len(list(self.root.glob('oci-*')))));blobs=root/'blobs/sha256';blobs.mkdir(parents=True)
        config={'os':'linux','architecture':'amd64','rootfs':{'type':'layers','diff_ids':['sha256:'+m.digest(x) for x in layers]},
                'config':{'Entrypoint':['/custom/start.sh']}}
        if diff_override:config['rootfs']['diff_ids'][0]=diff_override
        c=json.dumps(config).encode();(blobs/m.digest(c)).write_bytes(c)
        descriptors=[]
        for layer in layers:
            z=gzip.compress(layer);(blobs/m.digest(z)).write_bytes(z)
            descriptors.append({'mediaType':'application/vnd.docker.image.rootfs.diff.tar.gzip','size':len(z),'digest':'sha256:'+m.digest(z)})
        manifest={'schemaVersion':2,'config':{'size':len(c),'digest':'sha256:'+m.digest(c)},'layers':descriptors}
        raw=json.dumps(manifest).encode();(blobs/m.digest(raw)).write_bytes(raw)
        return root,m.digest(raw)

    def test_oci_whiteout_and_actual_entrypoint_use_final_overlay(self):
        one=tar_bytes([('data/PG_VERSION',b'14','file'),('custom/start.sh',b'old','file'),('etc/postgresql/a.conf',b'old','file')])
        two=tar_bytes([('data/.wh.PG_VERSION',b'','file'),('etc/postgresql/.wh..wh..opq',b'','file'),
                       ('custom/start.sh',b'new','file'),('etc/postgresql/b.conf',b'new','file')])
        root,h=self.oci([one,two]);result=m.inventory_oci(root,h,self.budget())
        self.assertEqual(result['pg_versions'],{})
        self.assertNotIn('etc/postgresql/a.conf',result['captured'])
        self.assertEqual(result['captured']['custom/start.sh']['raw'],b'new')
        self.assertEqual(len(result['ordered_layers']),2)

    def test_replaced_unselected_parent_hides_old_selected_descendants(self):
        one=tar_bytes([('data/PG_VERSION',b'15','file')]);two=tar_bytes([('data','elsewhere','symlink')])
        root,h=self.oci([one,two]);result=m.inventory_oci(root,h,self.budget())
        self.assertEqual(result['pg_versions'],{})

    def test_oci_raw_diff_identity_and_aggregate_expansion_are_enforced(self):
        raw=tar_bytes([('data/PG_VERSION',b'15','file')]);root,h=self.oci([raw],diff_override='sha256:'+'0'*64)
        with self.assertRaisesRegex(m.Closed,'digest'):m.inventory_oci(root,h,self.budget())
        root,h=self.oci([raw,raw])
        with self.assertRaisesRegex(m.Closed,'expansion'):m.inventory_oci(root,h,self.budget(max_regular_bytes=2*len(raw)-1))

    def test_docker_save_checks_config_and_ordered_raw_layers(self):
        layer=tar_bytes([('var/lib/postgresql/data/PG_VERSION',b'14','file')])
        config=json.dumps({'os':'linux','architecture':'amd64','rootfs':{'type':'layers','diff_ids':['sha256:'+m.digest(layer)]},
                           'config':{'Entrypoint':['/unseen/start']}}).encode()
        name=m.digest(config)+'.json';manifest=json.dumps([{'Config':name,'Layers':['layer/layer.tar'],'RepoTags':['fixture:only']}]).encode()
        outer=tar_bytes([('manifest.json',manifest,'file'),(name,config,'file'),('layer/layer.tar',layer,'file')])
        path,h=self.save('image.tar.gz',gzip.compress(outer));result=m.inventory_docker_save(path,h,self.budget())
        self.assertEqual(result['pg_versions']['var/lib/postgresql/data/PG_VERSION']['version'],'14')
        self.assertEqual(result['uninspected_absolute_entrypoints'],['unseen/start'])
        self.assertFalse(result['ready_to_host'])


if __name__=='__main__':unittest.main()
