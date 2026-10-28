import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from openwebrl import webprm_reddit_provenance as mod

class OriginalProvenance(unittest.TestCase):
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);raw=json.dumps(value).encode();path.write_bytes(raw);return hashlib.sha256(raw).hexdigest()
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='webprm-provenance-',dir='/tmp'));self.runtime=self.root/'runtime';self.scratch=self.root/'scratch';self.runtime.mkdir();self.scratch.mkdir()
        self.layout=self.runtime/'oci';blobs=self.layout/'blobs/sha256';blobs.mkdir(parents=True)
        payloads=[f'original-layer-{i}'.encode() for i in range(29)]
        descriptors=[{'digest':'sha256:'+hashlib.sha256(b).hexdigest(),'size':len(b)} for b in payloads]
        config={'rootfs':{'diff_ids':['sha256:'+hashlib.sha256(b+b'raw').hexdigest() for b in payloads]}}
        self.config=self.write(self.runtime/'original-registry-mirrors/webarenaimages-config.json',config)
        manifest={'config':{'digest':'sha256:'+self.config},'layers':descriptors}
        self.manifest=self.write(self.runtime/'original-registry-mirrors/webarenaimages-manifest.json',manifest)
        (blobs/self.config).write_bytes((self.runtime/'original-registry-mirrors/webarenaimages-config.json').read_bytes())
        for descriptor,payload in zip(descriptors,payloads):(blobs/descriptor['digest'][7:]).write_bytes(payload)
        converted=json.dumps(manifest).encode();oci=hashlib.sha256(converted).hexdigest();(blobs/oci).write_bytes(converted);self.write(self.layout/'index.json',{'manifests':[{'digest':'sha256:'+oci}]})
        rows=[{'index':i,'digest':d['digest'],'size':d['size'],'expected_diff_id':config['rootfs']['diff_ids'][i],'compressed_sha256_verified':True} for i,d in enumerate(descriptors)]
        download=dict(status='complete',source_manifest_digest='sha256:'+self.manifest,config_digest='sha256:'+self.config,layout_dir=str(self.layout),oci_reference='oci:'+str(self.layout),oci_manifest_digest='sha256:'+oci,layers=rows)
        self.download=self.runtime/'download.json';download_sha=self.write(self.download,download)
        self.base=self.scratch/'original-build';self.base.mkdir()
        self.build=self.runtime/'build.json';self.build_data=dict(schema_version=1,kind='original_reddit_sandbox_build',status='completed',returncode=0,source_manifest_sha256=self.manifest,config_sha256=self.config,sandbox_path=str(self.base),sandbox_root_identity=mod.root_identity(self.base),download_receipt_path=str(self.download),download_receipt_sha256=download_sha,argv=['apptainer','build','--disable-cache','--sandbox',str(self.base),'oci:'+str(self.layout)])
        self.build_sha=self.write(self.build,self.build_data)
        self.verification=self.runtime/'verified.json';self.verification_data=dict(status='verified',all_original_layers_verified=True,startup_content_final=True,missing_or_incomplete_layers=[],source_manifest_digest='sha256:'+self.manifest,config_sha256=self.config,layers=[dict(index=i,compressed_digest=d['digest'],compressed_bytes=d['size'],raw_diff_id=config['rootfs']['diff_ids'][i],verified=True) for i,d in enumerate(descriptors)])
        self.verification_sha=self.write(self.verification,self.verification_data)
        self.patcher=patch.multiple(mod,RUNTIME=self.runtime,SCRATCH=self.scratch,ORIGINAL_MANIFEST=self.manifest,ORIGINAL_CONFIG=self.config);self.patcher.start();self.addCleanup(self.patcher.stop)
    def instance(self,name='A'):
        state=self.scratch/name;state.mkdir();receipt=self.runtime/(name+'.json')
        data=dict(schema_version=1,kind='original_reddit_state_copy',status='completed',returncode=0,state_path=str(state),source_sandbox_path=str(self.base),state_root_identity=mod.root_identity(state),build_receipt_path=str(self.build),build_receipt_sha256=self.build_sha,original_verification_path=str(self.verification),original_verification_sha256=self.verification_sha,argv=['cp','-a','--reflink=auto','--',str(self.base),str(state)])
        sha=self.write(receipt,data)
        return dict(scope='original',image_sha256=self.manifest,state_path=str(state),build_receipt_path=str(self.build),build_receipt_sha256=self.build_sha,state_provenance_path=str(receipt),state_provenance_sha256=sha)
    def change_state(self,instance,**changes):
        path=Path(instance['state_provenance_path']);data=json.loads(path.read_text());data.update(changes);instance['state_provenance_sha256']=self.write(path,data)
    def test_matching_build_distinct_fresh_copies(self):
        a=mod.validate_original_provenance(self.instance('A'));b=mod.validate_original_provenance(self.instance('B'))
        self.assertEqual(a['build_receipt_sha256'],b['build_receipt_sha256']);self.assertNotEqual(a['state_provenance_sha256'],b['state_provenance_sha256']);self.assertEqual(a['verified_original_layers'],29)
    def test_partial_verification_rejected_even_with_updated_receipt_hash(self):
        self.verification_data['layers'].pop();self.verification_sha=self.write(self.verification,self.verification_data)
        with self.assertRaises(ValueError):mod.validate_original_provenance(self.instance())
    def test_false_layer_verification_rejected(self):
        self.verification_data['layers'][4]['verified']=False;self.verification_sha=self.write(self.verification,self.verification_data)
        with self.assertRaises(ValueError):mod.validate_original_provenance(self.instance())
    def test_diagnostic_missing_unknown_scope_rejected(self):
        i=self.instance()
        for scope in (None,'diagnostic','unknown'):
            i['scope']=scope
            with self.assertRaises(ValueError):mod.validate_original_provenance(i)
    def test_copy_source_command_mismatch_rejected(self):
        i=self.instance();self.change_state(i,argv=['cp','-a','different-source',i['state_path']])
        with self.assertRaises(ValueError):mod.validate_original_provenance(i)
    def test_failed_build_rejected(self):
        self.build_data['returncode']=1;self.build_sha=self.write(self.build,self.build_data)
        with self.assertRaises(ValueError):mod.validate_original_provenance(self.instance())
    def test_replaced_state_root_rejected(self):
        i=self.instance();p=Path(i['state_path']);p.rename(p.with_name('preserved-A'));p.mkdir()
        with self.assertRaises(ValueError):mod.validate_original_provenance(i)
    def test_receipt_tamper_rejected(self):
        i=self.instance();Path(i['state_provenance_path']).write_text('{}')
        with self.assertRaises(ValueError):mod.validate_original_provenance(i)
    def test_original_build_cannot_be_running_state(self):
        i=self.instance();i['state_path']=str(self.base);self.change_state(i,state_path=str(self.base),state_root_identity=mod.root_identity(self.base),argv=['cp','-a','--reflink=auto','--',str(self.base),str(self.base)])
        with self.assertRaises(ValueError):mod.validate_original_provenance(i)
    def test_wrong_config_pin_rejected(self):
        self.verification_data['config_sha256']='0'*64;self.verification_sha=self.write(self.verification,self.verification_data)
        with self.assertRaises(ValueError):mod.validate_original_provenance(self.instance())
    def test_home_path_and_symlink_rejected(self):
        with self.assertRaises(ValueError):mod.checked_path('/gpfs/home/zixianma/forbidden',self.scratch)
        link=self.scratch/'home-link';link.symlink_to('/gpfs/home/zixianma',target_is_directory=True)
        with self.assertRaises(ValueError):mod.checked_path(link,self.scratch)

if __name__=='__main__':unittest.main(verbosity=2)
