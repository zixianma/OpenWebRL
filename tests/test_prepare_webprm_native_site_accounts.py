import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('site_accounts',Path(__file__).resolve().parents[1]/'scripts/prepare_webprm_native_site_accounts.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SOURCE=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/tmp',prefix='native-accounts-test-');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_shopping_helper_scope_permissions_and_hash_match_worker_contract(self):
        out=self.root/'shopping';proof=m.prepare(SOURCE,'shopping',out)
        helper=proof['native_helper_accounts'];path=Path(helper['path']);value=json.loads(path.read_text())
        self.assertEqual(set(value),{'shopping_site_admin'});self.assertEqual(set(value['shopping_site_admin']),{'username','password'})
        self.assertEqual(path.stat().st_mode&0o777,0o600);self.assertEqual(out.stat().st_mode&0o777,0o700)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),helper['sha256']);self.assertFalse(proof['live_authentication_verified'])
        self.assertNotIn(value['shopping_site_admin']['password'],(out/'account-binding.json').read_text())

    def test_cms_and_gitlab_bind_only_their_native_accounts(self):
        for domain in ('shopping_admin','gitlab'):
            proof=m.prepare(SOURCE,domain,self.root/domain)
            self.assertIsNone(proof['native_helper_accounts']);self.assertIsNone(proof['storage_state']);self.assertFalse(proof['ready_to_host'])

    def test_unknown_domain_and_changed_source_refused_before_output(self):
        with self.assertRaises(m.GateClosed):m.prepare(SOURCE,'map',self.root/'no-output')
        with patch.object(m,'source_manifest',return_value={'files':{}}):
            with self.assertRaises(m.GateClosed):m.prepare(SOURCE,'shopping_admin',self.root/'no-output')
        self.assertFalse((self.root/'no-output').exists())

    def test_existing_and_symlink_destinations_are_not_reused(self):
        out=self.root/'existing';out.mkdir()
        with self.assertRaises(FileExistsError):m.prepare(SOURCE,'shopping_admin',out)
        link=self.root/'link';link.symlink_to(out)
        with self.assertRaises(m.GateClosed):m.prepare(SOURCE,'shopping_admin',link)


if __name__=='__main__':unittest.main()
