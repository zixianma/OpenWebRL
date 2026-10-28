"""Offline CMS reset-audit contracts for Table 5 cells; fixtures only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts import audit_webprm_table5_cms_cell as audit
from openwebrl import webprm_cms_site as cms
from openwebrl.webprm_ephemeral_state import PROTOCOL
from openwebrl.webprm_webarena import digest
from openwebrl.webprm_webarena_sites import INSTANCE_SCHEMA, SITE_RUNTIME_CONFIGURATION


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(value))
    return Path(path)


class CMSResetAuditTests(unittest.TestCase):
    SITE, DOMAIN = cms, 'shopping_admin'

    def setUp(self):
        cms = self.SITE
        self.tmp = TemporaryDirectory(prefix='webprm-cms-audit-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup); root = Path(self.tmp.name)
        cell = dict(cell_id='webarena_lite_000__gpt-4o-2024-05-13__no_search', domain=self.DOMAIN, task_id=0,
                    actor='gpt-4o-2024-05-13', selector='no_search')
        scratch = root/'scratch'; key = hashlib.sha256(cell['cell_id'].encode()).hexdigest()
        state = scratch/'states'/key; base = scratch/'baseline'; folder = root/'campaign/attempt-1/cells'/key
        build = save(root/'campaign/attempt-1/original-build/receipt.json', dict(kind=cms.BUILD_KIND, status='completed',
            baseline_path=str(base), materialized_content_sha256=cms.BASE_CONTENT_SHA256,
            hosting_proposal=cms.HOSTING_PROPOSAL, hosting_validation=cms.HOSTING_VALIDATION))
        copy = save(folder/'copy/receipt.json', dict(kind=cms.COPY_KIND, status='completed', returncode=0,
            state_path=str(state), source_path=str(base), build_receipt=ref(build),
            argv=['unshare', '--user', '--map-root-user', 'cp', '-a', '--reflink=auto', '--', str(base), str(state)]))
        identity = cms.base_identity()
        provenance = dict(domain=self.DOMAIN, base_identity_sha256=digest(identity), instance_id='1-x',
            state_path=str(state), reset_validation=cms.HOSTING_VALIDATION, build_receipt=ref(build), copy_receipt=ref(copy))
        instance = dict(schema=INSTANCE_SCHEMA, domain=self.DOMAIN, scope='original', endpoint=cms.ENDPOINT,
            state_path=str(state), base_identity=identity, base_identity_sha256=digest(identity),
            state_provenance=provenance, state_provenance_sha256=digest(provenance), service_pid=4242,
            service_validation=cms.HOSTING_VALIDATION, reset_validation=cms.HOSTING_VALIDATION, **cms.instance_extras())
        instance_path = save(folder/'site/instance.json', instance)
        lease = save(folder/'lease.json', dict(task_id='webarena_lite_000', actor_model=cell['actor'], selector='no_search',
            instance_receipt=str(instance_path), instance_receipt_sha256=ref(instance_path)['sha256']))
        claim = save(folder/'state-claim.json', dict(protocol=PROTOCOL, cell_id=cell['cell_id'], destination_absent_at_claim=True,
            preserve_historical_states=True, state_path=str(state), scratch_root=str(scratch), base_path=str(base),
            campaign_root=str(root/'campaign')))
        closure = save(folder/'site/closure.json', dict(owned_pids=[4242], remaining_owned_processes={}))
        self.episode = dict(task_id='webarena_lite_000', actor_model=cell['actor'], selector='no_search',
            reset_id=instance['state_provenance_sha256'], lease_sha256=ref(lease)['sha256'],
            configured_task_sha256='d'*64, site_runtime=SITE_RUNTIME_CONFIGURATION)
        recorded = save(folder/'episode/provenance.json', dict(original_instance=dict(instance, episode_claim=self.episode)))
        self.args = (dict(cell=cell, campaign_root=str(root/'campaign')),
            dict(instance=instance_path, lease=lease, copy_receipt=copy, state_claim=claim, site_closure=closure,
                 provenance=recorded),
            dict(endpoints={self.DOMAIN: cms.ENDPOINT}, site_runtime=SITE_RUNTIME_CONFIGURATION),
            dict(reset_id=instance['state_provenance_sha256']))
        self.instance, self.recorded = instance, recorded

    def test_complete_cms_chain_returns_the_worker_recorded_instance(self):
        value = audit.audit_site_reset(*self.args)
        self.assertEqual(value, json.loads(self.recorded.read_text())['original_instance'])

    def test_reset_identity_claim_and_episode_drift_are_rejected(self):
        inputs, paths, plan, result = self.args
        with self.assertRaisesRegex(ValueError, 'reset identity'):
            audit.audit_cms_reset(inputs, paths, plan, dict(result, reset_id='0'*64))
        with self.assertRaisesRegex(ValueError, 'reset identity'):
            audit.audit_cms_reset(inputs, paths, dict(plan, endpoints={self.DOMAIN: 'http://127.0.0.1:1/admin'}), result)
        save(paths['provenance'], dict(original_instance=dict(self.instance, episode_claim=dict(self.episode, selector='random'))))
        with self.assertRaisesRegex(ValueError, 'episode claim'):audit.audit_cms_reset(inputs, paths, plan, result)
        save(paths['site_closure'], dict(owned_pids=[1]))
        with self.assertRaisesRegex(ValueError, 'Service closure'):audit.audit_cms_reset(inputs, paths, plan, result)

    def test_reddit_cells_still_use_the_reddit_auditor(self):
        inputs, paths, plan, result = self.args
        reddit = deepcopy(inputs); reddit['cell']['domain'] = 'reddit'
        with self.assertRaises(Exception) as caught:audit.audit_site_reset(reddit, paths, plan, result)
        self.assertNotIn(self.SITE.LABEL, str(caught.exception))


    def test_wrapper_runs_the_unchanged_reviewed_auditor_with_only_reset_dispatch(self):
        import hashlib as _h
        from unittest.mock import patch
        reviewed = Path(audit.base.__file__)
        self.assertEqual(_h.sha256(reviewed.read_bytes()).hexdigest(),
            '778445cda9e28b714daaf2efca3fb70358b61a7c15904071301366cce9ad48fc')
        original = audit.base.audit_reddit_reset
        try:
            with patch.object(audit.base, 'main', side_effect=lambda: audit.base.audit_reddit_reset) as main:
                self.assertIs(audit.main(), audit.audit_site_reset)
            main.assert_called_once()
        finally:
            audit.base.audit_reddit_reset = original
        self.assertIs(audit.REDDIT_RESET, original)

class ShoppingResetAuditTests(CMSResetAuditTests):
    """The same chain through the Shopping site module, plus its helper-account receipt."""
    from openwebrl import webprm_shopping_site as SITE
    DOMAIN = 'shopping'

    def test_shopping_instance_without_reviewed_helper_receipt_is_rejected(self):
        inputs, paths, plan, result = self.args
        save(paths['instance'], {k: v for k, v in self.instance.items() if k != 'native_helper_accounts'})
        lease = json.loads(Path(paths['lease']).read_text())
        save(paths['lease'], dict(lease, instance_receipt_sha256=ref(paths['instance'])['sha256']))
        with self.assertRaisesRegex(ValueError, 'reset identity'):audit.audit_cms_reset(inputs, paths, plan, result)


if __name__ == '__main__':
    unittest.main()
