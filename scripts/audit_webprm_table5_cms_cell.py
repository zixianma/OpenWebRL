#!/usr/bin/env python3
"""Table 5 cell auditor for third-party Magento (CMS shopping_admin, Shopping) campaigns.

Runs the unchanged reviewed auditor (scripts/audit_webprm_table5_cell.py, whose
source hash the dispatch erratum pins) with exactly one substitution: the reset
auditor dispatches shopping_admin and shopping cells to the third-party Magento
chain below (with that domain's site module) and every other cell to the original
Reddit reset auditor.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_webprm_table5_cell as base
from scripts.audit_webprm_table5_cell import read, reference, sha
from openwebrl.webprm_webarena import digest

REDDIT_RESET = base.audit_reddit_reset


def site_module(domain):
    from openwebrl import webprm_cms_site, webprm_shopping_site
    return {'shopping_admin': webprm_cms_site, 'shopping': webprm_shopping_site}.get(domain)


def audit_cms_reset(inputs, paths, plan, result):
    """Third-party Magento: validated hosting recipe, per-cell build/copy chain and claim."""
    from openwebrl.webprm_webarena_sites import INSTANCE_SCHEMA
    from openwebrl.webprm_ephemeral_state import PROTOCOL as CLAIM_PROTOCOL
    cms = site_module(inputs['cell']['domain'])
    if cms is None:
        raise ValueError('This domain has no completed offline original-reset auditor')
    instance, lease, state, claim = (read(paths[k]) for k in ('instance','lease','copy_receipt','state_claim'))
    ident = f"webarena_lite_{inputs['cell']['task_id']:03d}"
    provenance = instance.get('state_provenance', {})
    copy_ref = dict(path=str(paths['copy_receipt']), sha256=sha(paths['copy_receipt']))
    if (any(lease.get(k) != v for k,v in dict(task_id=ident, actor_model=inputs['cell']['actor'],
            selector=inputs['cell']['selector']).items()) or
            reference(dict(path=lease['instance_receipt'],sha256=lease['instance_receipt_sha256'])) != paths['instance'] or
            instance.get('schema') != INSTANCE_SCHEMA or instance.get('domain') != cms.DOMAIN or
            instance.get('scope') != 'original' or instance.get('base_identity') != cms.base_identity() or
            instance.get('base_identity_sha256') != digest(instance['base_identity']) or
            instance.get('endpoint','').rstrip('/') != cms.ENDPOINT or
            plan['endpoints'].get(cms.DOMAIN,'').rstrip('/') != cms.ENDPOINT or
            instance.get('service_validation') != cms.HOSTING_VALIDATION or
            instance.get('reset_validation') != cms.HOSTING_VALIDATION or
            instance.get('state_provenance_sha256') != digest(provenance) or
            provenance.get('copy_receipt') != copy_ref or provenance.get('base_identity_sha256') != instance['base_identity_sha256'] or
            result.get('reset_id') != instance['state_provenance_sha256'] or
            any(instance.get(k) != v for k, v in cms.instance_extras().items())):
        raise ValueError(cms.LABEL + ' instance/lease/reset identity differs')
    expected_state = Path(claim['scratch_root'])/'states'/hashlib.sha256(inputs['cell']['cell_id'].encode()).hexdigest()
    if (claim.get('protocol') != CLAIM_PROTOCOL or claim.get('cell_id') != inputs['cell']['cell_id'] or
            claim.get('destination_absent_at_claim') is not True or claim.get('preserve_historical_states') is not True or
            claim.get('state_path') != str(expected_state) or instance.get('state_path') != str(expected_state) or
            provenance.get('state_path') != str(expected_state) or claim.get('campaign_root') != inputs['campaign_root']):
        raise ValueError(cms.LABEL + ' reset lacks its unique prospective state claim')
    build = read(reference(state['build_receipt']))
    site = read(paths['site_closure'])
    if type(instance.get('service_pid')) is not int or instance['service_pid'] not in site.get('owned_pids',[]):
        raise ValueError('Service closure belongs to another ' + cms.LABEL + ' instance')
    if (state.get('kind') != cms.COPY_KIND or state.get('status') != 'completed' or state.get('returncode') != 0 or
            state.get('state_path') != str(expected_state) or state.get('source_path') != build.get('baseline_path') or
            claim.get('base_path') != build.get('baseline_path') or state['state_path'] == build.get('baseline_path') or
            state.get('build_receipt') != provenance.get('build_receipt') or
            state.get('argv') != ['unshare','--user','--map-root-user','cp','-a','--reflink=auto','--',
                                  build['baseline_path'],str(expected_state)] or
            build.get('kind') != cms.BUILD_KIND or build.get('status') != 'completed' or
            build.get('materialized_content_sha256') != cms.BASE_CONTENT_SHA256 or
            build.get('hosting_proposal') != cms.HOSTING_PROPOSAL or build.get('hosting_validation') != cms.HOSTING_VALIDATION):
        raise ValueError(cms.LABEL + ' base/build/copy verification chain differs')
    cms.hosting()
    # The worker records the instance with its episode claim (generic site protocol).
    recorded = read(paths['provenance']).get('original_instance', {})
    episode = recorded.get('episode_claim', {})
    if (any(episode.get(k) != v for k,v in dict(task_id=ident, actor_model=inputs['cell']['actor'],
            selector=inputs['cell']['selector'], reset_id=instance['state_provenance_sha256'],
            lease_sha256=sha(paths['lease']), site_runtime=plan.get('site_runtime')).items()) or
            not re.fullmatch('[0-9a-f]{64}', str(episode.get('configured_task_sha256',''))) or
            set(episode) != {'task_id','actor_model','selector','reset_id','lease_sha256','configured_task_sha256','site_runtime'}):
        raise ValueError(cms.LABEL + ' episode claim differs from its lease/reset')
    return dict(instance, episode_claim=episode)


def audit_site_reset(inputs, paths, plan, result):
    if site_module(inputs['cell']['domain']) is not None:
        return audit_cms_reset(inputs, paths, plan, result)
    return REDDIT_RESET(inputs, paths, plan, result)


def main():
    base.audit_reddit_reset = audit_site_reset
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
