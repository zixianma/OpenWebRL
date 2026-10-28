"""Live shopping_admin (CMS) site adapter for Table 5 cells.

The hosting/reset recipe was validated by CPU job 357831 (third-party, digest-pinned
OCI image; equivalence to the original WebArena CMS state is NOT verified). This
module binds every live instance to that validation, checks the per-cell build/copy
provenance, and performs the released shopping_admin UI login. Nothing runs at import.

hosting/base_identity/validate_instance take an optional `site` module with the same
constants (webprm_shopping_site passes itself); the default is this CMS module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

from .webprm_webarena import EvaluationNotReady, digest

N = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation')
ORIGIN = 'http://127.0.0.1:7780'
ENDPOINT = ORIGIN + '/admin'
DOMAIN = 'shopping_admin'
LABEL = 'CMS'
NAME = 'cms'
HOSTING_PROPOSAL = dict(path=str(N/'cms-hosting-validation-v9/proposal.json'),
    sha256='d093cdc4cd9d7c37e2dd9c9c671992b590414c0f7ccc42e34598739ed3e6e6be')
HOSTING_VALIDATION = dict(path=str(N/'cms-hosting-validation-v3/campaign/attempt-357831/validation-result.json'),
    sha256='3899a3d5e1fc216cb5a2a0514dd108c46c5e70db30fe2727d405dc36d7ad145b')
# sha256 of the canonical materialization manifest without its 'destination' key;
# identical for attempts 357728 and 357831 (deterministic content identity).
BASE_CONTENT_SHA256 = '5b028c6d63ee964245cfdcbafb6d793e0d409b012ac02001145ed5ba24f12820'
BUILD_KIND = 'cms_thirdparty_baseline_materialization'
COPY_KIND = 'cms_thirdparty_state_copy'
# Per-cell build stage: the CMS image fits the materializer defaults.
MATERIALIZE_SECONDS = 1800


def materializer_limits():
    from .webprm_cms_materializer import Limits
    return Limits()


def hosting_profile():
    from .webprm_cms_hosting import CMS
    return CMS


def _site(site):
    return sys.modules[__name__] if site is None else site


def _ref(value, maximum=64 * 2**20):
    from .webprm_webarena_sites import canonical_runtime_path
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise EvaluationNotReady('Hash-bound CMS reference required')
    path = canonical_runtime_path(value['path'])
    if not path.is_file() or path.stat().st_size > maximum:
        raise EvaluationNotReady('CMS reference is absent or oversized')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != value['sha256']:
        raise EvaluationNotReady('CMS reference changed')
    return json.loads(raw)


def content_sha256(manifest):
    """Preserve the validated CMS manifest's original ASCII-JSON identity.

    The newer materializer reports empty FIFO/device lists that were absent
    during CMS validation. Only those empty reports are backward compatible;
    actual new special files still fail the pinned identity check.
    """
    body = {key: value for key, value in manifest.items() if key != 'destination'}
    for key in ('created_fifos', 'skipped_device_members'):
        if key in body:
            if body[key] != []:
                raise EvaluationNotReady('CMS special-file report differs from validated empty inventory')
            body.pop(key)
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def hosting(site=None):
    """The approved hosting proposal and its passing fresh-reset validation."""
    s = _site(site)
    proposal, validation = _ref(s.HOSTING_PROPOSAL), _ref(s.HOSTING_VALIDATION)
    second = _ref(validation.get('second', {}))
    if (validation.get('status') != 'passed' or validation.get('model_calls') != 0
            or validation.get('original_state_equivalence_verified') is not False
            or second.get('status') != 'passed' or second.get('gate', {}).get('fresh_reset_verified') is not True
            or second.get('gate', {}).get('native_authentication_verified') is not True):
        raise EvaluationNotReady(s.LABEL + ' hosting/reset validation did not pass')
    return proposal, validation


def base_identity(site=None):
    s = _site(site)
    proposal, _ = s.hosting()
    return dict(domain=s.DOMAIN, source=proposal['source'], oci_manifest_sha256=proposal['manifest']['sha256'],
        layer_proof_sha256=proposal['layer_proof']['sha256'], materialized_content_sha256=s.BASE_CONTENT_SHA256,
        hosting_proposal_sha256=s.HOSTING_PROPOSAL['sha256'], hosting_validation_sha256=s.HOSTING_VALIDATION['sha256'],
        original_state_equivalence_verified=False)


def validate_instance(instance, site=None):
    """Domain validator; the generic receipt checks have already passed."""
    s = _site(site)
    if instance.get('base_identity') != s.base_identity():
        raise EvaluationNotReady(s.LABEL + ' instance is not the validated third-party base')
    if instance['endpoint'].rstrip('/') != s.ENDPOINT:
        raise EvaluationNotReady(s.LABEL + ' endpoint differs from the validated origin')
    for key in ('service_validation', 'reset_validation'):
        if instance[key] != s.HOSTING_VALIDATION:
            raise EvaluationNotReady(s.LABEL + ' ' + key + ' must be the passing hosting validation')
    provenance = instance['state_provenance']
    build, copy = _ref(provenance.get('build_receipt')), _ref(provenance.get('copy_receipt'))
    if (build.get('kind') != s.BUILD_KIND or build.get('status') != 'completed'
            or build.get('materialized_content_sha256') != s.BASE_CONTENT_SHA256
            or copy.get('kind') != s.COPY_KIND or copy.get('status') != 'completed'
            or copy.get('state_path') != instance['state_path'] or copy.get('source_path') != build.get('baseline_path')
            or copy.get('build_receipt') != provenance['build_receipt']
            or copy.get('state_path') == build.get('baseline_path')):
        raise EvaluationNotReady(s.LABEL + ' per-cell build/copy provenance differs')
    return {s.NAME + '_base_content_sha256': s.BASE_CONTENT_SHA256, 'copy_receipt_sha256': provenance['copy_receipt']['sha256']}


def instance_extras():
    """Extra instance-receipt fields this site's worker needs (none for CMS)."""
    return {}


def released_account(source):
    """Released shopping_admin credentials via the reviewed private binding; never logged."""
    from .webprm_cms_hosting import read, reference
    proposal, _ = hosting()
    binding = read(reference(proposal['account_binding']))
    if binding.get('domain') != DOMAIN or any(
            hashlib.sha256((Path(source) / name).read_bytes()).hexdigest() != value
            for name, value in binding['source']['files'].items()):
        raise EvaluationNotReady('Released CMS authentication source/binding changed')
    path = reference(binding['native_account'], maximum=16384)
    if path.stat().st_mode & 0o077:
        raise EvaluationNotReady('Private native account permissions required')
    account = read(path, 16384)
    if set(account) != {'username', 'password'} or not all(isinstance(v, str) and v for v in account.values()):
        raise EvaluationNotReady('Released account fields differ')
    return account


def submit_admin_login(page):
    """Submit once and await DOM navigation within the existing auth watchdog.

    A cold Magento dashboard can exceed the 15-second element-action timeout.
    Keep that timeout for locating/clicking the button and use a separate bound
    for its actual navigation; a timeout never triggers a second submission.
    """
    with page.expect_navigation(wait_until='domcontentloaded', timeout=60000):
        page.get_by_role('button', name='Sign in', exact=True).click(no_wait_after=True)


def authenticate(source, instance, chromium, output):
    """Released shopping_admin login (as in webprm_cms_native.browser_control)."""
    from scripts.check_webprm_reddit_reset import write_new
    from .webprm_cms_native import DASHBOARD_PATH
    from .webprm_webarena_worker import write_json
    from playwright.sync_api import sync_playwright
    account = released_account(source)
    origin = instance['endpoint'].rstrip('/').removesuffix('/admin')
    if origin != ORIGIN:
        raise EvaluationNotReady('CMS login origin differs')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chromium, headless=True)
        try:
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(15000)
            def visit(path):
                response = page.goto(origin + path, wait_until='domcontentloaded', timeout=30000)
                if (response is None or response.status != 200 or
                        '{0.scheme}://{0.netloc}'.format(urlsplit(page.url)) != origin):
                    raise EvaluationNotReady('Native CMS route failed or escaped its origin')
            visit('/admin')
            page.get_by_placeholder('user name').fill(account['username'])
            page.get_by_placeholder('password').fill(account['password'])
            submit_admin_login(page)
            page.wait_for_load_state('domcontentloaded')
            visit(DASHBOARD_PATH)
            if 'Dashboard' not in page.content() or page.get_by_placeholder('user name').count():
                raise EvaluationNotReady('Native CMS dashboard authentication failed')
            state = context.storage_state()
            write_new(Path(output) / 'storage-state-private.json', state, private=True)
            write_json(Path(output) / 'authentication.json', dict(authenticated=True, domain=DOMAIN,
                account_binding_sha256=hosting()[0]['account_binding']['sha256'],
                state_provenance_sha256=instance['state_provenance_sha256']))
            return state
        except Exception as error:
            message = str(error)
            for value in account.values():
                message = message.replace(value, '[REDACTED]')
            raise EvaluationNotReady('Native authentication failed: ' + message[:2000]) from None
        finally:
            browser.close()


def website_ready(base_url=ORIGIN):
    """Readiness probe run inside the instance's network namespace."""
    from urllib.request import ProxyHandler, build_opener
    with build_opener(ProxyHandler({})).open(base_url + '/admin', timeout=5) as response:
        if response.status != 200:
            raise RuntimeError('CMS admin login page not ready')
        response.read(1024)
