"""Live GitLab site adapter for v9 Table 5 cells.

The hosting/reset recipe was validated by CPU job 389698 (gitlab-hosting-validation-v8: third-party,
digest-pinned OCI image; equivalence to the original WebArena GitLab state is NOT verified). The generic
third-party checks (validated hosting, base identity, per-cell build/copy provenance) are the reviewed
Magento ones in webprm_cms_site, called with this module's constants; this module supplies the GitLab
constants, the released byteblaze login (as webprm_gitlab_native.browser_control) and readiness.
Nothing runs at import.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from urllib.parse import urlsplit

from . import webprm_cms_site as generic
from .webprm_webarena import EvaluationNotReady, digest

N = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation')
ORIGIN = 'http://127.0.0.1:8023'
ENDPOINT = ORIGIN
DOMAIN = 'gitlab'
LABEL = 'GitLab'
NAME = 'gitlab'
HOSTING_PROPOSAL = dict(path=str(N/'gitlab-hosting-validation-v8/proposal.json'),
    sha256='7b48bf4a359ff406b361eb2938ae0298052c210c4cb0af747f13364721115b6f')
HOSTING_VALIDATION = dict(path=str(N/'gitlab-hosting-validation-v8/campaign/attempt-389698/validation-result.json'),
    sha256='2b1af66e6a3608033d25a9ef9ad3c309b6a3e5d7ccb0f60ff67214a84b4ea178')
# validation-result.json materialization.materialized_content_sha256 (manifest without 'destination').
BASE_CONTENT_SHA256 = '3c68a967a1aa285c8cc6c466e34ee0614959821d12ac90a13d46209eea8279f4'
BUILD_KIND = 'gitlab_thirdparty_baseline_materialization'
COPY_KIND = 'gitlab_thirdparty_state_copy'
# Job 389698 materialized the ~34.2 GB base in 754 s (validator cap 2700 s).
MATERIALIZE_SECONDS = 2700

_THIS = sys.modules[__name__]


def materializer_limits():
    from .webprm_cms_materializer import Limits
    return Limits(raw_bytes=90*2**30, final_bytes=90*2**30, members=30000000, special_members='fifo_and_skip_devices')


def hosting_profile():
    from .webprm_cms_hosting import GITLAB
    return GITLAB


def content_sha256(manifest):
    """As validate_webprm_gitlab_hosting.content_sha256 (destination-independent identity)."""
    return digest({key: value for key, value in manifest.items() if key != 'destination'})


def hosting():
    return generic.hosting(_THIS)


def base_identity():
    return generic.base_identity(_THIS)


def instance_extras():
    return {}


def validate_instance(instance):
    return generic.validate_instance(instance, _THIS)


def released_account(source):
    """Released byteblaze credentials via the reviewed private binding; never logged."""
    from .webprm_cms_hosting import read, reference
    proposal, _ = hosting()
    binding = read(reference(proposal['account_binding']))
    if binding.get('domain') != DOMAIN or any(
            hashlib.sha256((Path(source) / name).read_bytes()).hexdigest() != value
            for name, value in binding['source']['files'].items()):
        raise EvaluationNotReady('Released GitLab authentication source/binding changed')
    path = reference(binding['native_account'], maximum=16384)
    if path.stat().st_mode & 0o077:
        raise EvaluationNotReady('Private native GitLab account permissions required')
    account = read(path, 16384)
    if set(account) != {'username', 'password'} or not all(isinstance(v, str) and v for v in account.values()):
        raise EvaluationNotReady('Released GitLab account fields differ')
    return account


def authenticate(source, instance, chromium, output):
    """Released GitLab sign-in (the validated webprm_gitlab_native.browser_control steps)."""
    from scripts.check_webprm_reddit_reset import write_new
    from .webprm_webarena_worker import write_json
    from playwright.sync_api import sync_playwright
    account = released_account(source)
    origin = instance['endpoint'].rstrip('/')
    if origin != ORIGIN:
        raise EvaluationNotReady('GitLab login origin differs')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chromium, headless=True)
        try:
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(60000)
            def visit(path):
                response = page.goto(origin + path, wait_until='load', timeout=90000)
                if (response is None or response.status != 200 or
                        '{0.scheme}://{0.netloc}'.format(urlsplit(page.url)) != origin):
                    raise EvaluationNotReady('Native GitLab route failed or escaped its origin')
            visit('/users/sign_in')
            page.get_by_test_id('username-field').fill(account['username'])
            page.get_by_test_id('password-field').fill(account['password'])
            page.get_by_test_id('sign-in-button').click()
            page.wait_for_load_state('load')
            visit('/-/profile')
            if page.url != origin + '/-/profile':
                raise EvaluationNotReady('Native GitLab authentication failed')
            state = context.storage_state()
            write_new(Path(output) / 'storage-state-private.json', state, private=True)
            write_json(Path(output) / 'authentication.json', dict(authenticated=True, domain=DOMAIN,
                account_binding_sha256=hosting()[0]['account_binding']['sha256'],
                state_provenance_sha256=instance['state_provenance_sha256']))
            return state
        except Exception as error:
            message = str(error)
            for value in sorted(account.values(), key=len, reverse=True):
                message = message.replace(value, '[REDACTED]')
            raise EvaluationNotReady('Native authentication failed: ' + message[:2000]) from None
        finally:
            browser.close()


def website_ready(base_url=ORIGIN):
    """Readiness probe run inside the instance's network namespace (no proxy, no redirect)."""
    import urllib.request
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args, **_kwargs):
            raise RuntimeError('GitLab readiness must not redirect')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(base_url + '/users/sign_in', timeout=10) as response:
        if response.status != 200:
            raise RuntimeError('GitLab sign-in page not ready')
        response.read(1024)
