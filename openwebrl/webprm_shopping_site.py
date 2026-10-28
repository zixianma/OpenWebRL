"""Live Shopping (customer storefront) site adapter for Table 5 cells.

The hosting/reset recipe was validated by CPU job 358540 (third-party, digest-pinned
OCI image; equivalence to the original WebArena Shopping state is NOT verified). The
generic Magento checks are shared with webprm_cms_site; this module supplies the
Shopping constants, the released customer login and the private helper-account
receipt that shopping_get_latest_order_url needs. Nothing runs at import.
"""
from __future__ import annotations

from pathlib import Path
import sys
from urllib.parse import urlsplit

from . import webprm_cms_site as magento
from .webprm_webarena import EvaluationNotReady, digest

N = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation')
ORIGIN = 'http://127.0.0.1:7770'
ENDPOINT = ORIGIN
DOMAIN = 'shopping'
LABEL = 'Shopping'
NAME = 'shopping'
HOSTING_PROPOSAL = dict(path=str(N/'shopping-hosting-validation-v5/proposal.json'),
    sha256='9ff7b5c22f0d68ca6f42dd71602044b965717de3d4b841cd3d983c17ff943b55')
HOSTING_VALIDATION = dict(path=str(N/'shopping-hosting-validation-v2/campaign/attempt-358540/validation-result.json'),
    sha256='60733d113a7ebd0a205528b2adbaeb395c13f364c25ef6b1d115a83bd8f22544')
# sha256 of the canonical materialization manifest without its 'destination' key
# (validation-result.json materialization.materialized_content_sha256).
BASE_CONTENT_SHA256 = 'e2554ade14221ef36ca692d58eafc34a7e7b22e09039f31d945be2ff1a4e7db8'
BUILD_KIND = 'shopping_thirdparty_baseline_materialization'
COPY_KIND = 'shopping_thirdparty_state_copy'
# ~4.5M-file image: probe 358020 materialized in 2894 s.
MATERIALIZE_SECONDS = 5400

_THIS = sys.modules[__name__]


def materializer_limits():
    from .webprm_cms_materializer import Limits
    return Limits(raw_bytes=80*2**30, final_bytes=80*2**30, members=30000000)


def hosting_profile():
    from .webprm_cms_hosting import SHOPPING
    return SHOPPING


def content_sha256(manifest):
    """Keep Shopping's validated UTF-8 identity and its explicit report fields."""
    return digest({key: value for key, value in manifest.items() if key != 'destination'})


def hosting():
    return magento.hosting(_THIS)


def base_identity():
    return magento.base_identity(_THIS)


def released_accounts(source):
    """Customer login and helper REST admin accounts via the reviewed private binding; never logged."""
    from .webprm_cms_hosting import Closed
    from .webprm_shopping_native import released_accounts as accounts
    proposal, _ = hosting()
    try:
        return accounts(source, proposal['account_binding'])
    except Closed as error:
        raise EvaluationNotReady(str(error)) from None


def helper_accounts_receipt():
    """The hash-bound private helper-account file the hosting proposal's binding names."""
    from .webprm_cms_hosting import read, reference
    proposal, _ = hosting()
    binding = read(reference(proposal['account_binding']))
    receipt = binding.get('native_helper_accounts')
    if not isinstance(receipt, dict) or set(receipt) != {'path', 'sha256'}:
        raise EvaluationNotReady('Shopping binding lacks its private helper-account receipt')
    return dict(path=receipt['path'], sha256=receipt['sha256'])


def instance_extras():
    return dict(native_helper_accounts=helper_accounts_receipt())


def validate_instance(instance):
    """Generic Magento chain plus the helper-account receipt from the reviewed binding."""
    result = magento.validate_instance(instance, _THIS)
    if instance.get('native_helper_accounts') != helper_accounts_receipt():
        raise EvaluationNotReady('Shopping instance helper-account receipt differs from the reviewed binding')
    return result


def authenticate(source, instance, chromium, output):
    """Released shopping customer login (as in webprm_shopping_native.browser_control)."""
    from scripts.check_webprm_reddit_reset import write_new
    from .webprm_shopping_native import CHECK_PATH, LOGIN_PATH
    from .webprm_webarena_worker import write_json
    from playwright.sync_api import sync_playwright
    account, helpers = released_accounts(source)
    origin = instance['endpoint'].rstrip('/')
    if origin != ORIGIN:
        raise EvaluationNotReady('Shopping login origin differs')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chromium, headless=True)
        try:
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(30000)
            def visit(path, *, wait_until='domcontentloaded'):
                response = page.goto(origin + path, wait_until=wait_until, timeout=60000)
                if (response is None or response.status != 200 or
                        '{0.scheme}://{0.netloc}'.format(urlsplit(page.url)) != origin):
                    raise EvaluationNotReady('Native Shopping route failed or escaped its origin')
            # Released auto_login uses Page.goto's default load event. Login
            # scripts can finish after DOMContentLoaded and update form keys.
            visit(LOGIN_PATH, wait_until='load')
            page.get_by_label('Email', exact=True).fill(account['username'])
            page.get_by_label('Password', exact=True).fill(account['password'])
            page.get_by_role('button', name='Sign In').click()
            page.wait_for_load_state('domcontentloaded')
            visit(CHECK_PATH)
            if page.url != origin + CHECK_PATH:
                raise EvaluationNotReady('Native Shopping customer authentication failed')
            state = context.storage_state()
            write_new(Path(output) / 'storage-state-private.json', state, private=True)
            write_json(Path(output) / 'authentication.json', dict(authenticated=True, domain=DOMAIN,
                account_binding_sha256=hosting()[0]['account_binding']['sha256'],
                state_provenance_sha256=instance['state_provenance_sha256']))
            return state
        except Exception as error:
            message = str(error)
            # Longest first: a short username can be a substring of a password.
            for value in sorted((*account.values(), *helpers['shopping_site_admin'].values()), key=len, reverse=True):
                message = message.replace(value, '[REDACTED]')
            raise EvaluationNotReady('Native authentication failed: ' + message[:2000]) from None
        finally:
            browser.close()


def website_ready(base_url=ORIGIN):
    """Readiness probe run inside the instance's network namespace."""
    from urllib.request import ProxyHandler, build_opener
    with build_opener(ProxyHandler({})).open(base_url + '/customer/account/login/', timeout=10) as response:
        if response.status != 200:
            raise RuntimeError('Shopping customer login page not ready')
        response.read(1024)
