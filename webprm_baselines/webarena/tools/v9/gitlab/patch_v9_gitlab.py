# v9 GitLab cell path (2026-10-26). exec'd by build_v9_gitlab.py stage 1 on a GitLab package copy, with NEW = package root
# and G = this directory. Only the GitLab package copies change; running Reddit/Shopping/CMS packages are untouched.
#   - files: webprm_gitlab_site.py (adapter), run_webprm_table5_gitlab.py (generated stage), the validated GitLab
#     hosting primitives (validate_webprm_gitlab_hosting.py, webprm_gitlab_native.py, webprm_gitlab_recovery.py) and the
#     gitlab-hosting-validation-v8 webprm_cms_hosting.py (identical CMS/Shopping profiles; GitLab profile = v2 plan)
#   - site registry: the GitLab adapter gets its instance validator and released login
#   - launcher: GitLab joins the third-party (Magento-style) hosted domains with its own stage script/readiness/times
#   - Magento reset auditor: dispatches the gitlab site module too (same build/copy/claim chain)
import shutil, subprocess, sys
from pathlib import Path

GV8 = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/gitlab-hosting-validation-v8/frozen-source')
S = NEW/'source'
for rel in ('scripts/validate_webprm_gitlab_hosting.py', 'openwebrl/webprm_gitlab_native.py', 'openwebrl/webprm_gitlab_recovery.py',
            'tests/test_validate_webprm_gitlab_hosting.py', 'tests/test_webprm_gitlab_native.py', 'tests/test_webprm_gitlab_recovery.py'):
    assert not (S/rel).exists(), rel
    shutil.copy2(GV8/rel, S/rel)
shutil.copy2(GV8/'openwebrl/webprm_cms_hosting.py', S/'openwebrl/webprm_cms_hosting.py')
shutil.copy2(G/'webprm_gitlab_site.py', S/'openwebrl/webprm_gitlab_site.py')
subprocess.run([sys.executable, str(G/'gen_gitlab_stage.py'), str(S/'scripts/run_webprm_table5_cms.py'),
                str(S/'scripts/run_webprm_table5_gitlab.py')], check=True)


def _patch(path, pairs):
    text = path.read_text()
    for a, b in pairs:
        assert text.count(a) == 1, (path.name, a[:100])
        text = text.replace(a, b)
    path.write_text(text)


_patch(S/'openwebrl/webprm_webarena_sites.py', [
    ("SITE_ADAPTERS = MappingProxyType({",
     "def _gitlab_validator(instance):\n    from .webprm_gitlab_site import validate_instance\n    return validate_instance(instance)\n\n\n"
     "def _gitlab_authenticate(source, instance, chromium, output):\n    from .webprm_gitlab_site import authenticate\n"
     "    return authenticate(source, instance, chromium, output)\n\n\nSITE_ADAPTERS = MappingProxyType({"),
    ("    'gitlab': SiteAdapter('gitlab', NATIVE_LOGIN),",
     "    # v9: third-party GitLab image validated by job 389698 (see webprm_gitlab_site).\n"
     "    'gitlab': SiteAdapter('gitlab', NATIVE_LOGIN, _gitlab_validator, _gitlab_authenticate),"),
])
_patch(S/'scripts/launch_webprm_table5_campaign.py', [
    ("HOSTED_DOMAINS = {'reddit','shopping_admin','shopping'}", "HOSTED_DOMAINS = {'reddit','shopping_admin','shopping','gitlab'}  # v9 GitLab"),
    ("MAGENTO_DOMAINS = ('shopping_admin','shopping')",
     "MAGENTO_DOMAINS = ('shopping_admin','shopping','gitlab')  # v9: third-party OCI sites (GitLab uses its own stage script)"),
    ("    'shopping':'scripts/run_webprm_table5_cms.py'}",
     "    'shopping':'scripts/run_webprm_table5_cms.py','gitlab':'scripts/run_webprm_table5_gitlab.py'}"),
    ("SITE_STARTUP_SECONDS = {'reddit':180,'shopping_admin':900,'shopping':1200}",
     "SITE_STARTUP_SECONDS = {'reddit':180,'shopping_admin':900,'shopping':1200,'gitlab':1200}"),
    ("SITE_STOP_SECONDS = {'reddit':30,'shopping_admin':150,'shopping':150}",
     "SITE_STOP_SECONDS = {'reddit':30,'shopping_admin':150,'shopping':150,'gitlab':240}"),
    ("SITE_LABELS = {'reddit':'Reddit','shopping_admin':'CMS','shopping':'Shopping'}",
     "SITE_LABELS = {'reddit':'Reddit','shopping_admin':'CMS','shopping':'Shopping','gitlab':'GitLab'}"),
    ("    'shopping':'from openwebrl.webprm_shopping_site import website_ready; website_ready()'}",
     "    'shopping':'from openwebrl.webprm_shopping_site import website_ready; website_ready()',\n"
     "    'gitlab':'from openwebrl.webprm_gitlab_site import website_ready; website_ready()'}"),
    ("    'shopping':'scripts/audit_webprm_table5_cms_cell.py'}",
     "    'shopping':'scripts/audit_webprm_table5_cms_cell.py','gitlab':'scripts/audit_webprm_table5_cms_cell.py'}"),
    ("SITE_CODE = {'shopping_admin':CMS_CODE,'shopping':SHOPPING_CODE}",
     "GITLAB_CODE = SHOPPING_CODE + ('openwebrl/webprm_gitlab_site.py','scripts/run_webprm_table5_gitlab.py',\n"
     "    'scripts/validate_webprm_gitlab_hosting.py','openwebrl/webprm_gitlab_native.py','openwebrl/webprm_gitlab_recovery.py')\n"
     "SITE_CODE = {'shopping_admin':CMS_CODE,'shopping':SHOPPING_CODE,'gitlab':GITLAB_CODE}"),
    ("SITE_MODULES = {'shopping_admin':'openwebrl.webprm_cms_site','shopping':'openwebrl.webprm_shopping_site'}",
     "SITE_MODULES = {'shopping_admin':'openwebrl.webprm_cms_site','shopping':'openwebrl.webprm_shopping_site','gitlab':'openwebrl.webprm_gitlab_site'}"),
    ("STORAGE_SAMPLE_SECONDS = {'shopping':300,'shopping_admin':300,'reddit':300}",
     "STORAGE_SAMPLE_SECONDS = {'shopping':300,'shopping_admin':300,'reddit':300,'gitlab':300}"),
    ("DU_TIMEOUT_SECONDS = {'shopping':1800,'shopping_admin':1800,'reddit':1800}",
     "DU_TIMEOUT_SECONDS = {'shopping':1800,'shopping_admin':1800,'reddit':1800,'gitlab':1800}"),
])
_patch(S/'scripts/audit_webprm_table5_cms_cell.py', [
    ("    from openwebrl import webprm_cms_site, webprm_shopping_site\n"
     "    return {'shopping_admin': webprm_cms_site, 'shopping': webprm_shopping_site}.get(domain)",
     "    from openwebrl import webprm_cms_site, webprm_shopping_site, webprm_gitlab_site\n"
     "    return {'shopping_admin': webprm_cms_site, 'shopping': webprm_shopping_site, 'gitlab': webprm_gitlab_site}.get(domain)  # v9 GitLab"),
])
