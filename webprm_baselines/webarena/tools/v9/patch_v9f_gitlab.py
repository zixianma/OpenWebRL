# v9f (2026-10-26): ~25% of GitLab site starts in v9b/v9e failed in `gitlab-ctl reconfigure` at
# runit_service[alertmanager]: "sv restart /opt/gitlab/service/alertmanager ... timeout: down" -> reconfigure rc 1 ->
# "GitLab controller exited before configured" -> worker_exception (cell invalid). The monitoring stack (Prometheus,
# exporters, alertmanager) is not used by any web task; disable it in the per-cell gitlab.rb adaptation.
# exec'd after patch_v9_gitlab.py with NEW = package root.
vp = NEW/'source/scripts/validate_webprm_gitlab_hosting.py'
vt = vp.read_text()
a = "    updated+=\"\\npackage['modify_kernel_parameters'] = false\\n\"\n"
assert vt.count(a) == 1, 'v9f gitlab.rb anchor'
vt = vt.replace(a, a + "    updated+=\"prometheus_monitoring['enable'] = false\\nalertmanager['enable'] = false\\n\"  # v9f: unused monitoring stack\n")
vp.write_text(vt)
