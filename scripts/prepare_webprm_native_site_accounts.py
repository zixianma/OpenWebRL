#!/usr/bin/env python3
"""Bind released site login/helper accounts to private local evidence files.

Only parses pinned Python literals; never imports env_config, logs credentials,
contacts a site, creates a browser, or claims authentication has succeeded.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from openwebrl.webprm_source import ARBITER_REVISION,source_manifest
from scripts.inspect_webprm_cms_archive import checked_path,GateClosed,write_new

AUTH_FILES={
 'viewer/src/browser_env/env_config.py':'c178732e836b739e4f182f5709ea65f1ad5f0172acb099c8c759a45b94332ca1',
 'viewer/src/browser_env/auto_login.py':'b3bd57bbaf5e8bf7c86179f0863af3534d2b3ecc5f8885a216357d5bd4ba5da4'}
DOMAINS=('shopping_admin','shopping','gitlab')


def account_bindings(source,domain):
    if domain not in DOMAINS:raise GateClosed('Only the three original authenticated domains are supported')
    source=Path(source);manifest=source_manifest(source,AUTH_FILES,ARBITER_REVISION)
    if manifest['files']!=AUTH_FILES:raise GateClosed('Released native authentication source changed')
    tree=ast.parse((source/'viewer/src/browser_env/env_config.py').read_text())
    values=[n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ACCOUNTS' for t in n.targets)]
    if len(values)!=1:raise GateClosed('One released literal ACCOUNTS mapping required')
    accounts=ast.literal_eval(values[0]);selected={domain:accounts[domain]}
    if domain=='shopping':selected['shopping_site_admin']=accounts['shopping_site_admin']
    for row in selected.values():
        if not isinstance(row,dict) or set(row)!={'username','password'} or not all(isinstance(x,str) and x for x in row.values()):raise GateClosed('Released account contract changed')
    return selected,manifest


def prepare(source,domain,output):
    accounts,manifest=account_bindings(source,domain);output=checked_path(output)
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    files={'native-login-account-private.json':accounts[domain]}
    if domain=='shopping':files['native-helper-accounts-private.json']={'shopping_site_admin':accounts['shopping_site_admin']}
    refs={}
    for name,value in files.items():
        path=output/name;write_new(path,value)
        refs[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result=dict(schema='released_original_site_account_binding_v1',domain=domain,source=manifest,
        native_account=refs['native-login-account-private.json'],native_helper_accounts=refs.get('native-helper-accounts-private.json'),
        storage_state=None,live_authentication_verified=False,ready_to_host=False,
        disclosure='Private released credentials are deployment inputs. Native login, correct site identity and helper access must still be verified against an original fresh instance.')
    write_new(output/'account-binding.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--domain',choices=DOMAINS,required=True);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();result=prepare(a.source,a.domain,a.output)
    print(json.dumps(dict(domain=a.domain,private_account_binding=str(a.output/'account-binding.json'),live_authentication_verified=False)))


if __name__=='__main__':main()
