#!/usr/bin/env python3
"""Read original CMS/Shopping/GitLab startup files without extracting an image.

Consumes an already verified Docker-save archive and its identity receipt. Only
small selected text files are copied into private content-addressed evidence;
there is no image execution, service startup, mount or network request.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import stat
import sys
import tarfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.inspect_webprm_cms_archive import BoundedReader,member_name,check_time,checked_path,GateClosed,write_new,stream_sha

TAGS={'shopping_admin':'shopping_admin_final_0719:latest','shopping':'shopping_final_0712:latest','gitlab':'gitlab-populated-final-port8023:latest'}
MAGENTO_ROOTS=('var/lib/mysql','var/lib/postgresql','var/lib/redis','var/lib/elasticsearch','usr/share/java/elasticsearch/data',
    'var/www/magento2/pub/media','var/www/magento2/var','var/www/magento2/generated','var/www/magento2/app/etc','run','tmp','var/log')
GITLAB_ROOTS=('var/opt/gitlab','etc/gitlab','var/log/gitlab','opt/gitlab/etc','run','tmp')
STARTUP_FILES=frozenset(('etc/os-release','etc/alpine-release','etc/issue','etc/supervisord.conf','etc/supervisor/supervisord.conf',
    'entrypoint.sh','custom-entrypoint.sh','docker-entrypoint.sh','start.sh','usr/local/bin/docker-php-entrypoint',
    'etc/my.cnf','etc/mysql/my.cnf','etc/nginx/nginx.conf','var/www/magento2/app/etc/env.php',
    'etc/gitlab/gitlab.rb','opt/gitlab/embedded/bin/runsvdir-start'))
PREFIXES=('etc/supervisor.d/','etc/supervisor/conf.d/','etc/mysql/','etc/my.cnf.d/','etc/nginx/conf.d/',
          'etc/php/','opt/gitlab/sv/')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_receipt(path):
    path=checked_path(path)
    if path.stat().st_size>4*2**20:raise GateClosed('Bounded identity receipt required')
    return json.loads(path.read_text())


def chosen(name,entrypoints):
    if name in STARTUP_FILES or name in entrypoints:return True
    if any(name.startswith(prefix) for prefix in PREFIXES):
        return name.endswith(('.conf','.ini','.sh','.php','.rb','/run','/finish','/log/run'))
    return False


def inspect(archive,proof,domain,deadline,*,member_cap=2000000,text_cap=65536,total_text_cap=2*2**20):
    archive=checked_path(archive)
    if (domain not in TAGS or proof.get('status')!='offline_archive_identity_consistent' or proof.get('image_tag')!=TAGS[domain] or
        proof.get('archive_path')!=str(archive) or proof.get('archive_bytes')!=archive.stat().st_size or
        not re.fullmatch('[0-9a-f]{64}',proof.get('archive_sha256',''))):raise GateClosed('Exact original acquisition inspection required')
    captured={};links={};count=0;selected_bytes=0;roles={root:dict(layer_regular_files=0,layer_regular_bytes=0) for root in (GITLAB_ROOTS if domain=='gitlab' else MAGENTO_ROOTS)}
    with archive.open('rb') as source:
        before=os.fstat(source.fileno())
        if stream_sha(source,before.st_size,deadline)!=proof['archive_sha256']:raise GateClosed('Verified original archive changed')
        source.seek(0)
        with tarfile.open(fileobj=BoundedReader(source,4*2**20,deadline),mode='r:') as outer:
            inventory={}
            for item in outer:
                name=member_name(item.name)
                if name in inventory or len(inventory)>=4096:raise GateClosed('Duplicate or oversized Docker-save inventory')
                if not (item.isfile() or item.isdir()) or item.sparse:raise GateClosed('Unsupported Docker-save outer member')
                inventory[name]=item
            def metadata(name):
                item=inventory[name]
                if not item.isfile() or item.size>4*2**20:raise GateClosed('Bounded Docker metadata required')
                return outer.extractfile(item).read(item.size+1)
            raw_manifest=metadata('manifest.json')
            if hashlib.sha256(raw_manifest).hexdigest()!=proof['manifest_sha256']:raise GateClosed('Original manifest identity changed')
            manifest=json.loads(raw_manifest)
            if len(manifest)!=1 or manifest[0].get('RepoTags')!=[TAGS[domain]]:raise GateClosed('Wrong original domain tag')
            config_raw=metadata(manifest[0]['Config'])
            if hashlib.sha256(config_raw).hexdigest()!=proof['config_sha256']:raise GateClosed('Original config identity changed')
            config=json.loads(config_raw);runtime=config.get('config',{})
            entrypoints={x.lstrip('/') for key in ('Entrypoint','Cmd') for x in (runtime.get(key) or []) if isinstance(x,str) and x.startswith('/')}
            layers=manifest[0]['Layers']
            if (len(layers)!=len(proof['layers']) or config.get('rootfs',{}).get('diff_ids')!=[x['diff_id'] for x in proof['layers']]):raise GateClosed('Original layer proof incomplete')
            for number,name in enumerate(layers):
                layer_info=inventory[name];expected=proof['layers'][number]
                if expected['index']!=number or expected['path']!=name or expected['bytes']!=layer_info.size:raise GateClosed('Ordered layer proof differs')
                stream=outer.extractfile(layer_info)
                if stream_sha(stream,layer_info.size,deadline)!=expected['sha256']:raise GateClosed('Raw layer hash changed')
                stream.seek(0);new_files={};new_links={};deletions=[]
                with tarfile.open(fileobj=BoundedReader(stream,4*2**20,deadline),mode='r:') as layer:
                    for item in layer:
                        check_time(deadline);count+=1
                        if count>member_cap:raise GateClosed('Startup scan member cap')
                        name=member_name(item.name);parts=PurePosixPath(name).parts
                        if parts[-1].startswith('.wh.'):
                            parent='/'.join(parts[:-1]);target=parts[-1][4:]
                            deletions.append((parent,True) if target=='.wh..opq' else ((parent+'/' if parent else '')+target,False))
                            continue
                        if item.isfile():
                            for root,row in roles.items():
                                if name==root or name.startswith(root+'/'):
                                    row['layer_regular_files']+=1;row['layer_regular_bytes']+=item.size
                        if chosen(name,entrypoints):
                            if len(new_files)+len(new_links)+len(captured)+len(links)>=512:raise GateClosed('Selected startup file-count cap')
                            if item.isfile():
                                if item.size>text_cap or selected_bytes+item.size>total_text_cap:raise GateClosed('Selected original startup text exceeds bound')
                                raw=layer.extractfile(item).read(item.size+1)
                                if len(raw)!=item.size or b'\0' in raw:raise GateClosed('Selected startup member is not bounded text')
                                raw.decode('utf-8');selected_bytes+=item.size
                                new_files[name]=dict(layer=number,bytes=item.size,sha256=hashlib.sha256(raw).hexdigest(),raw=raw)
                            elif item.issym() or item.islnk():new_links[name]=dict(layer=number,target=item.linkname,type='symlink' if item.issym() else 'hardlink')
                            elif not item.isdir():raise GateClosed('Selected startup configuration has special filesystem semantics')
                        layer.members.clear()
                for prefix,opaque in deletions:
                    for mapping in (captured,links):
                        for path in list(mapping):
                            if opaque and not prefix or path.startswith(prefix+'/') or not opaque and path==prefix:mapping.pop(path)
                for path,value in new_files.items():captured[path]=value;links.pop(path,None)
                for path,value in new_links.items():links[path]=value;captured.pop(path,None)
        after=os.fstat(source.fileno())
    identity=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    if identity(before)!=identity(after) or identity(after)!=identity(archive.stat()):raise GateClosed('Original archive changed during startup scan')
    result=dict(schema='original_site_startup_inventory_v1',domain=domain,image_tag=TAGS[domain],archive_sha256=proof['archive_sha256'],
        config_sha256=proof['config_sha256'],ordered_diff_ids=[x['diff_id'] for x in proof['layers']],
        startup_config={k:runtime.get(k) for k in ('Entrypoint','Cmd','User','WorkingDir','Volumes','ExposedPorts','StopSignal')},
        environment_names=[x.split('=',1)[0] for x in runtime.get('Env',[])],
        startup_files={name:{k:v for k,v in row.items() if k!='raw'} for name,row in captured.items()},
        unresolved_selected_links=links,mutable_root_candidates=roles,
        mutable_inventory_semantics='Conservative cumulative layer counts, not final merged-tree size; private full-root cloning remains required.',
        selected_text_bytes=selected_bytes,scanned_members=count,extracted_image=False,executed_image=False,ready_to_host=False,
        next_gate='Review original startup scripts/services and actual DB/media paths, then pin bounded rootless startup and native mutation/reset controls. Verified-derived image scripts are not substitutes.')
    return result,captured


def save(root,result,captured,proof_ref):
    root=checked_path(root);root.mkdir(mode=0o700,parents=True,exist_ok=False);private=root/'private-startup-files';private.mkdir(mode=0o700)
    for row in captured.values():
        path=private/(row['sha256']+'.txt')
        if path.exists():continue
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as out:out.write(row['raw']);out.flush();os.fsync(out.fileno())
    result=dict(result,archive_identity_receipt=proof_ref,private_text_directory=str(private))
    write_new(root/'startup-inventory.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',required=True,type=Path);p.add_argument('--identity',required=True,type=Path)
    p.add_argument('--domain',choices=tuple(TAGS),required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--seconds',type=int,required=True)
    a=p.parse_args()
    if not 0<a.seconds<=7200 or not os.environ.get('SLURM_JOB_ID','').isdigit():raise GateClosed('Existing approved CPU allocation and bounded inspection duration required')
    proof=read_receipt(a.identity);result,captured=inspect(a.archive,proof,a.domain,time.time()+a.seconds)
    save(a.output,result,captured,dict(path=str(checked_path(a.identity)),sha256=sha(a.identity)))
    print(json.dumps(dict(domain=a.domain,startup_files=len(captured),executed_image=False,ready_to_host=False)))


if __name__=='__main__':main()
