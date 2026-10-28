"""Bounded Map selected-member inventory. No extraction, execution or network.

Selected text is private evidence. Ordered OCI whiteouts are applied to selected
files only; this is not a general image extractor or a hosting-readiness proof.
"""
from __future__ import annotations
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time

from scripts.prepare_webprm_map_hosting import DATASETS, clean_name, dataset_destination

ROOTS=(Path('/tmp'),Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'))
SELECTED_EXACT=frozenset(('etc/os-release','var/lib/dpkg/status','docker-entrypoint.sh','usr/local/bin/docker-entrypoint.sh',
    'app/start.sh','app/init.sh','app/config.sh','run.sh','usr/local/bin/run.sh','etc/postgresql-common/createcluster.conf'))
TEXT_SUFFIXES=('/PG_VERSION','/postgresql.conf','/pg_hba.conf','/postmaster.opts','/database.yml','/settings.yml','/Dockerfile')


class Closed(ValueError):pass


def checked(value):
    path=Path(value)
    if (not path.is_absolute() or path.resolve()!=path or not any(root in path.parents for root in ROOTS) or
        any(p.is_symlink() for p in (path,*path.parents))):raise Closed('Canonical nonsymlink runtime or temporary path required')
    return path


def digest(raw):return hashlib.sha256(raw).hexdigest()
def json_digest(value):return digest(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def decode(raw):
    def pairs(rows):
        out={}
        for key,value in rows:
            if key in out:raise Closed('Duplicate metadata key')
            out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=pairs)


@dataclass
class Budget:
    deadline:float
    max_archive_bytes:int=200*2**30
    max_regular_bytes:int=240*2**30
    max_members:int=2000000
    max_selected_bytes:int=32*2**20
    max_selected_files:int=1024
    max_text_file_bytes:int=4*2**20
    regular_bytes:int=0
    members:int=0
    selected_bytes:int=0
    selected_files:int=0
    raw_tar_bytes:int=0

    def __post_init__(self):
        if type(self.deadline) not in (float,int) or not math.isfinite(self.deadline) or not time.time()<self.deadline<=time.time()+14400:
            raise Closed('Bounded future inventory deadline required')
        for name in ('max_archive_bytes','max_regular_bytes','max_members','max_selected_bytes','max_selected_files','max_text_file_bytes'):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:raise Closed('Positive integral inventory cap required')

    def check(self):
        if time.time()>=self.deadline:raise Closed('Inventory deadline reached')

    def member(self,item):
        self.check();self.members+=1
        if self.members>self.max_members or item.size<0:raise Closed('Archive member cap or invalid size')
        if item.sparse or any(k.startswith('GNU.sparse') for k in item.pax_headers):raise Closed('Sparse member semantics require separate review')
        if item.isfile():
            self.regular_bytes+=item.size
            if self.regular_bytes>self.max_regular_bytes:raise Closed('Expanded regular-file cap reached')

    def capture(self,stream,item):
        if item.size>self.max_text_file_bytes or self.selected_bytes+item.size>self.max_selected_bytes or self.selected_files>=self.max_selected_files:
            raise Closed('Selected metadata byte/file cap reached')
        self.check();raw=stream.read(item.size)
        if len(raw)!=item.size or b'\0' in raw:raise Closed('Selected metadata is truncated or not text')
        raw.decode('utf-8');self.selected_bytes+=len(raw);self.selected_files+=1
        return raw


class HashReader:
    def __init__(self,stream,budget,maximum,*,charge_raw=False):
        self.stream,self.budget,self.maximum=stream,budget,maximum;self.bytes=0;self.hash=hashlib.sha256();self.charge_raw=charge_raw
    def read(self,n=-1):
        self.budget.check()
        if n<0:raise Closed('Unbounded archive read is forbidden')
        if self.bytes>=self.maximum:
            # Reaching the exact declared file size is EOF for this view. The
            # descriptor/stat identity check independently binds the source size.
            return b''
        raw=self.stream.read(min(n,2**20,self.maximum-self.bytes))
        if self.charge_raw:
            self.budget.raw_tar_bytes+=len(raw)
            if self.budget.raw_tar_bytes>self.budget.max_regular_bytes:raise Closed('Aggregate raw tar expansion cap reached')
        self.bytes+=len(raw);self.hash.update(raw);return raw
    def drain(self):
        while self.read(2**20):pass


def chosen(name,entrypoints=()):
    return (name in SELECTED_EXACT or name in entrypoints or name=='PG_VERSION' or name.endswith(TEXT_SUFFIXES) or
        name.startswith(('etc/postgresql/','etc/nginx/','etc/apache2/','etc/cron.d/')) and name.endswith(('.conf','.ini','.sh')))


def scan_layer(stream,budget,index,*,entrypoints=(),role=None):
    files={};links={};deletions=[];replacements=[];names=set();destinations=set();roles={};initializers=[]
    with tarfile.open(fileobj=stream,mode='r|') as archive:
        for item in archive:
            budget.member(item)
            if item.isdir() and item.name in ('.','./'):continue
            parts=clean_name(item.name);name='/'.join(parts)
            if name in names:raise Closed('Duplicate archive member')
            names.add(name)
            destination=dataset_destination(role,name,is_directory=item.isdir()) if role in DATASETS else name
            if role in DATASETS and destination:
                if destination in destinations:raise Closed('Duplicate mapped archive destination')
                destinations.add(destination)
            if role in DATASETS and parts[-1].startswith('.wh.'):
                raise Closed('Volume archives cannot apply OCI whiteout semantics')
            if parts[-1].startswith('.wh.'):
                if not item.isfile() or item.size!=0:raise Closed('Only empty regular OCI whiteouts are supported')
                parent='/'.join(parts[:-1]);target=parts[-1][4:]
                if target=='.wh..opq':deletions.append((parent,True))
                else:
                    if not target or target in ('.','..'):raise Closed('Invalid whiteout target')
                    deletions.append(((parent+'/' if parent else '')+target,False))
                continue
            if item.isfile() or item.issym() or item.islnk():replacements.append(name)
            if item.isfile() and destination:
                key=destination.split('/')[0];v=roles.setdefault(key,dict(regular_files=0,regular_bytes=0));v['regular_files']+=1;v['regular_bytes']+=item.size
                if name.endswith(('.sql','.sql.gz','.dump','.pbf')) and len(initializers)<128:
                    initializers.append(dict(path=name,destination=destination,bytes=item.size))
            if chosen(name,entrypoints) or destination and chosen(destination,entrypoints):
                if item.isfile():
                    raw=budget.capture(archive.extractfile(item),item)
                    files[name]=dict(layer=index,source_member=name,destination=destination,bytes=len(raw),sha256=digest(raw),raw=raw)
                elif item.issym() or item.islnk():
                    if budget.selected_files>=budget.max_selected_files:raise Closed('Selected metadata byte/file cap reached')
                    budget.selected_files+=1
                    links[name]=dict(layer=index,source_member=name,destination=destination,type='symlink' if item.issym() else 'hardlink',target=item.linkname)
                elif not item.isdir():raise Closed('Selected member has unsupported special-file semantics')
            elif not (item.isfile() or item.isdir() or item.issym() or item.islnk()):
                raise Closed('Special archive members require separate review')
            archive.members.clear()
    return dict(files=files,links=links,deletions=deletions,replacements=replacements,roles=roles,initializer_candidates=initializers)


def apply_layer(files,links,layer):
    for prefix,opaque in layer['deletions']:
        for mapping in (files,links):
            for name in list(mapping):
                if (opaque and not prefix or name.startswith(prefix+'/') or not opaque and name==prefix):mapping.pop(name)
    # Replacing an ancestor directory with a file/link hides earlier selected
    # descendants, even when that ancestor itself was not selected for capture.
    for prefix in layer['replacements']:
        for mapping in (files,links):
            for name in list(mapping):
                if name.startswith(prefix+'/'):mapping.pop(name)
    for name in (*layer['files'],*layer['links']):
        if any(name.startswith(prefix+'/') for prefix in layer['replacements']):
            raise Closed('Selected member lies beneath a replaced file or link')
    for name,row in layer['files'].items():files[name]=row;links.pop(name,None)
    for name,row in layer['links'].items():links[name]=row;files.pop(name,None)


def file_view(path,expected_sha256,budget):
    path=checked(path);info=path.stat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size>budget.max_archive_bytes or
            re.fullmatch('[0-9a-f]{64}',expected_sha256) is None):raise Closed('Bounded regular acquisition input with SHA256 required')
    return path,info


def unchanged(path,before,after):
    fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    current=path.stat()
    if any(getattr(before,k)!=getattr(after,k) or getattr(after,k)!=getattr(current,k) for k in fields):
        raise Closed('Acquired input changed during inventory')


def inventory_dataset(path,expected_sha256,role,budget):
    if role not in (*DATASETS,'frontend_source'):raise Closed('Explicit original dataset/source role required')
    path,before=file_view(path,expected_sha256,budget)
    with path.open('rb') as raw:
        source=HashReader(raw,budget,before.st_size)
        if path.name.endswith('.gz'):
            unpacked=gzip.GzipFile(fileobj=source,mode='rb')
            expanded=HashReader(unpacked,budget,budget.max_regular_bytes,charge_raw=True)
            layer=scan_layer(expanded,budget,0,role=role);expanded.drain()
            if unpacked.read(1):raise Closed('Raw archive expansion cap exceeded')
            unpacked.close()
        else:layer=scan_layer(source,budget,0,role=role)
        source.drain();after=os.fstat(raw.fileno())
    unchanged(path,before,after)
    if source.bytes!=before.st_size or source.hash.hexdigest()!=expected_sha256:raise Closed('Acquisition content checksum differs')
    if role in DATASETS and DATASETS[role]['kind']!='direct' and not set(DATASETS[role]['roles'])<=layer['roles'].keys():
        raise Closed('Required original data roles are absent')
    return result(path,expected_sha256,role,layer['files'],layer['links'],budget,
        role_totals=layer['roles'],initializer_candidates=layer['initializer_candidates'])


def inventory_docker_save(path,expected_sha256,budget):
    """Stream a gzip/plain Docker-save archive; keep only bounded layer metadata."""
    path,before=file_view(path,expected_sha256,budget);outer_names=set();metadata={};layers={}
    with path.open('rb') as raw:
        source=HashReader(raw,budget,before.st_size)
        with tarfile.open(fileobj=source,mode='r|*') as archive:
            for item in archive:
                budget.check();name='/'.join(clean_name(item.name))
                if name in outer_names or len(outer_names)>=4096:raise Closed('Duplicate or oversized Docker-save outer inventory')
                outer_names.add(name)
                if not (item.isfile() or item.isdir()) or item.sparse or item.size<0:raise Closed('Unsafe Docker-save outer member')
                if not item.isfile():continue
                stream=archive.extractfile(item)
                if name.endswith('/layer.tar') or name.endswith('.tar'):
                    if item.size>budget.max_regular_bytes:raise Closed('Raw Docker layer exceeds cap')
                    view=HashReader(stream,budget,item.size,charge_raw=True)
                    layer=scan_layer(view,budget,-1);view.drain()
                    if view.bytes!=item.size:raise Closed('Truncated Docker layer')
                    layers[name]=dict(layer,raw_sha256=view.hash.hexdigest(),raw_bytes=view.bytes)
                elif name=='manifest.json' or re.fullmatch('[0-9a-f]{64}[.]json',name):
                    if item.size>4*2**20 or len(metadata)>=64:raise Closed('Bounded Docker-save metadata required')
                    content=stream.read(item.size)
                    if len(content)!=item.size:raise Closed('Truncated Docker metadata')
                    metadata[name]=content
                archive.members.clear()
        source.drain();after=os.fstat(raw.fileno())
    unchanged(path,before,after)
    if source.bytes!=before.st_size or source.hash.hexdigest()!=expected_sha256:raise Closed('Docker-save acquisition checksum differs')
    manifest=decode(metadata.get('manifest.json',b'null'))
    if not isinstance(manifest,list) or len(manifest)!=1:raise Closed('Exactly one original frontend image required')
    item=manifest[0];config_name=item.get('Config','')
    if re.fullmatch('[0-9a-f]{64}[.]json',config_name) is None or config_name not in metadata:raise Closed('Content-addressed Docker config required')
    config_raw=metadata[config_name]
    if digest(config_raw)!=config_name[:-5]:raise Closed('Docker config content hash differs')
    config=decode(config_raw);order=item.get('Layers');diffs=config.get('rootfs',{}).get('diff_ids',[])
    if (not isinstance(order,list) or not order or len(order)!=len(set(order)) or set(order)!=set(layers) or
        len(order)!=len(diffs) or config.get('rootfs',{}).get('type')!='layers' or config.get('os')!='linux' or config.get('architecture')!='amd64'):
        raise Closed('Original frontend Docker-save layer/platform topology differs')
    files={};links={};proof=[]
    for index,(name,wanted) in enumerate(zip(order,diffs)):
        layer=layers[name]
        if wanted!='sha256:'+layer['raw_sha256']:raise Closed('Docker raw diff ID differs')
        for mapping in ('files','links'):
            for row in layer[mapping].values():row['layer']=index
        apply_layer(files,links,layer);proof.append(dict(index=index,path=name,raw_diff_id=wanted,raw_bytes=layer['raw_bytes']))
    runtime=config.get('config',{})
    # Absolute entrypoints not in the fixed selector require an explicit second
    # reviewed capture; never claim an unseen startup script was inspected.
    entrypoints={x.lstrip('/') for k in ('Entrypoint','Cmd') for x in (runtime.get(k) or []) if isinstance(x,str) and x.startswith('/')}
    missing=sorted(entrypoints-set(files)-set(links))
    return result(path,expected_sha256,'docker_save_selected_final_overlay',files,links,budget,
        config_sha256=digest(config_raw),manifest_sha256=digest(metadata['manifest.json']),ordered_layers=proof,
        startup_config={k:runtime.get(k) for k in ('Entrypoint','Cmd','User','WorkingDir','Volumes','ExposedPorts')},
        image_tags=item.get('RepoTags'),uninspected_absolute_entrypoints=missing,selected_overlay_only=True)


def inventory_oci(root,manifest_sha256,budget):
    root=checked(root)
    def metadata(checksum,maximum=4*2**20):
        if re.fullmatch('[0-9a-f]{64}',str(checksum)) is None:raise Closed('Pinned OCI metadata digest required')
        path=checked(root/'blobs/sha256'/checksum)
        if path.stat().st_size>maximum:raise Closed('OCI metadata cap reached')
        raw=path.read_bytes()
        if digest(raw)!=checksum:raise Closed('Pinned OCI metadata content differs')
        return raw,decode(raw)
    manifest_raw,manifest=metadata(manifest_sha256)
    descriptor=manifest.get('config',{})
    if (re.fullmatch('sha256:[0-9a-f]{64}',str(descriptor.get('digest'))) is None or
            type(descriptor.get('size')) is not int or descriptor['size']<=0):raise Closed('Pinned OCI config descriptor required')
    config_raw,config=metadata(descriptor['digest'][7:])
    if len(config_raw)!=descriptor['size']:raise Closed('Pinned OCI config size differs')
    if (manifest.get('schemaVersion')!=2 or config.get('os')!='linux' or config.get('architecture')!='amd64' or
            config.get('rootfs',{}).get('type')!='layers' or len(manifest.get('layers',[]))!=len(config['rootfs'].get('diff_ids',[])) or not manifest.get('layers')):
        raise Closed('Pinned OCI platform/layer topology required')
    files={};links={};proof=[];runtime=config.get('config',{})
    entrypoints={x.lstrip('/') for k in ('Entrypoint','Cmd') for x in (runtime.get(k) or []) if isinstance(x,str) and x.startswith('/')}
    for index,(descriptor,diff_id) in enumerate(zip(manifest['layers'],config['rootfs']['diff_ids'])):
        if (descriptor.get('mediaType') not in ('application/vnd.docker.image.rootfs.diff.tar.gzip','application/vnd.oci.image.layer.v1.tar+gzip') or
            type(descriptor.get('size')) is not int or descriptor['size']<=0 or
            re.fullmatch('sha256:[0-9a-f]{64}',str(descriptor.get('digest'))) is None or re.fullmatch('sha256:[0-9a-f]{64}',diff_id) is None):
            raise Closed('Exact compressed and raw layer descriptors required')
        path,before=file_view(root/'blobs/sha256'/descriptor['digest'][7:],descriptor['digest'][7:],budget)
        if before.st_size!=descriptor['size']:raise Closed('Compressed layer size differs')
        with path.open('rb') as raw:
            compressed=HashReader(raw,budget,before.st_size)
            with gzip.GzipFile(fileobj=compressed,mode='rb') as expanded:
                # This cap binds all bytes of each raw tar, including padding.
                uncompressed=HashReader(expanded,budget,budget.max_regular_bytes,charge_raw=True)
                layer=scan_layer(uncompressed,budget,index,entrypoints=entrypoints)
                uncompressed.drain()
                if expanded.read(1):raise Closed('Raw layer expansion cap exceeded')
            compressed.drain();after=os.fstat(raw.fileno())
        unchanged(path,before,after)
        if compressed.hash.hexdigest()!=descriptor['digest'][7:] or 'sha256:'+uncompressed.hash.hexdigest()!=diff_id:
            raise Closed('Pinned compressed/raw layer digest differs')
        apply_layer(files,links,layer);proof.append(dict(index=index,compressed_sha256=descriptor['digest'][7:],raw_diff_id=diff_id,raw_bytes=uncompressed.bytes))
    return result(root,manifest_sha256,'oci_selected_final_overlay',files,links,budget,config_sha256=digest(config_raw),
        ordered_layers=proof,startup_config={k:runtime.get(k) for k in ('Entrypoint','Cmd','User','WorkingDir','Volumes','ExposedPorts')},
        selected_overlay_only=True)


def result(path,checksum,kind,files,links,budget,**extra):
    versions={}
    for name,row in files.items():
        if name.split('/')[-1]=='PG_VERSION':
            raw=row['raw'].decode().strip()
            if not re.fullmatch(r'(?:[1-9][0-9]?|9[.][0-6])',raw):raise Closed('Malformed PG_VERSION')
            versions[row['destination']]=dict(version=raw,source_member=name,sha256=row['sha256'])
    return dict(schema='map_selected_inventory_v1',path=str(path),content_sha256=checksum,kind=kind,
        selected_files={k:{n:v for n,v in row.items() if n!='raw'} for k,row in files.items()},
        captured=files,unresolved_selected_links=links,pg_versions=versions,
        member_count=budget.members,regular_bytes=budget.regular_bytes,raw_tar_bytes=budget.raw_tar_bytes,selected_bytes=budget.selected_bytes,
        extracted=False,executed=False,ready_to_host=False,**extra)


def save_inventory(output,value):
    output=checked(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    captured=value['captured'];private=output/'private-selected';private.mkdir(mode=0o700)
    for row in captured.values():
        target=private/(row['sha256']+'.txt')
        if target.exists():continue
        with os.fdopen(os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600),'wb') as f:
            f.write(row['raw']);f.flush();os.fsync(f.fileno())
    proof={k:v for k,v in value.items() if k!='captured'};proof['private_selected_directory']=str(private)
    raw=(json.dumps(proof,sort_keys=True,indent=2)+'\n').encode()
    if len(raw)>4*2**20:raise Closed('Inventory receipt cap exceeded')
    with os.fdopen(os.open(output/'inventory.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600),'wb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    return proof


def load_inventory(output,expected_receipt_sha256):
    """Reload only bounded hash-bound selected evidence for later version gates."""
    output=checked(output);path=checked(output/'inventory.json')
    if path.stat().st_size>4*2**20:raise Closed('Inventory receipt cap exceeded')
    raw=path.read_bytes()
    if digest(raw)!=expected_receipt_sha256:raise Closed('Saved inventory receipt changed')
    value=decode(raw);private=checked(output/'private-selected')
    rows=value.get('selected_files')
    if (value.get('schema')!='map_selected_inventory_v1' or value.get('private_selected_directory')!=str(private) or
        not isinstance(rows,dict) or len(rows)>1024):raise Closed('Bounded private inventory schema required')
    captured={};total=0
    for name,row in rows.items():
        if (not isinstance(row,dict) or re.fullmatch('[0-9a-f]{64}',str(row.get('sha256'))) is None or
            type(row.get('bytes')) is not int or not 0<=row['bytes']<=4*2**20):raise Closed('Invalid selected evidence descriptor')
        member=checked(private/(row['sha256']+'.txt'));info=member.stat()
        total+=row['bytes']
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_mode&0o077 or
            info.st_size!=row['bytes'] or total>32*2**20):raise Closed('Selected evidence privacy/byte cap differs')
        content=member.read_bytes()
        if digest(content)!=row['sha256'] or b'\0' in content:raise Closed('Selected evidence content changed')
        content.decode('utf-8');captured[name]=dict(row,raw=content)
    return dict(value,captured=captured)
