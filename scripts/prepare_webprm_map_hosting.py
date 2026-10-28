#!/usr/bin/env python3
"""Offline Map archive inventory and exact owned frontend configuration.

Never starts services, mounts, extracts files, downloads, submits allocations,
opens browsers or deletes state. Full archive inventory is intended to run in
an approved CPU preparation allocation after acquisition is verified.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import time

RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
COMMIT='8d0e0a20d50e4a8c02d1520195182e19fc3c080f'
DATASETS={
 'osm_tile_server.tar':dict(roles=('osm-data','osm-tiles'),kind='docker_volume',strip_components=5),
 'nominatim_volumes.tar':dict(roles=('nominatim-data','nominatim-flatnode'),kind='docker_volume',strip_components=5),
 'osm_dump.tar':dict(roles=('nominatim_inputs',),kind='direct',strip_components=0),
 'osrm_routing.tar':dict(roles=('car','bike','foot'),kind='routing',strip_components=0)}
RECEIPT_SCHEMA='webprm_original_site_instance_v1'


class Closed(ValueError):pass


def checked(value):
    p=Path(value)
    if not p.is_absolute() or p.resolve()!=p or RUNTIME not in p.parents or any(x.is_symlink() for x in (p,*p.parents)):
        raise Closed('Only canonical nonsymlink project runtime paths are accepted')
    return p


def put(path,value):
    path=checked(path);raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    if len(raw)>4*2**20:raise Closed('Hosting receipt limit exceeded')
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())


def clean_name(value):
    if not isinstance(value,str) or '\x00' in value or '\\' in value or len(value)>4096:raise Closed('Unsafe archive member name')
    # Tar writers commonly preserve a single leading ./; no empty or parent
    # component may influence the extraction mapping.
    while value.startswith('./'):value=value[2:]
    parts=PurePosixPath(value).parts
    if not parts or parts[0]=='/' or '..' in parts or value.startswith('/'):raise Closed('Archive path escapes its data root')
    return parts


def dataset_destination(filename,name,*,is_directory=False):
    parts=clean_name(name);spec=DATASETS[filename]
    if spec['kind']=='docker_volume':
        # Official command strips five components. Observe the exact prefix
        # from the archive rather than presuming the archive was produced in
        # one filesystem depth. Require the last prefix component 'volumes'.
        if len(parts)<=5:
            if is_directory:return None
            raise Closed('Regular volume member has no path after five-component prefix')
        if parts[4]!='volumes' or parts[5] not in spec['roles']:raise Closed('Volume archive does not match the official five-component extraction layout')
        if len(parts)==6 and is_directory:return parts[5]
        if len(parts)<7 or parts[6]!='_data':raise Closed('Unexpected Docker volume content outside _data')
        return '/'.join((parts[5],)+parts[7:])
    if spec['kind']=='routing' and parts[0] not in spec['roles']:raise Closed('Routing archive has no original car/bike/foot root')
    return '/'.join(parts)


class DigestReader:
    def __init__(self,stream,deadline,cap):self.stream,self.deadline,self.cap=stream,deadline,cap;self.bytes=0;self.hash=hashlib.sha256()
    def read(self,n=-1):
        if time.time()>=self.deadline:raise Closed('Archive inventory deadline')
        if n<0:n=1024*1024
        data=self.stream.read(min(n,self.cap-self.bytes+1));self.bytes+=len(data)
        if self.bytes>self.cap:raise Closed('Archive byte cap exceeded')
        self.hash.update(data);return data


def inspect(path,expected_sha256,kind,deadline,*,compressed_byte_cap=200*2**30,expanded_byte_cap=256*2**30,member_cap=2000000):
    path=checked(path)
    if not re.fullmatch('[0-9a-f]{64}',expected_sha256) or kind not in (*DATASETS,'frontend_source','docker_save'):raise Closed('Expected acquisition checksum and declared archive role required')
    size=path.stat().st_size
    if size>compressed_byte_cap:raise Closed('Acquired archive exceeds inspection cap')
    rows={};names=set();hazards=[];regular=members=0;source_candidates=[];docker_manifest=False
    with path.open('rb') as raw:
        reader=DigestReader(raw,deadline,compressed_byte_cap)
        with tarfile.open(fileobj=reader,mode='r|*') as archive:
            for item in archive:
                members+=1
                if members>member_cap or time.time()>=deadline:raise Closed('Archive member/time cap exceeded')
                parts=clean_name(item.name);name='/'.join(parts)
                if name in names:raise Closed('Duplicate archive destination requires explicit review')
                names.add(name)
                if item.sparse or any(x.startswith('GNU.sparse') for x in item.pax_headers):raise Closed('Sparse archive semantics are not approved')
                if kind in DATASETS:destination=dataset_destination(kind,name,is_directory=item.isdir())
                else:destination=name
                if item.isfile():
                    regular+=item.size
                    if regular>expanded_byte_cap:raise Closed('Expanded regular-file inventory cap exceeded')
                    role=destination.split('/')[0] if destination else 'prefix'
                    row=rows.setdefault(role,dict(regular_files=0,regular_bytes=0));row['regular_files']+=1;row['regular_bytes']+=item.size
                    if name=='manifest.json' and kind=='docker_save':docker_manifest=True
                    if kind=='frontend_source' and (name.endswith(('.sql','.dump','.pbf','.sql.gz')) or parts[-1] in ('PG_VERSION','Dockerfile','database.yml','settings.yml')):
                        if len(source_candidates)<64:source_candidates.append(dict(path=name,bytes=item.size))
                elif not item.isdir():
                    if len(hazards)<100:hazards.append(dict(path=name,type=item.type.decode('ascii','replace'),linkname=item.linkname))
        # Tar readers may stop at end markers before trailing blocks. Hash the
        # entire acquired object, including padding/trailing compressed bytes.
        while reader.read(2**20):pass
    if reader.bytes!=size or reader.hash.hexdigest()!=expected_sha256:raise Closed('Archive differs from the acquired content identity')
    if kind=='docker_save' and not docker_manifest:raise Closed('Image file is not a Docker-save archive')
    if kind in DATASETS:
        expected=set(DATASETS[kind]['roles'])
        if DATASETS[kind]['kind']!='direct' and not expected<=rows.keys():raise Closed('Required original data volume/profile is absent')
    return dict(schema='map_archive_inventory_v1',archive=str(path),archive_sha256=expected_sha256,kind=kind,
        compressed_bytes=size,regular_bytes=regular,members=members,role_totals=rows,
        hazards=hazards,has_nonordinary_members=bool(hazards),frontend_database_candidates=source_candidates,
        extracted=False,ready_to_host=False,
        followup='Review links/devices and exact data roots before extraction. Frontend source must prove its original browse-page database initialization, separately from Nominatim.')


def replace_once(text,pattern,replacement,expected=1):
    value,count=re.subn(pattern,lambda _:replacement,text,flags=re.MULTILINE)
    if count!=expected:raise Closed('Original frontend configuration pattern count differs from the pinned template')
    return value


def configure_frontend(source,output):
    """Copy only three endpoint configuration files into fresh private overlay."""
    source,output=checked(source),checked(output)
    if source==output or source in output.parents or output in source.parents:raise Closed('Base and private configuration overlays must be disjoint')
    files={
      'config/settings.yml': [('^\\s*nominatim_url:.*$', 'nominatim_url: "http://127.0.0.1:8085/"'),
                              ('^\\s*fossgis_osrm_url:.*$', 'fossgis_osrm_url: "http://127.0.0.1"')],
      'vendor/assets/leaflet/leaflet.osm.js':[(r"url: 'https://tile.openstreetmap.org/\{z\}/\{x\}/\{y\}\.png'", "url: 'http://127.0.0.1:8080/tile/{z}/{x}/{y}.png'")],
      'app/assets/javascripts/index/directions/fossgis_osrm.js':[
          ('__OSMCarSuffix__',':5000'),('__OSMBikeSuffix__',':5001'),('__OSMFootSuffix__',':5002')]}
    # Inputs must be either the pinned gasse templates or matching original
    # extracted files; no public backend is silently retained on pattern drift.
    prepared=[]
    for relative,changes in files.items():
        path=checked(source/relative)
        if path.stat().st_size>2**20:raise Closed('Frontend configuration unexpectedly large')
        original=path.read_bytes();value=original.decode()
        for pattern,replacement in changes:value=replace_once(value,pattern,replacement,expected=2 if relative.endswith('leaflet.osm.js') else 1)
        prepared.append((relative,original,value.encode()))
    output.mkdir(mode=0o700,parents=True,exist_ok=False);receipts=[]
    for relative,original,new in prepared:
        target=checked(output/relative);target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with target.open('xb') as out:out.write(new);out.flush();os.fsync(out.fileno())
        receipts.append(dict(path=relative,original_sha256=hashlib.sha256(original).hexdigest(),configured_sha256=hashlib.sha256(new).hexdigest()))
    result=dict(schema='map_private_frontend_endpoint_overlay_v1',source=str(source),output=str(output),files=receipts,
        functional_change='Default tile, geocoder and car/bike/foot routing endpoints use allocation-owned loopback services. Alternative public tile layers remain original references; their task relevance and network isolation must be checked before readiness.',
        host_services_started=False,ready_to_host=False)
    put(output/'configuration-receipt.json',result);return result


def service_spec():
    return dict(schema='map_owned_service_topology_v1',generic_instance_schema=RECEIPT_SCHEMA,
      containment='One owned user namespace; each original database/backend gets its own network namespace. Frontend web and browser share the cell namespace. Owned Unix-socket bridges expose backend services on cell loopback while preserving original internal ports and avoiding the three PostgreSQL5432 collisions. No Docker daemon, SSH or public backend fallback.',
      bridges=[dict(service=name,internal_port=inside,cell_loopback_port=outside,socket_role=name+'.sock') for name,inside,outside in (('frontend_db',5432,5432),('tile',80,8080),('nominatim',8080,8085),('osrm-car',5000,5000),('osrm-bike',5000,5001),('osrm-foot',5000,5002))],
      services=[
       dict(name='frontend_db',image='openstreetmap-website-db.tar.gz',argv=['postgres'],port=5432,
            mounts={'frontend-db':'/var/lib/postgresql/data'},state='Private pristine restored Postgres data per cell; exact initializer awaits original image/source inventory.'),
       dict(name='frontend_web',image='openstreetmap-website-web.tar.gz',argv=['bundle','exec','rails','s','-p','3000','-b','0.0.0.0'],port=3000,
            mounts={'frontend-source':'/app','web-tmp':'/app/tmp','web-storage':'/app/storage','web-node-modules':'/app/node_modules','pids':'/tmp/pids'},state='Private writable runtime; source starts from immutable original and reviewed endpoint overlay.'),
       dict(name='tile',image='overv/openstreetmap-tile-server@sha256:8685e981fd97616995a61ecde194077942e29238ff5a5ef3ee742e6c70d5d7c3',argv=['run'],port=80,
            mounts={'osm-data':'/data/database','osm-tiles':'/data/tiles'},state='Postgres/cache are writable owned backend state; read-only HTTP query admission, not a read-only filesystem claim.'),
       dict(name='nominatim',image='mediagis/nominatim@sha256:d8b7f3b7b03549b1f0716b10cbf0148e48a0768e1bb9f4c4a180262dd2a20e3d',argv=['/app/start.sh'],port=8080,
            mounts={'nominatim-data':'/var/lib/postgresql/14/main','nominatim-flatnode':'/nominatim/flatnode','nominatim_inputs':'/nominatim/data'},
            environment={'IMPORT_STYLE':'extratags','PBF_PATH':'/nominatim/data/us-northeast-latest.osm.pbf','IMPORT_WIKIPEDIA':'/nominatim/data/wikimedia-importance.sql.gz'},
            state='Restore acquired populated volume; prove no fresh import or online update before sharing query endpoint.'),
       *[dict(name='osrm-'+mode,image='project-osrm/osrm-backend@sha256:b1ca5d72da456e82f81b8732a095c2357b00710099efa844040d1eb40b4b5386',
            argv=['osrm-routed','--algorithm','mld','/data/us-northeast-latest.osrm'],port=5000,mounts={mode:'/data'},state='Original acquired precomputed routing files bind read-only.') for i,mode in enumerate(('car','bike','foot'))]],
      native_validation=[
       'Query tiles as PNG and nonempty geocoding results in Pittsburgh, Boston, NewYork, Philadelphia and NewHampshire.',
       'Query car,bike,foot routes for the same regional endpoints; verify native distance/duration and routes, not only HTTP200.',
       'Load native frontend search, directions and object description/browse pages; verify each dependent request targets owned loopback services.',
       'Verify original frontend browse-page database is initialized from acquired source/image data, not substituted with Nominatim or an empty database.',
       'Run released task1 search and task87 browse evaluator fixtures plus task53 three-mode route controls with no actor/provider calls.',
       'Capture generic receipt ownership/schema plus native service identities, config hashes, private state and process closure.'],
      reset_sequence=[
       'Retain immutable acquired objects, OCI config identities, inspected base tree and original initialized frontend database identity.',
       'Create a unique prospective cell claim before state work; reflink only after independent-copy and database-mutation reset tests pass on the actual filesystem.',
       'Restore fresh per-cell frontend DB/web-tmp/web-storage/node_modules/runtime state; do not reuse a previous cell state path.',
       'Shared tile/Nominatim backends require native read-only query validation and no write/admin/update endpoints; their writable WAL/cache remain owned per-allocation, separate from immutable bases.',
       'After a deliberate frontend mutation, stop all descendants, instantiate a second fresh state and prove exact pristine native DB/query output and no first-cell mutation leakage.',
       'Only after durable rollouts/judgments/config/provenance and empty owned-process closure, prospective lifecycle cleanup may reclaim that cell state under its approved scope; historical states remain untouched.'],
      remaining_code_gates=['Bounded extractor for observed archives and OCI whiteout semantics','Rootless startup command/env from exact original image configs and native multi-namespace bridge validation','Native map adapter and lease/reset proofs','Browser and helper-fixture validation within an exact approved CPU request'],
      ready_to_host=False,allocation_or_browser_started=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    inv=sub.add_parser('inspect');inv.add_argument('--archive',type=Path,required=True);inv.add_argument('--sha256',required=True)
    inv.add_argument('--kind',choices=(*DATASETS,'frontend_source','docker_save'),required=True);inv.add_argument('--seconds',type=int,required=True);inv.add_argument('--output',type=Path,required=True)
    cfg=sub.add_parser('configure');cfg.add_argument('--source',type=Path,required=True);cfg.add_argument('--output',type=Path,required=True)
    spec=sub.add_parser('spec');spec.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='inspect':
        if not 0<args.seconds<=14400:raise Closed('Explicit bounded inventory duration required')
        put(args.output,inspect(args.archive,args.sha256,args.kind,time.time()+args.seconds))
    elif args.mode=='configure':configure_frontend(args.source,args.output)
    else:put(args.output,service_spec())


if __name__=='__main__':main()
