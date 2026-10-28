#!/usr/bin/env python3
"""Freeze a bounded Map reconstruction owner offline; never launch or download."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import validate_webprm_map_hosting as owner
from scripts.validate_webprm_cms_hosting import bytes_new,ref,verify_source

h=owner.h


def make_recipe(original):
    """Retain reviewed inert primitives, replacing the former unsafe owner entirely."""
    tree=ast.parse(original);parts=[]
    functions={'sha','edit','prefix','inroot','tile_xy','get','backends','browser'}
    for item in tree.body:
        if isinstance(item,(ast.Import,ast.ImportFrom,ast.Assign)) or isinstance(item,ast.FunctionDef) and item.name in functions:
            parts.append(ast.get_source_segment(original,item))
    source='\n\n'.join(parts)+'\n'
    source=source.replace("OUT=Path(__file__).resolve().parent","OUT=Path(__file__).resolve().parent")
    for target in ('/data/database/postgres/postmaster.pid','/var/lib/postgresql/14/main/postmaster.pid'):
        source=source.replace('rm -f '+target,'if [ -e '+target+' ]; then test ! -e '+target+'.webprm-preserved; mv -- '+target+' '+target+'.webprm-preserved; fi')
    source=source.replace(' HOME=/root','')
    source=source.replace('cd /nominatim && nominatim refresh --website --functions',
        "cd /nominatim && NOMINATIM_DATABASE_DSN='pgsql:dbname=nominatim;user=nominatim' nominatim refresh --website --functions")
    source=source.replace('cd /nominatim && PGUSER=nominatim nominatim admin --warm',
        "cd /nominatim && NOMINATIM_DATABASE_DSN='pgsql:dbname=nominatim;user=nominatim' nominatim admin --warm")
    source=source.replace('with urllib.request.urlopen(url,timeout=timeout) as r:return r.status,r.read(),r.headers.get(\'Content-Type\')',
        'with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url,timeout=timeout) as r:return r.status,r.read(8*2**20),r.headers.get(\'Content-Type\')')
    source=source.replace('def browser(res):','def browser(res, *, storage_output, diagnostics_output=None):')
    # Keep progress even when a later native assertion or browser interaction
    # fails. These receipts never replace the original native verdicts.
    source=source.replace("    ev=native.load_native_evaluator(source)","""    def checkpoint():
        if diagnostics_output is not None:h.put(Path(diagnostics_output),rep)
    ev=native.load_native_evaluator(source)""")
    source=source.replace("        # Task 90 (url_match)","        checkpoint()\n        # Task 90 (url_match)")
    source=source.replace("        # Task 87 (program_html", "        checkpoint()\n        # Task 87 (program_html")
    source=source.replace("        # Task 159 (directions:", "        checkpoint()\n        # Task 159 (directions:")
    source=source.replace("        # Foot and bike routing", "        checkpoint()\n        # Foot and bike routing")
    source=source.replace("            except Exception as e:rep.setdefault('ui_routes',{})[idx_name]='error: '+str(e)[:200]", """            except Exception as e:
                rep.setdefault('ui_routes',{})[idx_name]='error: '+str(e)[:200]
                detail=dict(traceback=traceback.format_exc()[-16000:])
                try:
                    detail['url']=pg.url
                    detail['dom']=pg.evaluate(\"({sidebar: document.querySelector('#sidebar_content')?.innerText, inputs: Array.from(document.querySelectorAll('input[name=route_from],input[name=route_to]')).map(e=>({name:e.name,value:e.value,visible:!!(e.offsetWidth||e.offsetHeight||e.getClientRects().length)})), engines:Array.from(document.querySelectorAll('select.routing_engines')).map(e=>({value:e.value,options:Array.from(e.options).map(o=>o.text)}))})\")
                except Exception as diagnostic_error:detail['diagnostic_error']=type(diagnostic_error).__name__
                rep.setdefault('ui_route_diagnostics',{})[idx_name]=detail
            checkpoint()""")
    source=source.replace("rep['home']=dict(status=r.status if r else None", "rep['home']=dict(status=r.status if r else None")
    source=source.replace("pg=c.new_page();pg.set_default_timeout(120000)","rep['initial_cookie_count']=len(c.cookies())\n        pg=c.new_page();pg.set_default_timeout(120000)")
    source=source.replace("pg.fill('[name=\"route_from\"]'","pg.fill('input[name=\"route_from\"]:visible'")
    source=source.replace("pg.fill('[name=\"route_to\"]'","pg.fill('input[name=\"route_to\"]:visible'")
    source=source.replace("pg.press('[name=\"route_to\"]'","pg.press('input[name=\"route_to\"]:visible'")
    source=source.replace("k=str(u.port) if u.port else u.scheme;rep['allowed'][k]=rep['allowed'].get(k,0)+1;r.continue_()",
        "k=str(u.port) if u.port else u.scheme;rep['allowed'][k]=rep['allowed'].get(k,0)+1\n                if sum(rep['allowed'].values())>10000:r.abort();return\n                r.continue_()")
    source=source.replace('        c.close();b.close()',
        "        if sum(rep['allowed'].values())>10000:raise RuntimeError('Browser request cap')\n        storage=json.dumps(c.storage_state(),sort_keys=True).encode()\n        Path(storage_output).write_bytes(storage);Path(storage_output).chmod(0o600)\n        rep['storage_state_sha256']=hashlib.sha256(storage).hexdigest()\n        c.close();b.close()")
    if 'rm -' in source or 'HOME=' in source or 'def main' in source or 'def inside' in source:
        raise h.Closed('Historical cleanup/owner escaped recipe reduction')
    ast.parse(source)
    return source


def prepare(root):
    root=h.private_path(root);root.mkdir(mode=0o700,exist_ok=False);frozen=root/'source';frozen.mkdir(mode=0o700)
    files=set(owner.FILES)
    # The native evaluator lazily imports its helper/transcript modules.
    files.update(str(p.relative_to(owner.REPO)) for p in (owner.REPO/'openwebrl').glob('webprm_*.py'))
    files.add('scripts/prepare_webprm_map_validation.py')
    for name in sorted(files):bytes_new(frozen/name,(owner.REPO/name).read_bytes())
    original=owner.N/'map-bringup-probe-v1/probe.py'
    reduced=make_recipe(original.read_text());recipe=frozen/'scripts/webprm_map_recipe.py';bytes_new(recipe,reduced.encode())
    metadata=owner.N/'map-bringup-acquire-v1/inventory/images.jsonl'
    images=[json.loads(line) for line in metadata.read_text().splitlines()]
    inputs=[]
    for row in images:
        if row.get('status')!='ok':raise h.Closed('Acquired Map image conversion failed')
        inputs.append(dict(path=str(owner.N/'map-bringup-acquire-v1/sif'/(row['name']+'.sif')),sha256=row['sif_sha256'],role='converted_image'))
    backend=owner.N/'map-original-backend-resume-exception-v1/completed.json'
    for row in h.read(backend)['result']['items']:
        if row['filename'].split('/')[-1] not in ('osm_tile_server.tar','nominatim_volumes.tar','osrm_routing.tar'):continue
        inputs.append(dict(path=row['path'],sha256=row['sha256'],role='original_backend_dataset'))
    frontend=owner.N/'map-original-frontend-acquisition-v2/completed.json'
    f=next(row for row in h.read(frontend)['files'] if row['filename']=='openstreetmap-website.tar.gz')
    inputs.append(dict(path=str(frontend.parent/f['filename']),sha256=f['sha256'],role='original_frontend_source'))
    if len(inputs)!=9:raise h.Closed('Expected five converted images, three datasets and frontend source')
    references=[original,metadata,backend,frontend,owner.N/'map-gasse-webarena__openstreetmap-templates__docker-compose.yml',
        owner.N/'map-gasse-webarena__03_docker_create_containers.sh',owner.N/'map-gasse-template-sources-v1.json']
    evidence=dict(schema='map_fresh_schema_reconstruction_source_basis_v1',references=[ref(p) for p in references],
        frontend_database='released_fresh_schema_reconstruction',browse_equivalence_supported=False,
        findings=['Pinned original website-db image has an empty PostgreSQL data directory.',
                  'Pinned acquired source Dockerfile installs postgres:11 and init shell creates the openstreetmap role and grant only.',
                  'Pinned compose initializes a fresh named db-data volume; no populated frontend dataset was acquired.',
                  'Acquired frontend db/structure.sql has 53 tables and schema_migrations inserts; no populated OSM objects.',
                  'All 26 released Map tasks depend on map search/routing; native controls cover URL90, search87, route159.',
                  'No OSM objects are invented and original populated browse-page equivalence is unsupported.'])
    bytes_new(root/'source-basis.json',(json.dumps(evidence,sort_keys=True,indent=2)+'\n').encode())
    bytes_new(root/'account-binding.json',b'{"domain":"map","authentication_required":false}\n')
    native=verify_source(owner.common.NATIVE);bytes_new(root/'native-source.json',(json.dumps(native,sort_keys=True,indent=2)+'\n').encode())
    plan=dict(protocol=owner.PROTOCOL,planned_configs=dict(recipe=ref(recipe)),source_basis=ref(root/'source-basis.json'),
        resource_scope='One 4-hour zero-GPU allocation; full independent immutable base plus state A and state B.',
        mutation='Frontend PostgreSQL dedicated marker table; no OSM object modifications.',
        reset='Pristine independent full copies restore frontend schema/marker and browser session; all native controls rerun.',
        closures='All service descendants owned by each phase subreaper; outer owner awaits every copy, service and browser.',
        network='Only loopback in one private network namespace per state; no public fallback, import, replication or model calls.',
        original_state_equivalence_verified=False,browse_equivalence_supported=False)
    bytes_new(root/'service-plan.json',(json.dumps(plan,sort_keys=True,indent=2)+'\n').encode())
    batch='\n'.join(['#!/bin/bash','#SBATCH --account=zixianma','#SBATCH --partition=gpu-h200','#SBATCH --qos=normal',
        '#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=8','#SBATCH --mem=128G','#SBATCH --time=240',
        '#SBATCH --job-name=webprm-map-validate','#SBATCH --output='+str(root/'slurm-%j.log'),'set -euo pipefail',
        'export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 BASH_ENV=/dev/null ENV=/dev/null',
        'exec '+shlex.quote(owner.common.PYTHON)+' -B '+shlex.quote(str(frozen/'scripts/validate_webprm_map_hosting.py'))+
        ' --run --proposal '+shlex.quote(str(root/'proposal.json'))+' --approval "$1"',''])
    bytes_new(frozen/'hosting.sbatch',batch.encode())
    proposal=dict(schema_version=1,status='prepared_not_approved',protocol=owner.PROTOCOL,campaign_root=str(root/'campaign'),
        frozen_source=str(frozen),source_sha256={str(p.relative_to(frozen)):h.sha(p) for p in frozen.rglob('*.py')},
        resources=owner.RESOURCES,limits=owner.LIMITS,source='third_party_digest_pinned_oci_candidate',
        original_state_equivalence_verified=False,browse_equivalence_supported=False,
        frontend_database='released_fresh_schema_reconstruction',service_plan=ref(root/'service-plan.json'),batch_script=ref(frozen/'hosting.sbatch'),
        account_binding=ref(root/'account-binding.json'),native_source_receipt=ref(root/'native-source.json'),inputs=inputs,
        source_basis=ref(root/'source-basis.json'),nltk_pythonpath=str(owner.common.R/'nltk-runtime/venv/lib/python3.12/site-packages'),
        nltk_data=str(owner.common.R/'nltk-runtime/nltk_data'),scratch_accounting='cumulative_across_all_registered_attempts_including_other_nodes')
    bytes_new(root/'proposal.json',(json.dumps(proposal,sort_keys=True,indent=2)+'\n').encode())
    return dict(proposal_path=str(root/'proposal.json'),proposal_sha256=h.digest(proposal),status='offline_prepared_not_approved',allocations_submitted=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().root),indent=2))
