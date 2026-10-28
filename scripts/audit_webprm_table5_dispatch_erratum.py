#!/usr/bin/env python3
"""Additive offline review of the frozen empty-dispatch audit rejection.

This never changes primary evidence, the original audit, live source, or state.
Run in a fresh Python process: all original audit dependencies must load from
the original source freeze. Only two explicitly hash-bound audit functions are
replaced in that process, and the replacement is disclosed in the new report.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PROTOCOL='table5_empty_dispatch_audit_erratum_v1'
ORIGINAL_SHA='02622744b726cb582219f149df3cd7689cdae4eff13dce5b85e048be0817b59d'
CORRECTED_SHA='c68a87879a910014b29f60267e6203c5392638226acd8a50dec210bea24f56da'
REJECTION=[dict(type='ValueError',message='Native dispatch differs from selected action')]
FUNCTIONS={'audit_empty_native_dispatch','audit_rollout'}


def sha(path):
    checksum=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(2**20),b''):checksum.update(data)
    return checksum.hexdigest()


def replacement_functions(path):
    if sha(path)!=CORRECTED_SHA:
        raise ValueError('Corrected auditor differs from reviewed erratum source')
    tree=ast.parse(Path(path).read_text())
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in FUNCTIONS]
    if {node.name for node in functions}!=FUNCTIONS or len(functions)!=2:
        raise ValueError('Exact two-function audit erratum required')
    return compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec')


def loaded_dependencies_are_frozen(root):
    for name,module in list(sys.modules.items()):
        if not (name.startswith('openwebrl.') or name.startswith('scripts.')):continue
        filename=getattr(module,'__file__',None)
        if filename and root not in Path(filename).resolve().parents:
            raise ValueError('Erratum needs a fresh process with original frozen audit dependencies')


def supplement(input_path,original_audit_path,corrected_source,output_path):
    input_path=Path(input_path);original_audit_path=Path(original_audit_path)
    corrected_source=Path(corrected_source);output_path=Path(output_path)
    inputs=json.loads(input_path.read_text())
    plan=json.loads(Path(inputs['worker_plan']['path']).read_text())
    matches=[Path(p) for p in plan['code_sha256'] if p.endswith('/scripts/audit_webprm_table5_cell.py')]
    if len(matches)!=1 or plan['code_sha256'][str(matches[0])]!=ORIGINAL_SHA or sha(matches[0])!=ORIGINAL_SHA:
        raise ValueError('This erratum supports only the identified original frozen auditor')
    replacement=replacement_functions(corrected_source)
    frozen_root=matches[0].parents[1]
    loaded_dependencies_are_frozen(frozen_root)
    spec=importlib.util.spec_from_file_location('_webprm_original_audit_for_erratum',matches[0])
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    inputs=original.read(input_path)
    original_report=original.read(original_audit_path)
    original_ref=dict(path=str(original_audit_path),sha256=sha(original_audit_path))
    source_ref=dict(path=str(corrected_source),sha256=CORRECTED_SHA)
    input_sha=sha(input_path)
    if (original_report.get('audit_verified') is not False or
            original_report.get('complete_for_disposal') is not False or
            original_report.get('errors')!=REJECTION or original_report.get('input_sha256')!=input_sha or
            original_report.get('artifact_refs')!=inputs.get('artifacts') or
            original_report.get('evidence_manifest_sha256')!=original.digest(inputs.get('artifacts'))):
        raise ValueError('Original rejection does not match the exact preserved input and error')
    output_path=original.canonical_runtime_path(str(output_path))
    campaign=original.canonical_runtime_path(inputs['campaign_root'])
    episode=original.canonical_runtime_path(inputs['artifacts']['result']['path']).parent
    if (campaign not in output_path.parents or output_path.parent==episode or episode in output_path.parents or
            frozen_root in output_path.parents):
        raise ValueError('Supplement must stay in the campaign outside immutable evidence and source')
    # Reproduce the original rejection before applying the disclosed correction.
    try:original.verify(inputs)
    except ValueError as error:
        if str(error)!=REJECTION[0]['message']:raise ValueError('Original audit has another unresolved failure') from error
    else:raise ValueError('Original rejection no longer reproduces; erratum is inapplicable')
    loaded_dependencies_are_frozen(frozen_root)
    exec(replacement,original.__dict__)
    rejection_proofs=[]
    corrected_check=original.audit_empty_native_dispatch
    def recorded_check(plan,trajectory,cursor,parsed,harness,row):
        corrected_check(plan,trajectory,cursor,parsed,harness,row)
        rejection_proofs.append(dict(turn=row['turn'],execution_error=row['execution_error'],
            preceding_observation_sha256=original.digest(trajectory[cursor-1]['observation']),
            selected_action_sha256=hashlib.sha256(row['proposal']['candidates'][row['selected_index']]['action'].encode()).hexdigest(),
            native_conversion_rejection_reproduced=True))
    original.audit_empty_native_dispatch=recorded_check
    report=dict(schema_version=1,protocol=PROTOCOL,input_sha256=input_sha,
        artifact_refs=inputs['artifacts'],evidence_manifest_sha256=original.digest(inputs['artifacts']),
        original_audit_reference=original_ref,original_audit_verified=False,
        original_rejection_reproduced=True,audit_verified=False,complete_for_disposal=False,errors=[],
        erratum=dict(original_auditor_sha256=ORIGINAL_SHA,corrected_source_reference=source_ref,
            implementation_reference=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
            replaced_functions=sorted(FUNCTIONS),primary_bytes_changed=False,primary_rerun=False),
        scope='Supplemental saved-evidence audit. Original rejection remains preserved. Only empty native dispatches whose exact pre-dispatch ActionParsingError is reconstructed from preceding saved bounds receive the corrected audit rule. Other frozen identity, closure, accounting, reset and native-evaluator checks still apply. No provider or website truth certification; no state disposal.')
    try:
        report.update(original.verify(inputs))
        if not rejection_proofs:raise ValueError('No reconstructed empty-dispatch rejection found')
        loaded_dependencies_are_frozen(frozen_root)
        for value in inputs['artifacts'].values():original.reference(value,root=campaign)
        original.reference(original_ref);original.reference(source_ref)
        if sha(input_path)!=input_sha or sha(matches[0])!=ORIGINAL_SHA:
            raise ValueError('Original evidence or frozen source changed during erratum audit')
        report['audit_verified']=True
    except Exception as error:
        report.update(audit_verified=False,complete_for_disposal=False)
        report['errors']=[dict(type=type(error).__name__,message=str(error)[:500])]
        report.pop('record',None)
    report['reconstructed_rejections']=rejection_proofs
    original.write_once(output_path,report)
    if report['audit_verified']:
        record=dict(report['record'],audit_sha256=sha(output_path),audit_erratum_protocol=PROTOCOL,
                    original_audit_sha256=original_ref['sha256'],corrected_audit_source_sha256=CORRECTED_SHA)
        original.write_once(output_path.with_suffix('.record.json'),record)
    return dict(audit_verified=report['audit_verified'],complete_for_disposal=report['complete_for_disposal'],
                errors=report['errors'],reconstructed_rejections=len(rejection_proofs),
                audit_path=str(output_path),audit_sha256=sha(output_path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('input','original-audit','corrected-source','output'):parser.add_argument('--'+flag,required=True,type=Path)
    args=parser.parse_args()
    result=supplement(args.input,args.original_audit,args.corrected_source,args.output)
    print(json.dumps(result,sort_keys=True))
    return 0 if result['audit_verified'] else 2


if __name__=='__main__':raise SystemExit(main())
