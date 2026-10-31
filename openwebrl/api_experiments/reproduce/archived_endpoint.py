"""Read-only adapter for an already qualified native episode's last browser action.

Never selects an earlier screenshot. Original PNG bytes/geometry are preserved.
Manual captures additionally require the last awaited page operation to be that
capture and no later function calls other than print/str.
"""
import ast,hashlib,json,struct
from pathlib import Path
ALLOWED_GEOMETRIES={(1280,1800),(1280,1657)}
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def contained(root,path):
    path=Path(path).resolve();need(path.is_relative_to(root.resolve()),'Evidence escapes episode');return path
def final_manual_capture(code,capture):
    """Bind the exact last awaited screenshot and reject later page mutations."""
    tree=ast.parse(code.read_text());awaits=[n for n in ast.walk(tree)if isinstance(n,ast.Await)]
    need(awaits,'No awaited screenshot in final command')
    last=max(awaits,key=lambda n:(n.lineno,n.col_offset));call=last.value
    need(isinstance(call,ast.Call)and isinstance(call.func,ast.Attribute)and call.func.attr=='screenshot','Final page operation is not a capture')
    need(any(isinstance(n,ast.Constant)and n.value==capture.name for n in ast.walk(call)),'Final screenshot filename cannot be bound')
    for node in ast.walk(tree):
        if isinstance(node,ast.Call)and (node.lineno,node.col_offset)>=(last.end_lineno,last.end_col_offset):
            need(isinstance(node.func,ast.Name)and node.func.id in ('print','str'),'Commands execute after saved capture')
    need(not any(isinstance(n,(ast.For,ast.AsyncFor,ast.While,ast.If,ast.Try,ast.With,ast.AsyncWith,ast.FunctionDef,ast.AsyncFunctionDef))for n in tree.body),'Ambiguous control flow around final capture')
    return {'final_awaited_capture_line':last.lineno,'no_later_browser_calls':True,'code_sha256':sha(code)}
def native_endpoint(directory,task,canonical):
    directory=Path(directory).resolve();files={};steps_path=directory/'browser-steps.jsonl'
    steps=[json.loads(line)for line in steps_path.read_text().split('\n')if line.strip()];need(steps,'No browser steps')
    nums=[r['browser_step']for r in steps];need(nums==sorted(set(nums)),'Unordered/duplicated browser steps')
    need(all(type(r.get('action'))is str for r in steps),'Missing selected browser action')
    last=steps[-1];evidence_path=directory/'external-judge-evidence.json';evidence=read(evidence_path)if evidence_path.exists()else None
    trajectory_path=directory/'trajectory.json';trajectory=read(trajectory_path);answer=trajectory.get('info',{}).get('submission','');need(type(answer)is str,'Malformed saved final answer')
    actions=[r['action']for r in steps]+(['Final answer: '+answer]if answer else[])
    if evidence:
        need(evidence['task']==task and evidence['browser_steps_sha256']==sha(steps_path)and evidence['trajectory_sha256']==sha(trajectory_path),'Archived evidence identity differs')
        need(evidence['action_history']==actions,'Archived selected action history differs');files[str(evidence_path)]=sha(evidence_path)
    proof={'last_browser_step':last['browser_step'],'screenshot_scope':'last logged browser action; no later browser action','source':'archived automated post-command capture'}
    if last.get('screenshot_path'):
        image=contained(directory,directory/last['screenshot_path']);need(not last.get('screenshot_error'),'Final capture has error')
        need(evidence is not None,'No frozen export binding for final raw capture')
        candidates=[c for c in evidence['captures']if c['browser_step']==last['browser_step']and Path(c['path']).resolve()==image]
        need(len(candidates)==1 and sha(image)==candidates[0]['sha256'],'Automated final capture identity differs')
    else:
        need(evidence,'No exported capture provenance for manual final screenshot')
        candidates=[c for c in evidence['captures']if c['browser_step']==last['browser_step']]
        need(candidates,'No PNG from last logged browser interaction')
        code=contained(directory,directory/last['code_path']);need(sha(code)==last['code_sha256'],'Final selected command changed')
        qualified=[]
        for candidate in candidates:
            image=contained(directory,candidate['path'])
            try:binding=final_manual_capture(code,image)
            except ValueError:continue
            need(sha(image)==candidate['sha256'],'Manual final capture changed');qualified.append((image,binding))
        need(len(qualified)==1,'No unambiguous terminal capture after final selected action')
        image,binding=qualified[0];proof.update(source='archived manual capture as final awaited browser operation',manual_capture_binding=binding);files[str(code)]=sha(code)
    with image.open('rb')as stream:header=stream.read(24)
    need(header[:8]==b'\x89PNG\r\n\x1a\n'and header[12:16]==b'IHDR','Not original PNG');geometry=struct.unpack('>II',header[16:24]);need(geometry in ALLOWED_GEOMETRIES,'Unqualified archived geometry '+str(geometry))
    need(canonical['valid']is True,'Original canonical invalid must be preserved')
    # The qualified canonical collection binds endpoint/accounting validity.
    # Inspect its raw native endpoint as an additional independent check.
    result_path=directory/'result.json';result=read(result_path)
    need(result.get('native_execution_complete')is True and result.get('exit_status')in ('Submitted','LimitsExceeded'),'Original native actor endpoint incomplete')
    for path in (steps_path,trajectory_path,image,result_path):files[str(path)]=sha(path)
    return {'screenshot_path':str(image),'screenshot_sha256':sha(image),'screenshot_dimensions':list(geometry),'archival_geometry_preserved_no_resize':True,'geometry_differs_from_1800px_browsercode':geometry!=(1280,1800),'task':task,'actions':actions,**proof},files
