import fs from 'node:fs';
import readline from 'node:readline';
import { Effect } from 'effect';
import { make } from './browsercode/packages/bcode-browser/src/browser-execute.ts';
import { SessionStore } from './browsercode/packages/bcode-browser/src/session-store.ts';
const workspace=process.env.WORKSPACE_DIR!;
if (!workspace.startsWith('/gpfs/scrubbed/zixianma/openwebrl-runtime/')) throw new Error('Unexpected workspace');
const tool=await Effect.runPromise(make(workspace+'/.bcode'));
const sid='bound-episode';const session=SessionStore.get(sid);let sequence=0;
const originalCall=session._call.bind(session);
session._call=async(method,params={})=>{fs.appendFileSync(workspace+'/cdp-calls.jsonl',JSON.stringify({epoch:Date.now()/1000,method})+'\n');return originalCall(method,params);};
const lines=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
for await(const line of lines){
 let request:any;
 try{
  request=JSON.parse(line);let result:any;
  if(request.op==='init'){
   await session.connect({wsUrl:request.wsUrl});const targets=(await session.Target.getTargets({})).targetInfos;const page=targets.find((t:any)=>t.type==='page'&&t.targetId===request.targetId);if(!page)throw new Error('Bound owned target missing');await session.use(page.targetId);result={connected:true};
  }else if(request.op==='execute'){
   if(/\brm\b|\b(?:spawn|exec|execFile|fork)\s*\(/.test(request.args.code))throw new Error('Process launch/removal commands are not approved');
   result=await Effect.runPromise(tool.execute({...request.args,timeout:request.args.timeout??120000},{sessionID:sid,workspaceDir:workspace}));
   const attachments=[];
   for(const image of result.screenshots){const name=`${workspace}/screenshots/browsercode-${sequence++}.png`;fs.mkdirSync(workspace+'/screenshots',{recursive:true});fs.writeFileSync(name,Buffer.from(image.base64,'base64'));attachments.push({path:name,media_type:image.mime});}
   result={output:result.output+'\nReturn value: '+result.result,image_attachments:attachments};
  }else if(request.op==='capture'){
   result=await session.Page.captureScreenshot({format:'png',captureBeyondViewport:false,fromSurface:true});
  }else if(request.op==='close'){
   await SessionStore.evict(sid);console.log(JSON.stringify({id:request.id,ok:true,result:{closed:true}}));break;
  }else throw new Error('Unknown bridge operation');
  console.log(JSON.stringify({id:request.id,ok:true,result}));
 }catch(error){console.log(JSON.stringify({id:request?.id,ok:false,error:String(error)}));}
}
