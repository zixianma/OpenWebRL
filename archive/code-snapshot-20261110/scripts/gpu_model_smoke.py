"""Real-checkpoint multimodal forward/backward smoke; not a full online RL run."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from slime.utils.ppo_utils import compute_policy_loss


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    device=torch.device('cuda:0')
    print('Loading processor and checkpoint',flush=True)
    processor=AutoProcessor.from_pretrained(args.checkpoint,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.checkpoint,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to(device)
    image=Image.new('RGB',(448,448),'white'); draw=ImageDraw.Draw(image)
    draw.rectangle((80,150,368,260),fill='blue');draw.text((140,190),'CONTINUE',fill='white')
    messages=[{'role':'user','content':[{'type':'image'}, {'type':'text','text':'Describe the button in one short sentence.'}]}]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[prompt],images=[image],return_tensors='pt').to(device)
    model.eval()
    with torch.no_grad():
        generated=model.generate(**inputs,max_new_tokens=32,do_sample=False)
    response=generated[:,inputs.input_ids.shape[1]:]
    response_text=processor.batch_decode(response,skip_special_tokens=True)[0]
    print('Generated:',response_text,flush=True)
    # Optimize the actual sampled response with a known positive test advantage.
    # This deliberately tests policy-loss numerics, not environment success.
    full=dict(inputs);full['input_ids']=generated;full['attention_mask']=torch.ones_like(generated)
    n=response.shape[1]
    def log_probs():
        logits=model(**full,use_cache=False,logits_to_keep=n+1).logits[:,:-1].float()
        return logits.log_softmax(-1).gather(-1,response.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad(): old=log_probs().detach()
    model.train();model.gradient_checkpointing_enable()
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-7)
    parameter=model.model.language_model.layers[-1].mlp.down_proj.weight
    before=parameter.detach().float().clone()
    new=log_probs()
    losses,clipfrac=compute_policy_loss(old-new,torch.ones_like(new),.2,.28)
    loss=losses.mean();loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    assert torch.isfinite(loss) and torch.isfinite(norm)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    change=(parameter.detach().float()-before).abs().max().item()
    # BF16 updates can round away at small LR; inspect all gradients/update stats.
    print('loss',loss.item(),'grad_norm',norm.item(),'max_parameter_change',change,flush=True)
    assert change>0, 'Sampled parameter did not change'
    model.save_pretrained(out/'hf-checkpoint',safe_serialization=True,max_shard_size='5GB')
    processor.save_pretrained(out/'hf-checkpoint')
    torch.save({'optimizer':optimizer.state_dict()},out/'optimizer.pt')
    report={'kind':'HF multimodal policy-loss smoke, not Megatron online RL','response':response_text,
        'loss':loss.item(),'grad_norm':norm.item(),'parameter_change':change,
        'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_seconds':time.monotonic()-started}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
