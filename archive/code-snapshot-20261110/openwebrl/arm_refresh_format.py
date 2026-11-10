"""Exact deployed SelectionARM text and image-token prefix, CPU only."""
import hashlib, io, json, math
from pathlib import Path
from openwebrl.arm_inference import load_selection_builder, selection_messages

BASE=Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
UPSTREAM=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/source')

class SelectionFormatter:
    def __init__(self):
        from tokenizers import Tokenizer
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        self.tokenizer=Tokenizer.from_file(str(BASE/'tokenizer.json'))
        env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
        def fail(message):raise ValueError(message)
        env.globals['raise_exception']=fail
        self.template=env.from_string((BASE/'chat_template.jinja').read_text())
        self.builder=load_selection_builder(UPSTREAM)
        self.image_token=self.tokenizer.token_to_id('<|image_pad|>')

    def prepare(self,row,order):
        from PIL import Image
        raw=Path(row['screenshot']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['screenshot_sha256']:raise ValueError('Screenshot changed')
        messages=selection_messages(self.builder,row['task'],row['url'],row['history'],
                                   [row['candidates'][i] for i in order],raw)
        messages[1]['content'][1]={'type':'image'}
        prompt=self.template.render(messages=messages,add_generation_prompt=True,tools=None)
        with Image.open(io.BytesIO(raw)) as im:w,h=im.size
        factor=32;hh=max(factor,round(h/factor)*factor);ww=max(factor,round(w/factor)*factor)
        if hh*ww>262144:
            scale=math.sqrt(h*w/262144);hh=math.floor(h/scale/factor)*factor;ww=math.floor(w/scale/factor)*factor
        elif hh*ww<65536:
            scale=math.sqrt(65536/(h*w));hh=math.ceil(h*scale/factor)*factor;ww=math.ceil(w*scale/factor)*factor
        ids=self.tokenizer.encode(prompt,add_special_tokens=False).ids
        if ids.count(self.image_token)!=1:raise ValueError('Expected one current screenshot')
        grid=[1,hh//16,ww//16];expanded=math.prod(grid)//4
        prefix=[x for token in ids for x in ([token]*expanded if token==self.image_token else [token])]
        if len(prefix)+32>8192:raise ValueError('Selection context exceeds8192 without truncation')
        return dict(prompt=prompt,prompt_tokens=len(prefix),image_size=[w,h],image_grid_thw=[grid],
                    prompt_token_sha256=hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest(),
                    permutation=order)
