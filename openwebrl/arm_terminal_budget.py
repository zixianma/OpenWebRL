"""Persisted spend cap around the unchanged native terminal-judge prompt/parser."""
import asyncio
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

from openwebrl.arm_rescue_yield import write_json


class CappedJudge:
    MAX_USD=30.
    MAX_CALLS=927  # 309 ARM trajectories, at most three explicit native attempts each.
    INPUT_USD_PER_MILLION=2.
    OUTPUT_USD_PER_MILLION=8.

    def __init__(self,root,client=None,*,max_calls=None,max_usd=None):
        if max_calls is not None:
            if type(max_calls) is not int or max_calls < 1: raise ValueError('Invalid judge call cap')
            self.MAX_CALLS=max_calls
        if max_usd is not None:
            if not 0 < max_usd <= 30: raise ValueError('Judge spend must remain within $30')
            self.MAX_USD=max_usd
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        if client is None:
            from openai import AsyncOpenAI
            base=os.getenv('JUDGE_API_BASE') or 'https://api.openai.com/v1'
            if base.rstrip('/')!='https://api.openai.com/v1':raise ValueError('Use the approved OpenAI endpoint')
            key=os.getenv('JUDGE_API_KEY') or os.getenv('OPENAI_API_KEY')
            if not key:raise ValueError('Missing terminal judge credential')
            client=AsyncOpenAI(api_key=key,base_url=base,max_retries=0,timeout=120)
        self.client=client;self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))

    @staticmethod
    def upper_cost(messages):
        text_bytes=0;images=0
        for message in messages:
            content=message.get('content','')
            if isinstance(content,str):text_bytes+=len(content.encode())
            else:
                for item in content:
                    if item.get('type')=='image_url':images+=1
                    else:text_bytes+=len(json.dumps(item).encode())
        return (text_bytes+images*20000+4096)*2/1e6 + 1024*8/1e6

    def reserve(self,upper):
        with (self.root/'owner.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            file=self.root/'ledger.json'
            state=json.loads(file.read_text()) if file.exists() else dict(calls=0,charged_or_reserved_usd=0.)
            if (self.root/'halt.json').exists() or state['calls']>=self.MAX_CALLS or state['charged_or_reserved_usd']+upper>self.MAX_USD:
                write_json(self.root/'halt.json',dict(reason='judge_budget_limit',state=state))
                raise ValueError('Terminal judge budget cap')
            state['calls']+=1;state['charged_or_reserved_usd']+=upper
            write_json(file,state);ident=state['calls']
            write_json(self.root/f'{ident:05d}.json',dict(status='reserved',charged_usd=upper))
            return ident

    def settle(self,ident,upper,usage):
        cost=(usage['prompt_tokens']*self.INPUT_USD_PER_MILLION+
              usage['completion_tokens']*self.OUTPUT_USD_PER_MILLION)/1e6
        with (self.root/'owner.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            file=self.root/'ledger.json';state=json.loads(file.read_text())
            state['charged_or_reserved_usd']+=cost-upper
            write_json(file,state)
            write_json(self.root/f'{ident:05d}.json',dict(status='received',charged_usd=cost,usage=usage))
            if cost>upper or state['charged_or_reserved_usd']>self.MAX_USD:
                write_json(self.root/'halt.json',dict(reason='unexpected_judge_cost',state=state))
                raise ValueError('Judge cost exceeded conservative reservation')

    async def create(self,**kwargs):
        if kwargs.get('model')!='gpt-4.1':raise ValueError('Only the priced native judge model is approved')
        upper=self.upper_cost(kwargs['messages']);ident=self.reserve(upper)
        try:
            reply=await self.client.chat.completions.create(**kwargs,max_completion_tokens=1024,store=False)
            self.settle(ident,upper,reply.usage.model_dump())
            return reply
        except BaseException:
            # An interrupted request retains its reservation; SDK retries are disabled.
            raise

    async def close(self):
        await self.client.close()
