#!/usr/bin/env python3
"""Run a bounded ARM/Sol diagnostic variant inside an existing user allocation."""
import argparse
import json

from run_arm_sol_gpu_diagnostic import training_plan
from run_arm_turn_bonus_cycles import execute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--provider',choices=['arm','sol'],required=True)
    parser.add_argument('--cycles',type=int,choices=[1,2],default=1)
    parser.add_argument('--attempt',type=int,required=True)
    parser.add_argument('--minutes',type=int,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.attempt<1: raise ValueError('Variants require a new numbered attempt directory')
    p=training_plan(args.job_id,args.minutes,interactive=True,attempt=args.attempt)
    p['requested_iterations']=args.cycles
    p['environment']['NUM_ROLLOUT']=str(args.cycles)
    p['command'][p['command'].index('--num-rollout')+1]=str(args.cycles)
    p['selector_provider']=args.provider
    if args.provider=='sol':
        p['arm_config'].update(selector_checkpoint='gpt-5.6-sol',selector_provider='openai-responses',
            reasoning_effort='medium')
    if args.execute: execute(p)
    else: print(json.dumps(p,indent=2))


if __name__=='__main__': main()
