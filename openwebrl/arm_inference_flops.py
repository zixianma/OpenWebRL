"""Auditable Qwen3-VL forward-work estimates, not hardware FLOP counters.

One multiply-add is two FLOPs. Count dense projections, attention QK/AV,
vision patch projection and all final/DeepStack mergers. Elementwise kernels,
softmax/normalization, padding, recomputation and CPU work are not counted.
"""
import base64
import hashlib
import math
import struct


def decoder_flops(config, prompt, generated, cached=0):
    if not (isinstance(prompt,int) and isinstance(generated,int) and isinstance(cached,int)):
        raise ValueError('Integer sequence lengths required')
    if prompt<1 or generated<1 or not 0<=cached<prompt:
        raise ValueError('Need a received generation with at least one uncached input token')
    h=config['hidden_size'];q=config['num_attention_heads']*config['head_dim']
    kv=config['num_key_value_heads']*config['head_dim'];ff=config['intermediate_size'];layers=config['num_hidden_layers']
    linear_per_token=layers*2*(h*q+2*h*kv+q*h+3*h*ff)
    head_per_token=2*h*config['vocab_size']
    # Autoregressive generation predicts token1 from the prefill's final state;
    # only tokens1..generated-1 require subsequent decoder forwards.
    d=generated-1;u=prompt-cached
    prefill_pairs=(prompt*(prompt+1)-cached*(cached+1))//2
    decode_pairs=d*prompt+d*(d+1)//2
    prefill_linear=u*linear_per_token
    decode_linear=d*linear_per_token
    prefill_attention=4*layers*q*prefill_pairs
    decode_attention=4*layers*q*decode_pairs
    prefill=prefill_linear+prefill_attention+head_per_token
    decode=decode_linear+decode_attention+d*head_per_token
    return dict(prefill=prefill,decode=decode,total=prefill+decode,
        dense_linear=prefill_linear+decode_linear,attention=prefill_attention+decode_attention,
        vocabulary_head=generated*head_per_token)


def vision_flops(config,grid):
    t,h,w=map(int,grid)
    if min(t,h,w)<1:raise ValueError('Positive visual grid required')
    n=t*h*w;d=config['hidden_size'];ff=config['intermediate_size'];depth=config['depth']
    merge=config['spatial_merge_size']**2
    if n%merge:raise ValueError('Visual grid must be spatially mergeable')
    patch=2*n*config['in_channels']*config['temporal_patch_size']*config['patch_size']**2*d
    linear=n*depth*(8*d*d+4*d*ff)  # QKV+output; two-layer GELU vision MLP.
    attention=4*depth*d*t*(h*w)**2  # Full, noncausal attention within each image/frame.
    merged_dim=d*merge
    mergers=(1+len(config['deepstack_visual_indexes']))*(n//merge)*2*(merged_dim**2+merged_dim*config['out_hidden_size'])
    return dict(patch=patch,linear=linear,attention=attention,mergers=mergers,total=patch+linear+attention+mergers)


def image_geometry(data,*,max_pixels,min_pixels=65536,patch=16,merge=2):
    if not isinstance(data,str):raise ValueError('Expected saved image data URL')
    raw=base64.b64decode(data.split(',',1)[-1])
    if raw[:8]==b'\x89PNG\r\n\x1a\n':width,height=struct.unpack('>II',raw[16:24])
    else:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as image:width,height=image.size
    # Same smart_resize rule as the installed Qwen2VLImageProcessorFast used
    # by both checkpoints, with each service's actual max_pixels setting.
    factor=patch*merge
    if min(height,width)<factor or max(height,width)/min(height,width)>200:
        raise ValueError('Image outside supported processor geometry')
    rh=round(height/factor)*factor;rw=round(width/factor)*factor
    if rh*rw>max_pixels:
        beta=math.sqrt(height*width/max_pixels)
        rh=math.floor(height/beta/factor)*factor;rw=math.floor(width/beta/factor)*factor
    elif rh*rw<min_pixels:
        beta=math.sqrt(min_pixels/(height*width))
        rh=math.ceil(height*beta/factor)*factor;rw=math.ceil(width*beta/factor)*factor
    return dict(grid=[1,rh//patch,rw//patch],sha256=hashlib.sha256(raw).hexdigest(),width=width,height=height)


def shared_state_cost(episodes):
    """Identical full-state prefill/image sharing within one policy/task only.

    Both the guided candidate set and ordinary episode subsets receive this
    identical idealized cache policy. No partial-prefix reuse is assumed.
    """
    states={};images={};decode=selector=0
    for episode in episodes:
        decode+=episode['actor_decode_flops']
        models=['actor']
        if 'selector_decode_flops' in episode:
            decode+=episode['selector_decode_flops'];models.append('selector')
        else:selector+=episode['selector_flops']
        for model in models:
            # Different model weights never share cached activations.
            for key,value in episode[model+'_prefill_states'].items():
                key=(model,key)
                if key in states and states[key]!=value:raise ValueError('Same state has different prefill geometry')
                states[key]=value
            for key,value in episode[model+'_vision_images'].items():
                key=(model,key)
                if key in images and images[key]!=value:raise ValueError('Same image has different geometry')
                images[key]=value
    return decode+selector+sum(states.values())+sum(images.values())


def match_expected_cost(arm_cost, actor_costs, actor_success, arm_success):
    """Outcome-independent mixture of adjacent k, with no extrapolation.

    A task is drawn from the benchmark distribution, then k is randomized
    independently of that task's outcome. This matches mean cost, not a hard
    budget for every task. k=0 means skip the task and count it as unsuccessful.
    """
    if len(actor_costs)!=len(actor_success) or not actor_costs:
        raise ValueError('Aligned nonempty cost/performance curve required')
    if arm_cost<0 or any(b<a for a,b in zip([0.,*actor_costs],actor_costs)):
        raise ValueError('Nonnegative monotone cost curve required')
    if arm_cost>actor_costs[-1]:
        return dict(covered=False,arm_cost=arm_cost,actor5_cost=actor_costs[-1],
                    reason='Guided cost exceeds measured curve; no extrapolation')
    costs=[0.,*actor_costs];scores=[0.,*actor_success]
    for i,(a,b) in enumerate(zip(costs,costs[1:])):
        if a<=arm_cost<=b and b>a:
            weight=(arm_cost-a)/(b-a)
            baseline=(1-weight)*scores[i]+weight*scores[i+1]
            return dict(covered=True,arm_cost=arm_cost,lower_k=i,upper_k=i+1,
                upper_k_probability=weight,actor_success_at_equal_expected_cost=baseline,
                arm_minus_actor=arm_success-baseline)
    # Entire curve is free: choose the largest k without discarding benefit.
    if arm_cost==0 and actor_costs[-1]==0:
        return dict(covered=True,arm_cost=0.,lower_k=len(actor_costs),upper_k=len(actor_costs),
                    upper_k_probability=1.,actor_success_at_equal_expected_cost=actor_success[-1],
                    arm_minus_actor=arm_success-actor_success[-1])
    raise ValueError('Uncovered cost bracket')
