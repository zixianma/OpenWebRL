import math
import pytest
from openwebrl.arm_inference_flops import decoder_flops,vision_flops,shared_state_cost,match_expected_cost

TEXT=dict(hidden_size=8,num_attention_heads=3,head_dim=4,num_key_value_heads=1,
          intermediate_size=10,num_hidden_layers=2,vocab_size=20)


def test_decoder_matches_explicit_token_projection_and_attention_counts():
    # Unequal q/hidden dimensions and GQA catch an incorrect 12h^2 shortcut.
    weights=[(8,12),(8,4),(8,4),(12,8),(8,10),(8,10),(10,8)]
    projections=2*sum(a*b for a,b in weights)*2
    prompt,generated,cached=5,4,2
    queries=list(range(cached+1,prompt+1))+list(range(prompt+1,prompt+generated))
    explicit=projections*len(queries)+sum(4*2*12*q for q in queries)+generated*2*8*20
    out=decoder_flops(TEXT,prompt,generated,cached)
    assert out['total']==explicit
    assert out['total']==out['dense_linear']+out['attention']+out['vocabulary_head']
    assert out['total']==out['prefill']+out['decode']


def test_first_output_comes_from_prefill_and_cache_does_not_remove_attention_keys():
    cold=decoder_flops(TEXT,5,1)
    warm=decoder_flops(TEXT,5,1,4)
    assert cold['decode']==warm['decode']==0
    assert warm['attention']==4*2*12*5  # Last query still sees all five keys.
    assert cold['vocabulary_head']==warm['vocabulary_head']==2*8*20
    with pytest.raises(ValueError):decoder_flops(TEXT,5,1,5)


def test_vision_counts_every_deepstack_merger_and_frame_attention():
    c=dict(hidden_size=8,intermediate_size=12,depth=2,spatial_merge_size=2,
           in_channels=3,temporal_patch_size=2,patch_size=2,out_hidden_size=6,
           deepstack_visual_indexes=[0])
    result=vision_flops(c,[2,2,2]);n=8
    assert result['patch']==2*n*(3*2*2*2)*8
    assert result['attention']==2*4*8*(4**2+4**2)
    assert result['mergers']==2*2*2*(32*32+32*6)
    assert result['total']==sum(v for k,v in result.items() if k!='total')


def test_ordinary_repeats_and_arm_candidates_get_the_same_state_cache_rule():
    a=dict(actor_decode_flops=10,selector_flops=0,actor_prefill_states={'state':100},actor_vision_images={'image':20})
    assert shared_state_cost([a,a])==140
    guided=dict(a,actor_decode_flops=20,selector_flops=30)
    assert shared_state_cost([guided])==shared_state_cost([a,a])+30


def test_equal_mean_cost_mixture_and_no_extrapolation():
    result=match_expected_cost(25.,[10.,30.,50.],[.2,.5,.7],.6)
    assert result['upper_k_probability']==.75
    assert result['actor_success_at_equal_expected_cost']==pytest.approx(.425)
    assert result['arm_minus_actor']==pytest.approx(.175)
    assert not match_expected_cost(51.,[10.,30.,50.],[.2,.5,.7],.6)['covered']
    assert match_expected_cost(5.,[10.],[.2],.6)['actor_success_at_equal_expected_cost']==.1


def test_state_cache_applies_to_selector_too_but_never_shares_different_weights():
    episode=dict(actor_decode_flops=10,selector_flops=35,actor_prefill_states={'state':100},actor_vision_images={'image':20},
        selector_decode_flops=5,selector_prefill_states={'state':20},selector_vision_images={'image':10})
    # Even identical image/state keys identify separate actor/critic weights.
    assert shared_state_cost([episode,episode])==150+2*15
