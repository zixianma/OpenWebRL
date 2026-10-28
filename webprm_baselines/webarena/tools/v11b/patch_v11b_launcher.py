# v11b (2026-10-27): v11 jobs 392753-392755 FAILED in switch_critic's local preflight: the 'Reply OK.' / max_tokens=8 probe
# made Piotr SelectionARM answer '</think>\n\nOK, the user just said' with finish_reason 'length', which
# webprm_worker.completion rejects for local routes. The Piotr preflight is now the selector-shaped payload that passed in
# reddit-table5-190-v3 (campaign/attempt-356527/local-preflight-api.jsonl, the one selector_piotr request: arbiter system
# prompt, 'Select Choice1' task, small screenshot, JSON-schema selection, max_tokens 128 -> stop '{"selection": 1}'),
# byte-frozen as openwebrl/webprm_piotr_preflight_v11b.json. exec'd by build_v11b.py after patch_v11_launcher.py (`t`).
V11B_PAIRS = [
    ("        payload=dict(model=model['served_model'],messages=[dict(role='user',content='Reply OK.')],temperature=0.,top_p=1.,max_tokens=8 if name in ('arbiter7b','piotr_selectionarm') else 2048)\n",
     "        payload=dict(model=model['served_model'],messages=[dict(role='user',content='Reply OK.')],temperature=0.,top_p=1.,max_tokens=8 if name in ('arbiter7b','piotr_selectionarm') else 2048)\n"
     "        if name=='piotr_selectionarm':payload=piotr_preflight_payload(model)  # v11b: selector-shaped probe (392753-392755)\n"),
]
for a, b in V11B_PAIRS:
    assert t.count(a) == 1, ('v11b launcher anchor', a[:120])
    t = t.replace(a, b)
V11B_HELPER = '''

# ---- v11b (2026-10-27): selector-shaped Piotr preflight (v11 jobs 392753-392755 failed on the 8-token probe). ----
V11B_PIOTR_PREFLIGHT = REPO/'openwebrl/webprm_piotr_preflight_v11b.json'
V11B_PIOTR_PREFLIGHT_SHA256 = 'a9b2ca972d4b223fbc1f0e358a1d7e24e375e5a16dcfd4f1ad120a891814c5ec'
V11B_PIOTR_PREFLIGHT_SOURCE = 'reddit-table5-190-v3/campaign/attempt-356527/local-preflight-api.jsonl (sha256 98aca61dc8831807501638ddd97d197feb846c98fa89c318609a8caef42439c7)'


def piotr_preflight_payload(model):
    """The byte-frozen selector-shaped Piotr probe (stop with {"selection": 1} in reddit-table5-190-v3)."""
    data = V11B_PIOTR_PREFLIGHT.read_bytes()
    if hashlib.sha256(data).hexdigest() != V11B_PIOTR_PREFLIGHT_SHA256:
        raise ValueError('Frozen Piotr preflight payload changed')
    payload = json.loads(data)
    if payload.get('model') != model['served_model'] or payload.get('max_tokens') != 128 or payload.get('temperature') != 0:
        raise ValueError('Piotr preflight payload differs from the served model / 128-token selector contract')
    return payload
'''
anchor = "\nif __name__=='__main__':main()"
assert t.count(anchor) == 1
t = t.replace(anchor, V11B_HELPER + anchor)
