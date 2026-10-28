# v9: the host broker admits n>1 only on a route that declares max_choices (native_policy: n=20, the paper's
# Best-of-N sampling); every other route keeps exactly one choice. The observed-usage overrun guard scales the
# per-choice output cap by n (budget reservation and the journal audit already multiply by n).
# exec'd by build_v9.py with NEW = package root.
bp = NEW/'source/openwebrl/webprm_concurrent_broker.py'
bt = bp.read_text()
for a, b in (
        ("if (type(tokens) is not int or not 0<tokens<=route['max_output_tokens'] or payload.get('n',1)!=1 or",
         "choices=payload.get('n',1)  # v9: n>1 only where the route declares max_choices\n"
         "                        if (type(tokens) is not int or not 0<tokens<=route['max_output_tokens']\n"
         "                                or type(choices) is not int or not 1<=choices<=route.get('max_choices',1) or"),
        ("usage['prompt_tokens']>input_bound or usage['completion_tokens']>tokens):",
         "usage['prompt_tokens']>input_bound or usage['completion_tokens']>tokens*choices):"),
):
    assert bt.count(a) == 1, ('v9 broker anchor', a)
    bt = bt.replace(a, b)
bp.write_text(bt)
# v9b (2026-10-26): jobs 389731-389734 failed at broker startup: webprm_broker._validate_routes requires the exact key
# set {local, endpoint, max_input_tokens, max_output_tokens}, so the launcher's max_choices key was rejected
# ("Route must fix local mode, endpoint and output token cap"). Validate max_choices here (positive int, paid route
# only), then run the unchanged released validator on the route without it.
bt = bp.read_text()
a = "_deadline(deadline);_validate_cells(cells,deadline);_validate_routes(routes);_validate_prices(cells,routes)"
assert bt.count(a) == 1, ('v9b broker anchor', a)
bt = bt.replace(a, "_deadline(deadline);_validate_cells(cells,deadline);_validate_routes(_v9b_strip_choices(routes));_validate_prices(cells,routes)")
helper = '''

def _v9b_strip_choices(routes):
    """v9b: max_choices (n>1 sampling) is the only route key beyond the released contract."""
    if not isinstance(routes, dict):
        return routes
    out = {}
    for key, route in routes.items():
        if isinstance(route, dict) and 'max_choices' in route:
            n = route['max_choices']
            if type(n) is not int or not 1 <= n <= 20 or (n > 1 and route.get('local') is not False):
                raise ValueError('max_choices must be an int in 1..20 and only on a paid route')
            route = {k: v for k, v in route.items() if k != 'max_choices'}
        out[key] = route
    return out
'''
anchor = "\ndef serve_concurrent_broker("
assert bt.count(anchor) == 1, 'v9b serve anchor'
bt = bt.replace(anchor, helper + anchor)
bp.write_text(bt)
