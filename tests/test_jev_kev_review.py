"""Target labels must come from the model's state, never the resulting page."""
from scripts.render_jev_kev_review import target_mapping


def decision():
    return dict(operation='CLICK', target='3', choice='e3', fingerprint='before',
        request={'questions': {'click_target': {'criteria': {
            '3': {'element': '[3] Women', 'role': 'link'}}}}})


def observation():
    return dict(actions=[dict(id='e3', node=17, rect=dict(x=10, y=20, w=30, h=40))],
        image='saved-image', w=1120, h=780)


def test_exact_state_resolves_label_and_observed_geometry():
    result = target_mapping(decision(), {'before': observation()})
    assert result['label'] == '[3] Women'
    assert result['cached_dom_node'] == 17
    assert result['before_screenshot'] == 'saved-image'
    assert result['observed_rect'] == dict(x=10, y=20, w=30, h=40)
    assert result['actual_execution_coordinates_recorded'] is False


def test_post_action_index_collision_cannot_supply_geometry():
    result = target_mapping(decision(), {'after': observation()})
    assert result['label'] == '[3] Women'
    assert 'observed_rect' not in result
    assert 'before_screenshot' not in result


def test_no_target_and_missing_action_do_not_invent_coordinates():
    assert target_mapping(None, {}) is None
    assert target_mapping({'target': None}, {}) is None
    page = observation()
    page['actions'][0]['id'] = 'e4'
    assert 'observed_rect' not in target_mapping(decision(), {'before': page})
