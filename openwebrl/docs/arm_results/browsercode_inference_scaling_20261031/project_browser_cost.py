"""Reproduce browser-cost scenarios from public aggregate session-minute counts.

No browser, model, provider API or network access. These are projections from
local browser lifetimes, not measured cloud charges. Traffic is an assumption.
"""
import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path


def number(value):
    if isinstance(value, bool):
        raise ValueError('Boolean rate or traffic volume')
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError('Rates and traffic must be finite and nonnegative')
    return result


def count(value):
    if type(value) is not int or value < 0:
        raise ValueError('Counts must be nonnegative integers')
    return value


def project(data, traffic_mb):
    if data['schema'] != 'browsercode-browser-cost-projection-v1' or data['actual_cloud_billing'] is not False:
        raise ValueError('Expected explicitly projected browser cost')
    price = data['price_assumption']
    hourly = number(price['browser_usd_per_hour'])
    direct = number(price['traffic_usd_per_gb']['direct'])
    residential = number(price['traffic_usd_per_gb']['residential'])
    volumes = [number(value) for value in traffic_mb]
    if not volumes or len(volumes) != len(set(volumes)):
        raise ValueError('Unique traffic scenarios required')
    methods = {'serial_high', 'luna_single_high', 'parallel3_select_high', 'parallel5_select_high',
               'luna_single_xhigh_12288', 'luna_single_high_24576', 'luna_single_high_6144'}
    expected = {(benchmark, method) for benchmark in ('online_mind2web', 'webvoyager', 'deepshop') for method in methods}
    identities = [(point['benchmark'], point['method']) for point in data['points']]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError('Exactly 21 distinct benchmark/method points required')
    rows = []
    for point in data['points']:
        valid, invalid, requested = [count(point[key]) for key in ('valid', 'invalid', 'requested')]
        if valid + invalid != requested:
            raise ValueError('Task partition differs')
        scope = point['valid_task_lifetime_projection']
        observed = count(scope['first_close_lifetime_observed_tasks'])
        missing = count(scope['first_close_lifetime_missing_tasks'])
        minutes = count(scope['per_session_rounded_minutes_sum_observed'])
        if observed != valid or missing != 0 or minutes < observed or not observed:
            raise ValueError('Complete valid-task first-close lifetimes required')
        rental = Decimal(minutes) * hourly / 60 / observed
        for mb in volumes:
            # The provider's billing example converts reported MB to GB /1024.
            direct_network = mb / 1024 * direct
            residential_network = mb / 1024 * residential
            rows.append(dict(benchmark=point['benchmark'], method=point['method'],
                requested=requested, valid=valid, invalid=invalid,
                lifetime_observed_valid_tasks=observed,
                assumed_traffic_mb_per_task=str(mb),
                projected_hosting_usd_per_valid_task=str(rental),
                projected_direct_network_usd_per_task=str(direct_network),
                projected_residential_network_usd_per_task=str(residential_network),
                projected_hosting_plus_direct_usd_per_task=str(rental + direct_network),
                projected_hosting_plus_residential_usd_per_task=str(rental + residential_network)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('browser_cost_projection.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--traffic-mb', nargs='+', default=['0', '10', '50', '100'])
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path('/gpfs/home/zixianma')):
        raise ValueError('Home is read-only')
    rows = project(json.loads(args.input.read_text()), args.traffic_mb)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'scenario_rows': len(rows), 'actual_cloud_billing': False, 'api_calls': 0}))


if __name__ == '__main__':
    main()
