#!/usr/bin/env python3
"""Read GPFS personal quota before GPU work; never delete files or obtain resources."""
import argparse
import getpass
import json
import re
import subprocess
from urllib.parse import unquote


def grace_seconds_lower_bound(value):
    """Conservatively interpret GPFS's displayed remaining grace duration.

    Unknown/expired values grant no extra capacity. Subtract one displayed unit
    to cover rounding; e.g. '7 days' grants at least six days, not exactly seven.
    """
    match = re.fullmatch(r'\s*(\d+)\s*(days?|hours?|minutes?|seconds?)\s*', value.lower())
    if not match:
        return 0
    unit = match[2].rstrip('s')
    multiplier = {'day':86400, 'hour':3600, 'minute':60, 'second':1}[unit]
    return max(0, int(match[1])-1)*multiplier


def parse_quota(text, username, fileset='scrubbed'):
    header = None
    for line in text.splitlines():
        fields = line.split(':')
        if fields[:3] == ['mmlsquota', 'user', 'HEADER']:
            header = fields
        elif header and fields[:2] == ['mmlsquota', 'user']:
            row = {k: unquote(v) for k, v in zip(header, fields)}
            if row.get('name') == username and row.get('filesetname') == fileset:
                values = {k: int(row[k]) * 1024 for k in
                          ['blockUsage','blockQuota','blockLimit','blockInDoubt']}
                ceilings = [values[k] for k in ['blockQuota','blockLimit'] if values[k]]
                if not ceilings:
                    raise ValueError('No finite quota reported; require explicit storage review')
                used = values['blockUsage'] + values['blockInDoubt']
                return dict(user=username, fileset=fileset, used_bytes=values['blockUsage'],
                            in_doubt_bytes=values['blockInDoubt'],
                            soft_limit_bytes=values['blockQuota'], hard_limit_bytes=values['blockLimit'],
                            block_grace=row.get('blockGrace','unknown'),
                            safe_headroom_bytes=max(0, min(ceilings)-used),
                            hard_headroom_bytes=max(0, values['blockLimit']-used))
    raise ValueError('Requested user/fileset quota missing')


def check_storage_quota(minimum_free_bytes=2*1024**4, minimum_grace_seconds=48*3600):
    user = getpass.getuser()
    text = subprocess.check_output(['/usr/lpp/mmfs/bin/mmlsquota', '-u', user, '-Y',
                                    'mmfs1:scrubbed'], text=True, stderr=subprocess.PIPE, timeout=30)
    report = parse_quota(text, user)
    grace = grace_seconds_lower_bound(report['block_grace'])
    # These launchers run at most24h. Requiring48h of remaining grace keeps a
    # full extra day of margin; every GPU worker rechecks this at startup.
    use_grace = (report['hard_limit_bytes'] > 0 and grace >= minimum_grace_seconds
                 and report['used_bytes'] >= report['soft_limit_bytes'] > 0)
    headroom = report['hard_headroom_bytes'] if use_grace else report['safe_headroom_bytes']
    report.update(minimum_free_bytes=minimum_free_bytes,
                  grace_seconds_lower_bound=grace,
                  minimum_grace_seconds=minimum_grace_seconds,
                  launch_quota_basis='hard_limit_with_active_grace' if use_grace else 'soft_limit',
                  launch_headroom_bytes=headroom,
                  passed=headroom >= minimum_free_bytes)
    if not report['passed']:
        raise OSError('Personal scrubbed quota headroom %.3f TiB is below required %.3f TiB; '
                      'basis=%s, grace=%s; filesystem free space is not personal quota.' %
                      (headroom/1024**4, minimum_free_bytes/1024**4,
                       report['launch_quota_basis'],report['block_grace']))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minimum-free-tib', type=float, default=2)
    args = parser.parse_args()
    print(json.dumps(check_storage_quota(int(args.minimum_free_tib*1024**4)),indent=2))
