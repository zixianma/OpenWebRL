"""Cross-job cohort invariants; no browser, model or scheduler access."""
from collections import Counter
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from prepare_controlled_sft_suite import partition_tasks, schedule


def test_parallel_shards_keep_every_task_paired_across_all_arms():
    ids=[f'task-{i}' for i in range(300)]
    shards=partition_tasks(ids)
    assert shards==partition_tasks(ids)
    assert list(map(len,shards))==[150,150]
    assert not set(shards[0]) & set(shards[1])
    assert set(shards[0]+shards[1])==set(ids)
    primary=[];smokes=[]
    for shard in shards:
        rows=schedule(shard)
        assert len(rows)==612
        primary.extend(r for r in rows if r['category']=='primary')
        smokes.extend(r for r in rows if r['category']=='smoke')
        for task in shard:
            assert {r['condition'] for r in rows if r['task_id']==task and r['category']=='primary'}=={'L01','L08','L10','L11'}
    assert len({r['item_id'] for r in primary+smokes})==1224
    assert Counter(r['condition'] for r in primary)==dict.fromkeys(['L01','L08','L10','L11'],300)
    assert Counter(r['condition'] for r in smokes)==dict.fromkeys(['L01','L08','L10','L11'],6)


@pytest.mark.parametrize('ids,count', [([str(i) for i in range(299)],2),(['duplicate']*300,2),([str(i) for i in range(300)],3)])
def test_partition_refuses_missing_duplicated_or_extra_shards(ids,count):
    with pytest.raises(ValueError):partition_tasks(ids,count=count)
