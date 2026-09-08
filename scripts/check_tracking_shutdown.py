"""CPU-only offline W&B multiprocess shutdown check; preserves its output files."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import wandb
from slime.utils.logging_utils import finish_tracking



def worker(root, number):
    folder=root/f'worker{number}'
    folder.mkdir()
    wandb.init(project='openwebrl-offline-checks',id=f'shutdown{number}',mode='offline',
               dir=str(folder),settings=wandb.Settings(x_disable_stats=True))
    wandb.log({'probe/value':number})
    finish_tracking(SimpleNamespace(use_wandb=True))
    (folder/'closed.json').write_text(json.dumps({'finished':wandb.run is None}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    root=parser.parse_args().output
    root.mkdir(parents=True, exist_ok=False)
    folder=root/'primary';folder.mkdir()
    wandb.init(project='openwebrl-offline-checks',id='shutdown0',mode='offline',
               dir=str(folder),settings=wandb.Settings(x_disable_stats=True))
    ctx=mp.get_context('spawn')
    processes=[ctx.Process(target=worker,args=(root,i)) for i in [1,2]]
    for process in processes:process.start()
    for process in processes:
        process.join(40)
        assert process.exitcode==0,process.exitcode
    wandb.log({'probe/value':0})
    finish_tracking(SimpleNamespace(use_wandb=True))
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto.wandb_internal_pb2 import Record
    values=[]
    files=list(root.rglob('*.wandb'))
    assert len(files)==3,len(files)
    for path in files:
        ds=DataStore();ds.open_for_scan(str(path))
        while True:
            data=ds.scan_data()
            if data is None:break
            record=Record();record.ParseFromString(data)
            if record.HasField('history'):
                for item in record.history.item:
                    key=item.key or '.'.join(item.nested_key)
                    if key=='probe/value':values.append(json.loads(item.value_json))
        ds.close()
    assert sorted(values)==[0,1,2],values
    report={'offline_multiprocess_shutdown':'passed','worker_exit_codes':[p.exitcode for p in processes],
            'flushed_history_values':sorted(values),'wandb_files':len(files)}
    (root/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
