#!/usr/bin/env python3
"""Small CPU import/operator check before spending the refresh GPU budget."""
import importlib.metadata as metadata
import json,os,subprocess
from pathlib import Path

def main():
    import torch,torchvision,torchaudio
    import peft,trl,xgrammar
    from transformers import Qwen3VLForConditionalGeneration
    from llamafactory.model.loader import load_model
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    root=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction')
    headers=[root.parent/'src/python-headers/Include',root.parent/'src/python-headers']
    expected_cpath=':'.join(map(str,headers))
    if os.environ.get('CPATH')!=expected_cpath:raise ValueError('Refresh launcher must export validated Python headers via CPATH')
    source=root/'refresh-torch28/triton/backends/nvidia/driver.c'
    check=subprocess.run(['cc','-fsyntax-only','-I'+str(source.parent/'include'),str(source)],capture_output=True,text=True)
    if check.returncode:raise RuntimeError('Triton Python/CUDA header check failed: '+check.stderr[-2000:])
    assert torch.__version__=='2.8.0+cu128',torch.__version__
    assert torchvision.__version__=='0.23.0+cu128',torchvision.__version__
    assert torchaudio.__version__=='2.8.0+cu128',torchaudio.__version__
    assert Path(torch.__file__).is_relative_to(root/'refresh-torch28')
    torch.set_num_threads(1)
    layer=torch.nn.Conv3d(3,4,kernel_size=(2,2,2))
    output=layer(torch.ones(1,3,2,4,4));output.square().mean().backward()
    assert torch.isfinite(layer.weight.grad).all()
    kept=torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]),torch.ones(1),.5)
    assert kept.tolist()==[0]
    value=dict(passed=True,validation='CPU imports and tiny Conv3d backward/NMS; GPU validation remains mandatory',
        packages={p:metadata.version(p) for p in ('torch','torchvision','torchaudio','triton','transformers','peft','trl','llamafactory')},
        torch_path=torch.__file__,torchvision_path=torchvision.__file__,cuda_build=torch.version.cuda,
        triton_driver_compiler_check=True,python_headers=list(map(str,headers)))
    dest=root.parent/'arm-turn-bonus-preparation/arm-refresh-20260924/baseline40/environment-readiness.json'
    dest.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))

if __name__=='__main__':main()
