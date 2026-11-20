"""Optional real-data CUDA smoke: tiny GRU resume/fine-tune and one AE update."""
import argparse, json, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pyarrow.parquet as pq
import torch
from types import SimpleNamespace
import train_full as full
import supervised_ae_mlp as ae

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--report',type=Path)
    args=p.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this smoke test')
    device=torch.device('cuda');parquet=pq.ParquetFile(args.data)
    split=json.loads((ROOT/'configs/split.json').read_text())
    groups=split['train_groups'][:2]
    with np.load(ROOT/'configs/normalization.npz') as s: mean,std=s['mean'],s['std']
    recipe=json.loads((ROOT/'configs/gru_full.json').read_text())
    recipe['train_groups']=groups
    recipe['experiment']='smoke_only_two_train_sequences'
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        full.run_stage(recipe,mean,std,parquet,'train',folder/'base',1,device)
        path=full.run_stage(recipe,mean,std,parquet,'train',folder/'base',3,device)
        result=full.run_stage(recipe,mean,std,parquet,'finetune',folder/'fine',1,device,path)
        saved=torch.load(result,map_location='cpu',weights_only=True)
        full.assert_checkpoint(saved,recipe,'finetune',mean,std,complete=True)
        if saved['ema_updates']<1: raise AssertionError('No optimizer updates')
        digest=full.sha256(result)
        full.run_stage(recipe,mean,std,parquet,'finetune',folder/'fine',1,device,path)
        assert full.sha256(result)==digest
    full.base.seed_everything(20260928)
    model=ae.SupervisedMLP(ae.Config(mode='history',auxiliary_source='decoder')).to(device)
    before={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    settings=SimpleNamespace(seed=20260928,rows_per_sequence=512,batch_sequences=8,mode='history',
                             reconstruction_weight=.1,auxiliary_weight=.25,correlation_weight=0.)
    losses=ae.train_epoch(model,optimizer,parquet,groups,mean,std,settings,1,device)
    for prefix in ('encoder.','decoder.','head.','auxiliary.'):
        assert any(not torch.equal(before[n],v.cpu()) for n,v in model.state_dict().items() if n.startswith(prefix))
    assert all(np.isfinite(v) for v in losses.values())
    report=dict(passed=True,train_sequences=2,gru_base_epochs=4,gru_finetune_epochs=1,
                checkpoint_resume_and_completed_noop=True,ae_optimizer_updates=1,
                ae_all_four_parameter_groups_changed=True,ae_losses=losses,
                note='Pipeline smoke on two training sequences, not full retraining; no release weights modified.')
    print(json.dumps(report,indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
