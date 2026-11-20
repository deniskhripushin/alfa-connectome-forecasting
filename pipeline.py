"""Portable entry points for the selected Connectome solution."""
import argparse, hashlib, json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def contract(train_path):
    import numpy as np
    import pyarrow.parquet as pq
    import train_gru_mse_fold as base
    split=json.loads((ROOT/'configs/split.json').read_text())
    p=pq.ParquetFile(train_path)
    if p.num_row_groups!=10607 or p.metadata.num_rows!=10607*20000:
        raise ValueError('Expected the original train file: 10607 complete sequences')
    if any(p.metadata.row_group(i).num_rows!=20000 for i in range(p.num_row_groups)):
        raise ValueError('Each row group must be a complete sequence')
    actual=base.load_split(ROOT/'sequence_folds.json',0,0,0,20260928)
    if actual[0]!=split['train_groups'] or actual[1]!=split['holdout_groups']:
        raise ValueError('Original ordered fold partition changed')
    with np.load(ROOT/'configs/normalization.npz',allow_pickle=False) as f:
        mean,std=f['mean'].copy(),f['std'].copy()
    return split,mean,std,p

def train_gru(args):
    import torch
    import train_full
    split,mean,std,parquet=contract(args.train)
    recipe=json.loads((ROOT/'configs/gru_full.json').read_text())
    recipe['experiment']='portable_full10607_fixed4plus1_v1'
    recipe['files']={p.relative_to(ROOT).as_posix():sha(p) for p in
                     [ROOT/'pipeline.py',*sorted((ROOT/'src').glob('*.py')),ROOT/'configs/normalization.npz',ROOT/'configs/split.json']}
    stat=args.train.stat()
    recipe['train_file']={'path':str(args.train.resolve()),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    device=torch.device(args.device)
    output=args.output.resolve()
    base_path=output/'base4/epoch_04.pt'
    if args.stage in ('base','all'):
        train_full.run_stage(recipe,mean,std,parquet,'train',output/'base4',4,device)
    if args.stage in ('finetune','all'):
        train_full.run_stage(recipe,mean,std,parquet,'finetune',output/'finetune1',1,device,base_path)

def train_ae(args):
    import supervised_ae_mlp as ae
    split,mean,std,_=contract(ROOT/'datasets/train.parquet')
    def frozen_normalization(parquet,groups,max_sequences,seed):
        if groups!=split['train_groups'] or max_sequences!=256 or seed!=20260928:
            raise ValueError('Frozen AE normalization requires the original fold and seed')
        return mean.copy(),std.copy(),list(split['normalization_groups'])
    ae.compute_normalization=frozen_normalization
    settings=json.loads((ROOT/'configs/ae20.json').read_text())
    options=['--project',str(ROOT),'--run',str(args.output.resolve()),'--device',args.device]
    excluded={'project','run','evaluate','device'}
    for k,v in settings.items():
        if k in excluded: continue
        if isinstance(v,bool):
            if v: options.append('--'+k.replace('_','-'))
        else: options += ['--'+k.replace('_','-'),str(v)]
    sys.argv=['supervised_ae_mlp.py',*options]
    ae.main()

def check():
    import numpy as np
    import torch
    import train_gru_mse_fold as g
    import supervised_ae_mlp as a
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Release file changed: '+name)
    gs=torch.load(ROOT/'weights/gru_full.pt',map_location='cpu',weights_only=True)
    ae=torch.load(ROOT/'weights/ae20.pt',map_location='cpu',weights_only=True)
    with np.load(ROOT/'configs/normalization.npz') as f:
        mean,std=f['mean'],f['std']
        np.testing.assert_array_equal(gs['model_state_dict']['feature_mean'].numpy().reshape(-1),mean)
        np.testing.assert_array_equal(ae['mean'].numpy(),mean)
        np.testing.assert_array_equal(ae['std'].numpy(),std)
        gm=g.ConnectomeGRU(g.ModelConfig(**gs['model_config']),mean,std)
    gm.load_state_dict(gs['model_state_dict'],strict=True)
    am=a.SupervisedMLP(a.Config(**ae['config']))
    am.load_state_dict(ae['model_state_dict'],strict=True)
    split=json.loads((ROOT/'configs/split.json').read_text())
    assert len(split['train_groups'])==8485 and len(split['holdout_groups'])==2122
    assert sorted(split['train_groups']+split['holdout_groups'])==list(range(10607))
    assert not set(split['train_groups']) & set(split['holdout_groups'])
    assert set(split['normalization_groups']) <= set(split['train_groups'])
    assert gs['epoch']==5 and ae['epoch']==20
    print('PASS: release hashes, exact original archive, model tensors, normalization and split.')
    print('No training or data download was started.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('check')
    g=sub.add_parser('train-gru')
    g.add_argument('--train',type=Path,default=ROOT/'datasets/train.parquet')
    g.add_argument('--output',type=Path,default=ROOT/'runs/gru_full')
    g.add_argument('--stage',choices=['base','finetune','all'],default='all')
    g.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=sub.add_parser('train-ae')
    a.add_argument('--output',type=Path,default=ROOT/'runs/ae20')
    a.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args()
    if args.command.startswith('train-') and args.device=='cuda':
        from run_with_cuda_init import initialize_cuda_before_torch
        initialize_cuda_before_torch()
    if args.command=='check': check()
    elif args.command=='train-gru': train_gru(args)
    elif args.command=='train-ae': train_ae(args)

if __name__=='__main__': main()
