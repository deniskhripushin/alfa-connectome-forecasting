"""Check callback parity, benchmark one logical CPU, or score labeled validation."""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import numpy as np

def runtime(directory,native=True):
    path=directory/('solution.py' if native else 'base_runtime.py')
    spec=importlib.util.spec_from_file_location('_release_'+str(native),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.PredictionModel(directory)

def pin_cpu():
    if hasattr(os,'sched_getaffinity'):
        allowed=os.sched_getaffinity(0);cpu=min(allowed);os.sched_setaffinity(0,{cpu});return cpu
    if os.name=='nt':
        import ctypes
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.GetCurrentProcess.restype=ctypes.c_void_p
        api.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
        api.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
        process=api.GetCurrentProcess();allowed=ctypes.c_size_t();system=ctypes.c_size_t()
        if not api.GetProcessAffinityMask(process,ctypes.byref(allowed),ctypes.byref(system)): raise ctypes.WinError(ctypes.get_last_error())
        mask=allowed.value & -allowed.value
        if not api.SetProcessAffinityMask(process,mask): raise ctypes.WinError(ctypes.get_last_error())
        return mask.bit_length()-1
    raise RuntimeError('CPU affinity is not supported on this platform')

def samples(args):
    if args.data:
        import pyarrow.parquet as pq
        from utils import FEATURE_COLUMNS,validate_sequence
        p=pq.ParquetFile(args.data)
        result=[]
        for i in args.groups:
            t=p.read_row_group(i,columns=['seq_ix','step_in_seq','need_prediction',*FEATURE_COLUMNS],use_threads=False)
            seq,need=validate_sequence(t)
            x=np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
            result.append((seq,x,need))
        return result
    with np.load(args.source/'normalization.npz') as f: mean,std=f['mean'],f['std']
    rng=np.random.default_rng(20260928)
    result=[]
    for seq in [1,2]:
        z=rng.normal(size=(1024,112)).astype(np.float32)
        z[19,0]=25;z[20,1]=-25;z[98:101,2]=0
        result.append((seq,mean+std*z,np.arange(len(z))>=99))
    return result

def verify(args):
    import torch
    from train_gru_mse_fold import ConnectomeGRU,ModelConfig
    from supervised_ae_mlp import SupervisedMLP,Config
    from mlp_experiment import make_features
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    gs=torch.load(args.gru,map_location='cpu',weights_only=True)
    ae=torch.load(args.ae,map_location='cpu',weights_only=True)
    with np.load(args.source/'normalization.npz') as f: mean,std=f['mean'],f['std']
    np.testing.assert_array_equal(mean,gs['model_state_dict']['feature_mean'].numpy().reshape(-1))
    np.testing.assert_array_equal(std,gs['model_state_dict']['feature_std'].numpy().reshape(-1))
    np.testing.assert_array_equal(mean,ae['mean'].numpy());np.testing.assert_array_equal(std,ae['std'].numpy())
    gru=ConnectomeGRU(ModelConfig(**gs['model_config']),mean,std).eval()
    gru.load_state_dict(gs['model_state_dict'],strict=True)
    mlp=SupervisedMLP(Config(**ae['config'])).eval();mlp.load_state_dict(ae['model_state_dict'],strict=True)
    native,python=runtime(args.source),runtime(args.source,False)
    data=samples(args);data.append((data[0][0],data[0][1][:256],data[0][2][:256]))
    errors={'prediction':0.,'hidden':0.,'history':0.};count=0
    for seq,x,need in data:
        features=make_features(x,mean,std,'history')
        predictions=[]
        for step,row in enumerate(x):
            point=SimpleNamespace(seq_ix=seq,step_in_seq=step,need_prediction=bool(need[step]),state=row)
            a,b=native.predict(point),python.predict(point)
            if need[step]: np.testing.assert_array_equal(a,b)
            elif a is not None or b is not None: raise AssertionError('Warmup must return None')
            np.testing.assert_array_equal(native.hidden,python.hidden)
            np.testing.assert_array_equal(native.engineered,python.engineered)
            np.testing.assert_array_equal(native.ema,python.ema)
            predictions.append(native.prediction.copy())
            errors['history']=max(errors['history'],float(np.max(np.abs(native.engineered-features[step]))))
            np.testing.assert_allclose(native.engineered,features[step],atol=2e-6,rtol=2e-6)
        with torch.inference_mode():
            h=None;gp=[]
            for start in range(0,len(x),512):
                pred,h=gru(torch.from_numpy(x[start:start+512])[None],h);gp.append(pred[0].numpy())
            ap=mlp(torch.from_numpy(features)).numpy()
            reference=np.concatenate(gp)*np.array([.7,.65],np.float32)+ap*np.array([.3,.35],np.float32)
        predictions=np.asarray(predictions)
        errors['prediction']=max(errors['prediction'],float(np.max(np.abs(reference-predictions))))
        errors['hidden']=max(errors['hidden'],float(np.max(np.abs(h.numpy()-native.hidden))))
        np.testing.assert_allclose(predictions,reference,atol=1e-5,rtol=1e-5)
        np.testing.assert_allclose(native.hidden,h.numpy(),atol=1e-5,rtol=1e-5)
        count+=len(x)
    for point in [SimpleNamespace(seq_ix=999999,step_in_seq=1,need_prediction=True,state=x[0]),
                  SimpleNamespace(seq_ix=data[-1][0],step_in_seq=999,need_prediction=True,state=x[0])]:
        try: native.predict(point)
        except ValueError: pass
        else: raise AssertionError('Invalid sequence order was accepted')
    return dict(passed=True,callbacks=count,exact_native_vs_numpy=True,max_abs_error_vs_pytorch=errors,
                data='real complete validation sequences + reset replay' if args.data else 'synthetic + reset replay',
                source=str(args.source),note='Sampled parity; not a rerun of full official validation.')

def benchmark(args):
    cpu=pin_cpu();data=samples(args)
    # Remove file IO and model startup from steady-state callback measurement.
    timings=[];rows=sum(len(x) for _,x,_ in data)
    for repeat in range(args.repeats):
        model=runtime(args.source)
        for step,row in enumerate(data[0][1][:256]):
            model.predict(SimpleNamespace(seq_ix=-12345,step_in_seq=step,need_prediction=step>=99,state=row))
        points=[SimpleNamespace(seq_ix=seq,step_in_seq=i,need_prediction=bool(need[i]),state=row)
                for seq,x,need in data for i,row in enumerate(x)]
        start=time.perf_counter()
        for point in points: model.predict(point)
        elapsed=time.perf_counter()-start;timings.append(elapsed)
        print(f'Repeat {repeat+1}: {elapsed:.3f}s / {rows} callbacks',flush=True)
    return dict(logical_cpu=cpu,threads=1,rows=rows,repeats=args.repeats,seconds=timings,
                median_microseconds_per_row=float(np.median(timings))/rows*1e6,
                includes='callback, history, ONNX and return copy; excludes IO/startup/point creation',
                server_runtime_historical_seconds=3281,
                note='Local timings do not establish runtime on another CPU or dataset.')

def evaluate(args):
    if not args.data: raise ValueError('--data must point to labeled official valid.parquet')
    import pyarrow.parquet as pq
    from utils import FEATURE_COLUMNS,TARGET_COLUMNS,validate_sequence,GlobalAccumulator
    pin_cpu();p=pq.ParquetFile(args.data);score=GlobalAccumulator();model=runtime(args.source)
    started=time.perf_counter()
    for group in range(p.num_row_groups):
        t=p.read_row_group(group,columns=['seq_ix','step_in_seq','need_prediction','is_scored',*FEATURE_COLUMNS,*TARGET_COLUMNS],use_threads=False)
        seq,need=validate_sequence(t)
        x=np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        y=np.column_stack([t[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        prediction=np.zeros_like(y)
        for i,row in enumerate(x):
            value=model.predict(SimpleNamespace(seq_ix=seq,step_in_seq=i,need_prediction=bool(need[i]),state=row))
            if value is not None: prediction[i]=value
        mask=need&t['is_scored'].to_numpy().astype(bool)
        score.add(y,prediction,mask)
        if group%50==0: print(f'{group+1}/{p.num_row_groups}',flush=True)
    return dict(**score.result(),seconds=time.perf_counter()-started)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['verify','benchmark','evaluate'])
    p.add_argument('--source',type=Path,default=ROOT/'submission')
    p.add_argument('--data',type=Path)
    p.add_argument('--groups',type=int,nargs='+',default=[0,936,1872])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--gru',type=Path,default=ROOT/'weights/gru_full.pt')
    p.add_argument('--ae',type=Path,default=ROOT/'weights/ae20.pt')
    p.add_argument('--report',type=Path)
    args=p.parse_args()
    if args.repeats<1: p.error('repeats must be positive')
    result={'verify':verify,'benchmark':benchmark,'evaluate':evaluate}[args.mode](args)
    print(json.dumps(result,indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__': main()
