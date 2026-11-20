"""Install trained tensors into the shipped, verified 21-node ONNX topology."""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper,helper
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def export(gru_path,ae_path,output):
    from train_gru_mse_fold import ModelConfig,ConnectomeGRU
    from supervised_ae_mlp import Config,SupervisedMLP
    if output.exists(): raise FileExistsError('Choose a new export directory')
    gs=torch.load(gru_path,map_location='cpu',weights_only=True)
    ae=torch.load(ae_path,map_location='cpu',weights_only=True)
    mean=gs['model_state_dict']['feature_mean'].numpy().reshape(-1)
    std=gs['model_state_dict']['feature_std'].numpy().reshape(-1)
    np.testing.assert_array_equal(mean,ae['mean'].numpy())
    np.testing.assert_array_equal(std,ae['std'].numpy())
    if gs['model_config']!=vars(ModelConfig()): raise ValueError('GRU topology differs')
    expected=vars(Config(mode='history',auxiliary_source='decoder'))
    if ae['config']!=expected: raise ValueError('AE topology differs')
    ConnectomeGRU(ModelConfig(**gs['model_config']),mean,std).load_state_dict(gs['model_state_dict'],strict=True)
    SupervisedMLP(Config(**ae['config'])).load_state_dict(ae['model_state_dict'],strict=True)
    graph=onnx.load(ROOT/'submission/model.onnx')
    nodes={n.name:n for n in graph.graph.node}
    initializers={v.name:v for v in graph.graph.initializer}
    def array(t): return t.detach().cpu().numpy()
    def gates(key):
        r,z,n=np.split(array(gs['model_state_dict'][key]),3,axis=0)
        return np.concatenate((z,r,n),axis=0)
    replacements={}
    for layer,name in enumerate(('/new/gru/GRU','/new/gru/GRU_1')):
        node=nodes[name]
        attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
        assert attrs['linear_before_reset']==1 and attrs['hidden_size']==128
        replacements[node.input[1]]=gates(f'gru.weight_ih_l{layer}')[None]
        replacements[node.input[2]]=gates(f'gru.weight_hh_l{layer}')[None]
        replacements[node.input[3]]=np.concatenate((gates(f'gru.bias_ih_l{layer}'),gates(f'gru.bias_hh_l{layer}')))[None]
    replacements[nodes['/new/head/MatMul'].input[1]]=array(gs['model_state_dict']['head.weight']).T.copy()
    replacements['new.head.bias']=array(gs['model_state_dict']['head.bias'])
    for key,tensor in ae['model_state_dict'].items():
        if key.startswith(('encoder.','head.')): replacements['ae.'+key]=array(tensor)
    assert len(replacements)==18
    for name,value in replacements.items():
        if value.dtype!=np.float32 or not np.isfinite(value).all(): raise ValueError(name)
        if list(value.shape)!=list(initializers[name].dims): raise ValueError('Shape mismatch '+name)
        initializers[name].CopyFrom(numpy_helper.from_array(value,name))
    for item in graph.metadata_props:
        if item.key=='gru_epoch': item.value=str(gs['epoch'])
        elif item.key=='ae_epoch': item.value=str(ae['epoch'])
    onnx.checker.check_model(graph,full_check=True)
    output.mkdir(parents=True)
    onnx.save(graph,output/'model.onnx')
    np.savez(output/'normalization.npz',mean=mean,std=std)
    for name in ('solution.py','base_runtime.py','history.c','history_windows.dll','history_linux.so'):
        shutil.copyfile(ROOT/'submission'/name,output/name)
    receipt={'gru_checkpoint':str(gru_path),'ae_checkpoint':str(ae_path),'gru_shares':[.7,.65],
             'replaced_parameter_initializers':sorted(replacements),'nodes':len(graph.graph.node),
             'gru_epoch':gs['epoch'],'ae_epoch':ae['epoch'],
             'note':'Topology retained from selected submission; all inference weights replaced. Validate before packaging.'}
    (output/'export.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('Exported FP32 one-GRU + AE:',output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gru',type=Path,default=ROOT/'weights/gru_full.pt')
    p.add_argument('--ae',type=Path,default=ROOT/'weights/ae20.pt')
    p.add_argument('--output',type=Path,default=ROOT/'runs/export')
    args=p.parse_args(); export(args.gru,args.ae,args.output)
