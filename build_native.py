"""Rebuild the causal-history library with Zig 0.13.0, without fast-math/FMA."""
import argparse, os, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zig',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'runs/native')
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    version=subprocess.check_output([str(a.zig.resolve()),'version'],text=True).strip()
    if version!='0.13.0': raise ValueError('Original compiler version is 0.13.0')
    a.output.mkdir(parents=True)
    shutil.copyfile(ROOT/'submission/history.c',a.output/'history.c')
    env=os.environ.copy();env['ZIG_GLOBAL_CACHE_DIR']=str((ROOT/'runs/zig-cache').resolve())
    common=[str(a.zig.resolve()),'cc','-O3','-ffp-contract=off','-fno-fast-math',
            '-fno-stack-protector','-fno-builtin','-shared','-mcpu=baseline',str(a.output/'history.c')]
    for target,name,flags in [('x86_64-windows-gnu','history_windows.dll',[]),
          ('x86_64-linux-gnu','history_linux.so',['-fPIC','-nostdlib','-Wl,-z,noexecstack','-Wl,--no-undefined'])]:
        subprocess.run(common+['-target',target,*flags,'-o',str(a.output/name)],env=env,check=True)
    print('Built both libraries in',a.output,'; original libraries unchanged.')
