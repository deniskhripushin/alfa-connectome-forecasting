"""Create a new archive without modifying the original scored artifact."""
import argparse, hashlib, json, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FILES=('solution.py','base_runtime.py','model.onnx','normalization.npz','history_windows.dll','history_linux.so','history.c')

def package(source,output):
    if output.exists(): raise FileExistsError(output)
    for n in FILES:
        if not (source/n).is_file(): raise FileNotFoundError(source/n)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for n in FILES:
            info=zipfile.ZipInfo(n,(2026,10,20,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(source/n).read_bytes())
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert z.namelist()==list(FILES)
        for n in FILES: assert z.read(n)==(source/n).read_bytes()
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,
                      'sha256':hashlib.sha256(output.read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'submission')
    p.add_argument('--output',type=Path,default=ROOT/'runs/submission.zip')
    a=p.parse_args();package(a.source,a.output)
