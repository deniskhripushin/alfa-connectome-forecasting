"""Initialize bundled CUDA before importing PyTorch to fix Windows RNN shutdown.

PyTorch 2.11/cu128 can abort during native teardown after using cuDNN RNN
dropout on Windows. Initializing the runtime before loading torch's DLLs avoids
the teardown-order failure. This does not change RNG, kernels, or model settings.
No exit codes or exceptions are suppressed; Python still shuts down normally.
"""
import ctypes
import importlib.util
from pathlib import Path
import runpy
import sys


_runtime = None


def initialize_cuda_before_torch():
    global _runtime
    if sys.platform != "win32" or _runtime is not None:
        return
    if "torch" in sys.modules:
        raise RuntimeError("CUDA startup fix must run BEFORE importing torch")
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("PyTorch is not installed in the active environment")
    library_dir = Path(next(iter(spec.submodule_search_locations))) / "lib"
    libraries = list(library_dir.glob("cudart64_*.dll"))
    if len(libraries) != 1:
        raise RuntimeError(f"Expected exactly one bundled CUDA runtime in {library_dir}")
    runtime = ctypes.WinDLL(str(libraries[0].resolve()))
    runtime.cudaFree.argtypes = [ctypes.c_void_p]
    runtime.cudaFree.restype = ctypes.c_int
    runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
    runtime.cudaGetErrorString.restype = ctypes.c_char_p
    # cudaFree(NULL) initializes CUDA without allocating a tensor or touching RNG.
    status = runtime.cudaFree(None)
    if status != 0:
        message = runtime.cudaGetErrorString(status)
        raise RuntimeError(f"Early CUDA initialization failed ({status}): {message!r}")
    _runtime = runtime


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_with_cuda_init.py SCRIPT.py [arguments ...]")
    script = Path(sys.argv[1]).resolve(strict=True)
    initialize_cuda_before_torch()
    print("CUDA runtime initialized before PyTorch; normal exit/error handling retained.", flush=True)
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path[0] = str(script.parent)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
