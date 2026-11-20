"""The same float32 ONNX model with fused native causal-history preprocessing."""
import ctypes
import importlib.util
from math import isfinite
from pathlib import Path
import platform
import numpy as np

_spec = importlib.util.spec_from_file_location(
    '_full_gru_native_base', Path(__file__).resolve().with_name('base_runtime.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
BaseModel = _base.PredictionModel


class History(ctypes.Structure):
    _fields_ = [('mean', ctypes.c_void_p), ('std', ctypes.c_void_p),
                ('engineered', ctypes.c_void_p), ('ema', ctypes.c_void_p), ('ring', ctypes.c_void_p)]


class PredictionModel(BaseModel):
    def __init__(self, directory=None):
        directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
        super().__init__(directory)
        system, machine = platform.system(), platform.machine().lower()
        if machine not in ('amd64', 'x86_64'):
            raise RuntimeError(f'Native history requires x86_64, got {machine}')
        if system not in ('Windows', 'Linux'):
            raise RuntimeError(f'Unsupported system: {system}')
        library = 'history_windows.dll' if system == 'Windows' else 'history_linux.so'
        self.native = ctypes.CDLL(str(directory/library))
        self.native.history_version.argtypes = []
        self.native.history_version.restype = ctypes.c_int
        if self.native.history_version() != 1:
            raise RuntimeError('History module ABI mismatch')
        self.context = History(*(v.ctypes.data for v in (self.mean, self.std, self.engineered, self.ema, self.ring)))
        self.context_pointer = ctypes.byref(self.context)
        self.history_step = self.native.history_step
        self.history_step.argtypes = [ctypes.POINTER(History), ctypes.c_void_p, ctypes.c_int]
        self.history_step.restype = ctypes.c_int

    def predict(self, point):
        seq, step = int(point.seq_ix), int(point.step_in_seq)
        if seq != self.seq:
            if step != 0:
                raise ValueError('A new sequence must start at zero')
            for states in self.buffers:
                for state in states:
                    state.fill(0)
            self.index, self.states = 0, self.buffers[0]
            self.ring.fill(0)
            self.ema.fill(0)
            self.seq, self.previous_step = seq, None
        if self.previous_step is not None and step != self.previous_step+1:
            raise ValueError('Rows must arrive in order')
        if not 0 <= step < 20000:
            raise ValueError('Step outside sequence')
        x = np.asarray(point.state, dtype=np.float32)
        if x.shape != (112,):
            raise ValueError('Expected 112 features')
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x)
        if not self.history_step(self.context_pointer, x.ctypes.data, step):
            raise ValueError('Expected finite features')
        self.session.run_with_iobinding(self.bindings[self.index], self.run_options)
        self.index = 1-self.index
        self.states = self.buffers[self.index]
        self.previous_step = step
        if not point.need_prediction:
            return None
        if not (isfinite(self.prediction[0]) and isfinite(self.prediction[1])):
            raise ValueError('Expected two finite predictions')
        return self.prediction.copy()
