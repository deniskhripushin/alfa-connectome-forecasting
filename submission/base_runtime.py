"""Full-train seed-20260928 GRU plus unchanged AE20, fixed shares 70/65."""
from pathlib import Path
from math import isfinite
import numpy as np
import onnxruntime as ort


class PredictionModel:
    def __init__(self, directory=None):
        directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
        with np.load(directory/'normalization.npz', allow_pickle=False) as saved:
            self.mean, self.std = saved['mean'].copy(), saved['std'].copy()
        if (self.mean.shape != (112,) or self.std.shape != (112,) or
                self.mean.dtype != np.float32 or self.std.dtype != np.float32 or
                not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or
                not (self.std > 0).all()):
            raise ValueError('Expected the frozen float32 normalization')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(directory/'model.onnx'), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        inputs = {'features': [1, 1, 112], 'mlp_features': [1, 560],
                  'new_h0': [1, 1, 128], 'new_h1': [1, 1, 128]}
        outputs = {'prediction': [1, 2], 'next_new_h0': [1, 1, 128], 'next_new_h1': [1, 1, 128]}
        if ({v.name: v.shape for v in self.session.get_inputs()} != inputs or
                {v.name: v.shape for v in self.session.get_outputs()} != outputs or
                self.session.get_modelmeta().custom_metadata_map.get('ensemble') != 'full10607_gru_ae20_70_65'):
            raise ValueError('Unexpected one-GRU plus AE20 graph')
        self.buffers = [[np.zeros((1, 1, 128), np.float32) for _ in range(2)] for _ in range(2)]
        self.index, self.states = 0, self.buffers[0]
        self.ring = np.zeros((20, 112), np.float32)
        # Preserve the original history calculation, including float64 EMA accumulation.
        self.ema = np.zeros((2, 112), np.float64)
        self.alpha = np.array([[2/21], [2/101]], np.float64)
        self.beta = 1-self.alpha
        self.beta32 = self.beta.astype(np.float32)
        self.engineered = np.empty(560, np.float32)
        self.z = self.engineered[:112]
        self.delta1 = self.engineered[112:224]
        self.delta20 = self.engineered[224:336]
        self.ema_features = self.engineered[336:]
        self.history_tail = self.engineered[112:]
        self.residual = np.empty((2, 112), np.float64)
        self.residual_flat = self.residual.reshape(-1)
        self.ema_increment = np.empty((2, 112), np.float64)
        self.initial_ema = np.empty((2, 112), np.float32)
        self.output = np.empty((1, 2), np.float32)
        self.prediction = self.output.reshape(2)
        self.run_options = ort.RunOptions()
        self.bindings, self.ort_values = [], []
        for index in range(2):
            binding = self.session.io_binding()
            binding.bind_cpu_input('features', self.z.reshape(1, 1, 112))
            binding.bind_cpu_input('mlp_features', self.engineered.reshape(1, 560))
            for name, value in zip(('new_h0', 'new_h1'), self.buffers[index]):
                binding.bind_cpu_input(name, value)
            for name, value in zip(outputs, (self.output, *self.buffers[1-index])):
                ov = ort.OrtValue.ortvalue_from_numpy(value)
                self.ort_values.append(ov)
                binding.bind_ortvalue_output(name, ov)
            self.bindings.append(binding)
        self.seq, self.previous_step = None, None

    @property
    def hidden(self):
        return np.concatenate(self.states, axis=0)

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
        x = np.asarray(point.state, dtype=np.float32)
        if x.shape != (112,) or not np.isfinite(x).all():
            raise ValueError('Expected 112 finite features')
        np.subtract(x, self.mean, out=self.z)
        np.divide(self.z, self.std, out=self.z)
        np.maximum(self.z, -10, out=self.z)
        np.minimum(self.z, 10, out=self.z)
        if step >= 1:
            np.subtract(self.z, self.ring[(step-1) % 20], out=self.delta1)
        else:
            self.delta1.fill(0)
        if step >= 20:
            np.subtract(self.z, self.ring[(step-20) % 20], out=self.delta20)
        else:
            self.delta20.fill(0)
        if step == 0:
            np.multiply(self.beta32, self.z, out=self.initial_ema)
            self.ema[:] = self.initial_ema
        else:
            self.ema *= self.beta
        np.multiply(self.alpha, self.z, out=self.ema_increment)
        self.ema += self.ema_increment
        np.subtract(self.z, self.ema, out=self.residual)
        self.ema_features[:] = self.residual_flat
        np.maximum(self.history_tail, -10, out=self.history_tail)
        np.minimum(self.history_tail, 10, out=self.history_tail)
        self.ring[step % 20] = self.z
        self.session.run_with_iobinding(self.bindings[self.index], self.run_options)
        self.index = 1-self.index
        self.states = self.buffers[self.index]
        self.previous_step = step
        if not point.need_prediction:
            return None
        if not (isfinite(self.prediction[0]) and isfinite(self.prediction[1])):
            raise ValueError('Expected two finite predictions')
        return self.prediction.copy()
