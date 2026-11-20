"""Optional small MLP output head for the existing GRU backbone."""
from dataclasses import dataclass

import numpy as np
import torch

from train_gru_mse_fold import ConnectomeGRU, ModelConfig


@dataclass
class MLPHeadConfig(ModelConfig):
    head_width: int = 32


class GRUWithMLPHead(ConnectomeGRU):
    def __init__(self, config, feature_mean, feature_std):
        if config.head_width < 1:
            raise ValueError('Head width must be positive')
        super().__init__(config, feature_mean, feature_std)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.head_width),
            torch.nn.ReLU(),
            torch.nn.Linear(config.head_width, config.output_size),
        )


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = MLPHeadConfig(**checkpoint['model_config'])
    state = checkpoint['model_state_dict']
    mean = state['feature_mean'].reshape(-1).numpy()
    std = state['feature_std'].reshape(-1).numpy()
    if (mean.shape != (config.input_size,) or std.shape != mean.shape or
            not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('Invalid normalization')
    model = GRUWithMLPHead(config, mean, std)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint
