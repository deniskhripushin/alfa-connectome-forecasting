"""Train the GRU with weighted MSE or Pearson loss, validating EMA weights."""
import argparse
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict
import json
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch

import train_gru_mse_fold as base


@contextmanager
def training_loss(correlation_weight, loss_name='weighted_mse', model=None, l2_weight=0.0):
    if loss_name not in ('weighted_mse', 'weighted_pearson'):
        raise ValueError('Unknown GRU loss')
    if loss_name == 'weighted_pearson' and correlation_weight:
        raise ValueError('Pure Pearson does not use an additional correlation weight')
    if not math.isfinite(l2_weight) or l2_weight < 0 or (l2_weight and model is None):
        raise ValueError('L2 requires a model and a finite nonnegative coefficient')
    original = base.weighted_mse
    if correlation_weight or loss_name == 'weighted_pearson' or l2_weight:
        if correlation_weight or loss_name == 'weighted_pearson':
            from supervised_ae_mlp import weighted_correlation_loss

        def combined(prediction, target):
            if loss_name == 'weighted_pearson':
                loss = weighted_correlation_loss(prediction.reshape(-1, 2), target.reshape(-1, 2))
            else:
                loss = original(prediction, target)
                if correlation_weight:
                    loss = loss + correlation_weight * weighted_correlation_loss(
                        prediction.reshape(-1, 2), target.reshape(-1, 2))
            if l2_weight:
                penalty = sum((p.float().square().sum() for p in model.parameters() if p.requires_grad),
                              prediction.new_zeros((), dtype=torch.float32))
                loss = loss + .5 * l2_weight * penalty
            return loss

        # Reuse the unchanged chunking/AMP loop without changing the baseline module on disk.
        base.weighted_mse = combined
    try:
        yield
    finally:
        base.weighted_mse = original


class EMAWeights:
    def __init__(self, model, decay):
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        for averaged, current in zip(self.model.parameters(), model.parameters()):
            averaged.mul_(self.decay).add_(current, alpha=1-self.decay)
        for averaged, current in zip(self.model.buffers(), model.buffers()):
            averaged.copy_(current)
        self.updates += 1

    def attach(self, optimizer, model):
        # GradScaler skips optimizer.step on overflow, so this hook also skips EMA.
        return optimizer.register_step_post_hook(lambda optimizer, args, kwargs: self.update(model))


def make_optimizer(model, args):
    options = dict(lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), momentum=args.momentum, **options)
    return torch.optim.AdamW(model.parameters(), **options)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=base.DEFAULT_TRAIN_PATH)
    parser.add_argument('--manifest', type=Path, default=base.DEFAULT_MANIFEST_PATH)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--init-checkpoint', type=Path,
                        help='Fine-tune saved EMA weights with a fresh optimizer and EMA')
    parser.add_argument('--check-inputs', action='store_true')
    for name, default in [('fold', 0), ('epochs', 8), ('patience', 2), ('batch-size', 16),
                          ('chunk-length', 512), ('hidden-size', 128), ('num-layers', 2),
                          ('normalization-sequences', 256), ('max-train-sequences', 0),
                          ('max-valid-sequences', 0), ('progress-every', 20), ('seed', 20260928)]:
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--l2-weight', type=float, default=0.0,
                        help='Explicit lambda/2 * sum of squared trainable parameters, including biases')
    parser.add_argument('--optimizer', choices=['adamw', 'sgd'], default='adamw')
    parser.add_argument('--momentum', type=float, default=.9, help='SGD momentum')
    parser.add_argument('--ema-decay', type=float, default=.999)
    parser.add_argument('--head-width', type=int, default=0,
                        help='0 keeps the original linear head; positive width adds Linear/ReLU/Linear')
    parser.add_argument('--correlation-weight', type=float, default=0.0,
                        help='Add batch target-wise weighted correlation loss to weighted MSE')
    parser.add_argument('--loss', choices=['weighted_mse', 'weighted_pearson'], default='weighted_mse',
                        help='Pure weighted Pearson replaces MSE, without an MSE anchor')
    args = parser.parse_args()
    if not np.isfinite(args.ema_decay) or not 0 < args.ema_decay < 1:
        parser.error('EMA decay must be in (0,1)')
    if min(args.epochs, args.patience, args.batch_size, args.chunk_length, args.hidden_size,
           args.num_layers, args.normalization_sequences, args.progress_every) < 1:
        parser.error('Sizes, epochs and patience must be positive')
    if args.max_train_sequences < 0 or args.max_valid_sequences < 0:
        parser.error('Sequence limits must be nonnegative')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('Invalid learning rate or weight decay')
    if not np.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        parser.error('Dropout must be in [0,1)')
    if args.head_width < 0:
        parser.error('Head width must be nonnegative')
    if not np.isfinite(args.correlation_weight) or args.correlation_weight < 0:
        parser.error('Correlation weight must be finite and nonnegative')
    if not np.isfinite(args.momentum) or not 0 <= args.momentum < 1:
        parser.error('SGD momentum must be in [0,1)')
    if args.loss == 'weighted_pearson' and args.correlation_weight:
        parser.error('Use --correlation-weight 0 with pure weighted Pearson')
    if not np.isfinite(args.l2_weight) or args.l2_weight < 0:
        parser.error('L2 weight must be finite and nonnegative')
    if args.l2_weight and args.weight_decay:
        parser.error('Use --weight-decay 0 when testing explicit L2')
    return args


def warm_start(path, config, train, valid):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('weights_source') != 'ema' or saved['model_config'] != asdict(config):
        raise ValueError('Warm start requires EMA weights and the same GRU architecture')
    for key, groups in (('train_groups', train), ('valid_groups', valid)):
        if set(saved[key]) != set(groups):
            raise ValueError(f'Warm-start split mismatch: {key}')
    state = saved['model_state_dict']
    mean = state['feature_mean'].reshape(-1).numpy().copy()
    std = state['feature_std'].reshape(-1).numpy().copy()
    if (mean.shape != (112,) or std.shape != (112,) or not np.isfinite(mean).all() or
            not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('Invalid warm-start normalization')
    if not set(saved['normalization_groups']).issubset(set(train)):
        raise ValueError('Normalization groups must belong to training split')
    return saved, mean, std


def save_checkpoint(path, raw_model, ema, optimizer, scaler, args, epoch, loss, result, norm, train, valid):
    payload = dict(epoch=epoch, model_config=asdict(raw_model.config),
                   model_state_dict=ema.model.state_dict(), weights_source='ema',
                   raw_model_state_dict=raw_model.state_dict(),
                   optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
                   ema_updates=ema.updates, ema_decay=ema.decay, train_loss=loss,
                   validation_result=result, normalization_groups=norm, train_groups=train, valid_groups=valid,
                   training_args={key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()})
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    args.loss_name = ('weighted_pearson' if args.loss == 'weighted_pearson' else
                      'weighted_mse_plus_correlation' if args.correlation_weight else 'weighted_mse')
    if args.l2_weight:
        args.loss_name += '_plus_l2'
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Choose a new output directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    base.seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    train, valid, _ = base.load_split(args.manifest, args.fold, args.max_train_sequences,
                                     args.max_valid_sequences, args.seed)
    parquet = pq.ParquetFile(args.train)
    if min(train+valid) < 0 or max(train+valid) >= parquet.num_row_groups:
        raise ValueError('Invalid row group index')
    model_class, config_class = base.ConnectomeGRU, base.ModelConfig
    config_kwargs = dict(hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout)
    if args.head_width:
        from gru_mlp_head import GRUWithMLPHead, MLPHeadConfig
        model_class, config_class = GRUWithMLPHead, MLPHeadConfig
        config_kwargs['head_width'] = args.head_width
    config = config_class(**config_kwargs)
    initial = None
    if args.init_checkpoint:
        initial, mean, std = warm_start(args.init_checkpoint, config, train, valid)
        norm = initial['normalization_groups']
        args.initial_checkpoint_sha256 = hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest()
        args.initial_epoch = int(initial['epoch'])
        args.optimizer_initialization = 'fresh_from_ema_weights'
    else:
        if args.check_inputs:
            raise ValueError('--check-inputs requires --init-checkpoint')
        mean, std, norm = base.compute_normalization(parquet, train, args.normalization_sequences, args.seed+args.fold)
    model = model_class(config, mean, std).to(device)
    if initial is not None:
        model.load_state_dict(initial['model_state_dict'], strict=True)
    if args.check_inputs:
        print(f'Fine-tune inputs PASSED: EMA epoch={args.initial_epoch}; train={len(train)}; '
              f'holdout={len(valid)}; saved normalization; fresh {args.optimizer} LR={args.learning_rate}', flush=True)
        return
    optimizer = make_optimizer(model, args)
    scaler = torch.amp.GradScaler('cuda')
    ema = EMAWeights(model, args.ema_decay)
    hook = ema.attach(optimizer, model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir/'run.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    print(f'GRU + EMA decay={ema.decay}; train={len(train)}; holdout={len(valid)}; constant LR={args.learning_rate}', flush=True)
    print('Validation/checkpoints use EMA weights; holdout is train-fold proxy, not official validation.', flush=True)
    print(f'Loss: {args.loss_name}; correlation weight={args.correlation_weight}', flush=True)
    print(f'Explicit L2: lambda={args.l2_weight}; penalty=lambda/2 * sum(theta^2); all trainable parameters', flush=True)
    print(f'Optimizer: {args.optimizer}; weight_decay={args.weight_decay}; '
          f'momentum={args.momentum if args.optimizer == "sgd" else "n/a"}', flush=True)
    best, stale, history = -math.inf, 0, []
    started = time.perf_counter()
    try:
        first_epoch = int(initial['epoch']) if initial is not None else 0
        if initial is not None:
            result = base.evaluate_holdout(ema.model, parquet, valid, args.batch_size,
                                           args.chunk_length, device, args.progress_every)
            best = float(result['weighted_pearson'])
            if not math.isfinite(best):
                raise FloatingPointError('Nonfinite initial validation score')
            save_checkpoint(args.output_dir/'best.pt', model, ema, optimizer, scaler, args,
                            first_epoch, initial['train_loss'], result, norm, train, valid)
            history.append(dict(epoch=first_epoch, fine_tune_epoch=0, initial_checkpoint=True,
                                weights_source='ema', ema_updates=0, **result))
            (args.output_dir/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            print(f'Initial EMA checkpoint retained as control: proxy WP={best:.9f}', flush=True)
        for local_epoch in range(1, args.epochs+1):
            epoch = first_epoch + local_epoch
            print(f'\nEpoch {epoch}; additional epoch {local_epoch}/{args.epochs}', flush=True)
            with training_loss(args.correlation_weight, args.loss, model, args.l2_weight):
                loss = base.train_epoch(model, optimizer, scaler, parquet, train, args.batch_size,
                                        args.chunk_length, device, epoch, args.seed, args.progress_every)
            result = base.evaluate_holdout(ema.model, parquet, valid, args.batch_size,
                                           args.chunk_length, device, args.progress_every)
            score = result['weighted_pearson']
            if not math.isfinite(score) or not math.isfinite(loss):
                raise FloatingPointError('Nonfinite loss or validation score')
            history.append(dict(epoch=epoch, fine_tune_epoch=local_epoch, train_loss=loss, weights_source='ema', ema_updates=ema.updates,
                                loss_name=args.loss_name, correlation_weight=args.correlation_weight,
                                l2_weight=args.l2_weight,
                                learning_rate=optimizer.param_groups[0]['lr'], **result))
            print(f'EMA: t0={result["t0"]:.6f}; t1={result["t1"]:.6f}; WP={score:.6f}; updates={ema.updates}', flush=True)
            for name in (['last.pt', 'best.pt'] if score > best+1e-6 else ['last.pt']):
                save_checkpoint(args.output_dir/name, model, ema, optimizer, scaler, args, epoch, loss,
                                result, norm, train, valid)
            if score > best+1e-6:
                best, stale = score, 0
                print('Saved best EMA checkpoint', flush=True)
            else:
                stale += 1
            (args.output_dir/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            if stale >= args.patience:
                print('Early stopping', flush=True)
                break
    finally:
        hook.remove()
    torch.cuda.synchronize()
    print(f'Best proxy WP={best:.6f}; elapsed={base.format_duration(time.perf_counter()-started)}\nCheckpoint: {args.output_dir/"best.pt"}')


if __name__ == '__main__':
    main()
