"""Supervised denoising AE + MLP for Connectome regression, with an MLP control.

Inspired by Yirun Zhang's Jane Street supervised autoencoder, not a reproduction.
Only train-fold rows are used for normalization and all three training losses.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from mlp_experiment import read_sequence, make_features, evaluate
from train_gru_fold import compute_normalization, load_split, seed_everything
from train_gru_mse_fold import weighted_mse
from utils import FEATURE_COLUMNS


@dataclass
class Config:
    kind: str = "ae"
    mode: str = "current"
    width: int = 128
    latent: int = 64
    dropout: float = 0.1
    noise_std: float = 0.03
    activation: str = "relu"
    batch_norm: bool = False
    auxiliary_source: str = "encoder"


class SupervisedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.kind not in ("ae", "mlp") or config.mode not in ("current", "history"):
            raise ValueError("Unknown model or feature mode")
        if config.activation not in ("relu", "swish", "silu"):
            raise ValueError("Unknown activation")
        if config.auxiliary_source not in ("encoder", "decoder"):
            raise ValueError("Unknown auxiliary source")
        if config.kind != "ae" and config.auxiliary_source != "encoder":
            raise ValueError("Decoder auxiliary head requires an AE model")
        activation = nn.ReLU if config.activation == "relu" else nn.SiLU
        size = len(FEATURE_COLUMNS) * (5 if config.mode == "history" else 1)
        if config.kind == "ae":
            self.encoder = nn.Sequential(nn.Linear(size, config.width), activation(),
                                         nn.Linear(config.width, config.latent), activation())
            self.decoder = nn.Sequential(nn.Linear(config.latent, config.width), activation(),
                                         nn.Linear(config.width, size))
            self.auxiliary = nn.Linear(config.latent, 2)
        head_size = size + (config.latent if config.kind == "ae" else 0)
        head = []
        for input_size, output_size in ((head_size, config.width), (config.width, 64)):
            head.append(nn.Linear(input_size, output_size))
            if config.batch_norm:
                head.append(nn.BatchNorm1d(output_size, eps=1e-5, momentum=0.1,
                                           affine=True, track_running_stats=True))
            head.extend((activation(), nn.Dropout(config.dropout)))
        head.append(nn.Linear(64, 2))
        self.head = nn.Sequential(*head)
        if config.kind == "ae" and config.auxiliary_source == "decoder":
            # Keep shared-layer initialization and subsequent RNG state identical
            # to the encoder-head baseline despite the larger auxiliary layer.
            with torch.random.fork_rng(devices=[]):
                self.auxiliary = nn.Linear(size, 2)

    def encode(self, x):
        if self.training and self.config.noise_std:
            x = x + torch.randn_like(x) * self.config.noise_std
        return self.encoder(x)

    def forward(self, x):
        # Decoder and auxiliary head are absent from the inference path.
        inputs = torch.cat((x, self.encode(x)), dim=-1) if self.config.kind == "ae" else x
        return 2.0 * torch.tanh(self.head(inputs))

    def training_outputs(self, x):
        if self.config.kind == "mlp":
            return self(x), None, None
        latent = self.encode(x)
        prediction = 2.0 * torch.tanh(self.head(torch.cat((x, latent), dim=-1)))
        reconstruction = self.decoder(latent)
        auxiliary_input = reconstruction if self.config.auxiliary_source == "decoder" else latent
        return prediction, 2.0 * torch.tanh(self.auxiliary(auxiliary_input)), reconstruction


def weighted_correlation_loss(prediction, target):
    prediction, target = prediction.float().clamp(-2, 2), target.float().clamp(-2, 2)
    weight = target.abs()
    mass = weight.sum(dim=0).clamp_min(1e-8)
    centered_p = prediction - (weight * prediction).sum(dim=0) / mass
    centered_y = target - (weight * target).sum(dim=0) / mass
    covariance = (weight * centered_p * centered_y).sum(dim=0) / mass
    variance_p = (weight * centered_p.square()).sum(dim=0) / mass
    variance_y = (weight * centered_y.square()).sum(dim=0) / mass
    denominator = (variance_p.clamp_min(1e-8) * variance_y.clamp_min(1e-8)).sqrt()
    correlation = (covariance / denominator).clamp(-1, 1)
    # Degenerate target columns have no meaningful correlation; MSE still applies.
    valid = (weight.sum(dim=0) > 0) & (variance_y > 1e-8)
    return ((1-correlation) * valid).sum() / valid.sum().clamp_min(1)


def losses(model, x, y, reconstruction_weight, auxiliary_weight, correlation_weight=0.0):
    if model.training and model.config.batch_norm and len(x) < 2:
        raise ValueError("BatchNorm training requires at least two sampled rows per update")
    prediction, auxiliary, reconstruction = model.training_outputs(x)
    main = weighted_mse(prediction, y)
    aux = weighted_mse(auxiliary, y) if auxiliary is not None else main.new_zeros(())
    rec = (reconstruction - x).square().mean() if reconstruction is not None else main.new_zeros(())
    corr = weighted_correlation_loss(prediction, y) if correlation_weight else main.new_zeros(())
    total = main + auxiliary_weight * aux + reconstruction_weight * rec + correlation_weight * corr
    return total, torch.stack((main.detach(), aux.detach(), rec.detach(), corr.detach()))


def step_lr_scheduler(scheduler, optimizer, score):
    before = [group["lr"] for group in optimizer.param_groups]
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(score)
        else:
            scheduler.step()
    after = [group["lr"] for group in optimizer.param_groups]
    return before, after, any(new < old for old, new in zip(before, after))


class EMAWeights:
    def __init__(self, model, decay):
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        for averaged, current in zip(self.model.parameters(), model.parameters()):
            averaged.mul_(self.decay).add_(current, alpha=1-self.decay)
        # Running statistics and counters keep their original meaning.
        for averaged, current in zip(self.model.buffers(), model.buffers()):
            averaged.copy_(current)
        self.updates += 1


def train_epoch(model, optimizer, parquet, groups, mean, std, args, epoch, device, ema=None):
    model.train()
    order = np.random.default_rng(args.seed + epoch).permutation(groups)
    features, targets = [], []
    totals = np.zeros(5, dtype=np.float64)
    rows_seen = 0
    for position, group in enumerate(order, 1):
        x, y, mask = read_sequence(parquet, int(group))
        eligible = np.flatnonzero(mask)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch, int(group)]))
        rows = rng.choice(eligible, min(args.rows_per_sequence, len(eligible)), replace=False)
        features.append(make_features(x, mean, std, args.mode, rows))
        targets.append(y[rows])
        if len(features) == args.batch_sequences or position == len(order):
            batch = torch.from_numpy(np.concatenate(features)).to(device)
            target = torch.from_numpy(np.concatenate(targets)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = losses(model, batch, target, args.reconstruction_weight, args.auxiliary_weight,
                                 args.correlation_weight)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            totals += np.r_[loss.item(), parts.cpu().numpy()] * len(batch)
            rows_seen += len(batch)
            features, targets = [], []
        if position % 128 == 0 or position == len(order):
            print(f"  train {position}/{len(order)}; loss={totals[0]/max(rows_seen, 1):.6f}", flush=True)
    return dict(zip(("total", "main_wmse", "aux_wmse", "reconstruction_mse", "main_correlation_loss"),
                    (totals / rows_seen).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", choices=("ae", "mlp"), default="ae")
    parser.add_argument("--mode", choices=("current", "history"), default="current")
    parser.add_argument("--activation", choices=("relu", "swish", "silu"), default="relu")
    parser.add_argument("--batch-norm", action="store_true",
                        help="BatchNorm after each hidden Linear in the main MLP only")
    parser.add_argument("--auxiliary-source", choices=("encoder", "decoder"), default="encoder",
                        help="Input to the auxiliary target head; main prediction path is unchanged")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--rows-per-sequence", type=int, default=512)
    parser.add_argument("--batch-sequences", type=int, default=8)
    parser.add_argument("--max-train-sequences", type=int, default=0)
    parser.add_argument("--max-valid-sequences", type=int, default=0)
    parser.add_argument("--normalization-sequences", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--correlation-weight", type=float, default=0.0,
                        help="Additional batch-wise clipped weighted Pearson loss on the main head only")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-scheduler", choices=("constant", "plateau", "step"), default="constant")
    parser.add_argument("--lr-step-epoch", type=int, default=5,
                        help="For step scheduling, reduce LR after completing this epoch")
    parser.add_argument("--lr-patience", type=int, default=2,
                        help="Reduce after this many tolerated non-improving epochs (2 means reduce on the third)")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="0 disables EMA; otherwise validate and save EMA weights updated after every optimizer step")
    args = parser.parse_args()
    if not np.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1:
        parser.error("EMA decay must be finite and in [0,1)")
    if args.model != "ae" and args.auxiliary_source == "decoder":
        parser.error("--auxiliary-source decoder requires --model ae")
    if min(args.epochs, args.patience, args.rows_per_sequence, args.batch_sequences,
           args.normalization_sequences, args.width, args.latent) < 1:
        parser.error("Sizes, epochs and patience must be positive")
    nonnegative = (args.noise_std, args.reconstruction_weight, args.auxiliary_weight, args.correlation_weight,
                   args.weight_decay, args.max_train_sequences, args.max_valid_sequences)
    if any(not np.isfinite(v) or v < 0 for v in nonnegative) or not 0 <= args.dropout < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Invalid loss weights, regularization, limits or learning rate")
    if args.lr_patience < 0 or not np.isfinite(args.lr_factor) or not 0 < args.lr_factor < 1:
        parser.error("LR patience must be nonnegative and LR factor must be between 0 and 1")
    if args.lr_step_epoch < 1:
        parser.error("LR step epoch must be positive")
    if not np.isfinite(args.min_learning_rate) or args.min_learning_rate <= 0:
        parser.error("Minimum learning rate must be finite and positive")
    if args.lr_scheduler == "plateau":
        if args.min_learning_rate >= args.learning_rate:
            parser.error("Minimum learning rate must be below the starting learning rate")
        if args.patience <= args.lr_patience:
            parser.error("Early stopping patience must exceed LR patience so LR can decrease before stopping")
    root = args.project.resolve()
    run = args.run if args.run.is_absolute() else root / args.run
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate project venv or select --device cpu")
    torch.set_num_threads(1)
    seed_everything(args.seed)
    if args.evaluate:
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        if checkpoint["feature_columns"] != FEATURE_COLUMNS or checkpoint["format_version"] != 1:
            raise ValueError("Checkpoint contract mismatch")
        config = Config(**checkpoint["config"])
        model = SupervisedMLP(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        parquet = pq.ParquetFile(root / "datasets/valid.parquet")
        result = evaluate(model, parquet, list(range(parquet.num_row_groups)),
                          checkpoint["mean"].numpy(), checkpoint["std"].numpy(), config.mode, device, official=True)
        result.update(checkpoint_epoch=checkpoint["epoch"], config=checkpoint["config"],
                      evaluation="official_need_prediction_and_is_scored")
        (run / "official_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Use a new output directory: {run}")
    train, holdout, _ = load_split(root / "sequence_folds.json", args.fold,
                                  args.max_train_sequences, args.max_valid_sequences, args.seed)
    if not train or not holdout or set(train) & set(holdout):
        raise ValueError("Invalid train/holdout partition")
    parquet = pq.ParquetFile(root / "datasets/train.parquet")
    mean, std, norm_groups = compute_normalization(parquet, train, args.normalization_sequences, args.seed + args.fold)
    config = Config(args.model, args.mode, args.width, args.latent, args.dropout, args.noise_std,
                    args.activation, args.batch_norm, args.auxiliary_source)
    model = SupervisedMLP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ema = EMAWeights(model, args.ema_decay) if args.ema_decay else None
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience,
            threshold=1e-6, threshold_mode="abs", min_lr=args.min_learning_rate)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[args.lr_step_epoch], gamma=args.lr_factor)
    run.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (run / "run.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"{config}; parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"Train={len(train)}; holdout={len(holdout)}; device={device}; LR={args.learning_rate}; scheduler={args.lr_scheduler}")
    print("Holdout is train-fold proxy, NOT official is_scored validation.", flush=True)
    if ema is not None:
        print(f"Validation/checkpoints use EMA weights; decay={args.ema_decay}", flush=True)
    best, stale, history = -float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs}; LR={learning_rate:.8g}", flush=True)
        components = train_epoch(model, optimizer, parquet, train, mean, std, args, epoch, device, ema)
        evaluation_model = ema.model if ema is not None else model
        result = evaluate(evaluation_model, parquet, holdout, mean, std, args.mode, device)
        wp = result["weighted_pearson"]
        if not np.isfinite(wp):
            raise ValueError("Nonfinite validation score")
        print(f"Losses={components}; t0={result['t0']:.6f}; t1={result['t1']:.6f}; WP={wp:.6f}", flush=True)
        if wp > best + 1e-6:
            best, stale = wp, 0
            checkpoint = dict(format_version=1, config=asdict(config), epoch=epoch, seed=args.seed,
                              model_state_dict=evaluation_model.state_dict(), mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                              feature_columns=FEATURE_COLUMNS, train_groups=train, valid_groups=holdout,
                              normalization_groups=norm_groups, settings=settings, validation_result=result,
                              weights_source="ema" if ema is not None else "training",
                              ema_updates=ema.updates if ema is not None else 0)
            temporary = run / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(run / "best.pt")
            print("Saved best checkpoint", flush=True)
        else:
            stale += 1
        _, next_rates, reduced = step_lr_scheduler(scheduler, optimizer, wp)
        if reduced:
            # Allow a full early-stopping window at the newly reduced LR.
            stale = 0
            print(f"LR reduced: {learning_rate:.8g} -> {next_rates[0]:.8g}", flush=True)
        history.append({"epoch": epoch, "learning_rate": learning_rate,
                        "weights_source": "ema" if ema is not None else "training",
                        "ema_updates": ema.updates if ema is not None else 0,
                        "next_learning_rate": next_rates[0], "lr_reduced": reduced,
                        "early_stopping_stale": stale, "losses": components, "validation": result})
        (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale >= args.patience:
            print("Early stopping")
            break
    print(f"Best proxy WP={best:.6f}; elapsed={time.perf_counter()-started:.1f}s\nCheckpoint: {run / 'best.pt'}")


if __name__ == "__main__":
    main()
