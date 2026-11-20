"""Train a stateful GRU on one Alfa Connectome sequence fold.

The split is read from sequence_folds.json. Complete Parquet row groups are
assigned either to training or holdout, so rows from one sequence never cross
the split boundary. The holdout metric uses every row after the warm-up because
train.parquet does not contain the hidden official is_scored mask.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from utils import (
    FEATURE_COLUMNS,
    GlobalAccumulator,
    SEQUENCE_LENGTH,
    TARGET_COLUMNS,
    WARMUP,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = ROOT / "datasets" / "train.parquet"
DEFAULT_MANIFEST_PATH = ROOT / "sequence_folds.json"
DEFAULT_OUTPUT_DIR = ROOT / "checkpoints" / "gru_fold0"


@dataclass
class ModelConfig:
    input_size: int = len(FEATURE_COLUMNS)
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    output_size: int = len(TARGET_COLUMNS)
    input_clip: float = 10.0


class ConnectomeGRU(nn.Module):
    """GRU with fixed train-fold normalization embedded in the model state."""

    def __init__(
        self,
        config: ModelConfig,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(feature_mean, dtype=torch.float32).reshape(1, 1, -1),
        )
        self.register_buffer(
            "feature_std",
            torch.as_tensor(feature_std, dtype=torch.float32).reshape(1, 1, -1),
        )
        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(config.hidden_size, config.output_size)

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features - self.feature_mean) / self.feature_std
        normalized = normalized.clamp(-self.config.input_clip, self.config.input_clip)
        states, hidden = self.gru(normalized, hidden)
        # The official metric clips predictions to [-2, 2]. A bounded head keeps
        # training and inference within that same useful output range.
        predictions = 2.0 * torch.tanh(self.head(states))
        return predictions, hidden


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--normalization-sequences", type=int, default=256)
    parser.add_argument(
        "--max-train-sequences",
        type=int,
        default=0,
        help="0 uses every sequence outside the holdout fold",
    )
    parser.add_argument(
        "--max-valid-sequences",
        type=int,
        default=0,
        help="0 uses every sequence in the holdout fold",
    )
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_split(
    manifest_path: Path,
    fold: int,
    max_train_sequences: int,
    max_valid_sequences: int,
    seed: int,
) -> tuple[list[int], list[int], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_folds = int(manifest["n_folds"])
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds - 1}]")

    valid_groups = [int(group) for group in manifest["train_folds"][str(fold)]]
    train_groups = [
        int(group)
        for other_fold in range(n_folds)
        if other_fold != fold
        for group in manifest["train_folds"][str(other_fold)]
    ]

    # Limits are deterministic and stratified across a shuffled list. They are
    # useful for a pilot run before committing to the complete fold.
    rng = np.random.default_rng(seed + fold)
    rng.shuffle(train_groups)
    rng.shuffle(valid_groups)
    if max_train_sequences > 0:
        train_groups = train_groups[:max_train_sequences]
    if max_valid_sequences > 0:
        valid_groups = valid_groups[:max_valid_sequences]

    if not train_groups or not valid_groups:
        raise ValueError("both train and holdout splits must contain sequences")
    if set(train_groups) & set(valid_groups):
        raise ValueError("train and holdout row groups overlap")
    return train_groups, valid_groups, manifest


def read_sequence(
    parquet: pq.ParquetFile,
    row_group: int,
    include_targets: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    columns = [*FEATURE_COLUMNS, *TARGET_COLUMNS] if include_targets else FEATURE_COLUMNS
    table = parquet.read_row_group(row_group, columns=columns, use_threads=False)
    if table.num_rows != SEQUENCE_LENGTH:
        raise ValueError(
            f"row group {row_group} has {table.num_rows} rows, expected {SEQUENCE_LENGTH}"
        )
    features = np.column_stack(
        [table[column].to_numpy(zero_copy_only=False) for column in FEATURE_COLUMNS]
    ).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise ValueError(f"nonfinite feature in row group {row_group}")

    if not include_targets:
        return features, None
    targets = np.column_stack(
        [table[column].to_numpy(zero_copy_only=False) for column in TARGET_COLUMNS]
    ).astype(np.float32, copy=False)
    if not np.isfinite(targets).all():
        raise ValueError(f"nonfinite target in row group {row_group}")
    return features, targets


def read_batch(
    parquet: pq.ParquetFile,
    row_groups: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    feature_parts = []
    target_parts = []
    for row_group in row_groups:
        features, targets = read_sequence(parquet, row_group)
        feature_parts.append(features)
        target_parts.append(targets)
    return np.stack(feature_parts), np.stack(target_parts)


def compute_normalization(
    parquet: pq.ParquetFile,
    train_groups: list[int],
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    count_to_use = min(max(1, sample_size), len(train_groups))
    rng = np.random.default_rng(seed)
    sampled_groups = rng.choice(train_groups, size=count_to_use, replace=False).tolist()
    sums = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    sum_squares = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    count = 0
    started = time.perf_counter()

    print(f"Computing normalization on {count_to_use:,} train sequences...")
    for position, row_group in enumerate(sampled_groups, start=1):
        features, _ = read_sequence(parquet, int(row_group), include_targets=False)
        values = features.astype(np.float64, copy=False)
        sums += values.sum(axis=0)
        sum_squares += np.square(values).sum(axis=0)
        count += len(values)
        if position % 32 == 0 or position == count_to_use:
            print(f"  normalization: {position:,}/{count_to_use:,} sequences")

    mean = sums / count
    variance = np.maximum(sum_squares / count - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    print(
        f"Normalization ready in {format_duration(time.perf_counter() - started)}; "
        f"rows={count:,}; std range=[{std.min():.6g}, {std.max():.6g}]"
    )
    return mean.astype(np.float32), std.astype(np.float32), sampled_groups


def weighted_smooth_l1(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    clipped_targets = targets.clamp(-2.0, 2.0)
    weights = clipped_targets.abs()
    losses = nn.functional.smooth_l1_loss(
        predictions,
        clipped_targets,
        reduction="none",
        beta=0.25,
    )
    return (weights * losses).sum() / weights.sum().clamp_min(1e-8)


def train_batch(
    model: ConnectomeGRU,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    features: np.ndarray,
    targets: np.ndarray,
    chunk_length: int,
    device: torch.device,
) -> tuple[float, int]:
    model.train()
    hidden = None
    loss_sum = 0.0
    updates = 0

    for start in range(0, SEQUENCE_LENGTH, chunk_length):
        stop = min(start + chunk_length, SEQUENCE_LENGTH)
        score_start = max(start, WARMUP)
        x = torch.from_numpy(features[:, start:stop]).to(device, non_blocking=True)
        y = torch.from_numpy(targets[:, start:stop]).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            predictions, hidden = model(x, hidden)
            local_start = score_start - start
            loss = weighted_smooth_l1(
                predictions[:, local_start:],
                y[:, local_start:],
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        hidden = hidden.detach()
        loss_sum += float(loss.detach())
        updates += 1
    return loss_sum, updates


def train_epoch(
    model: ConnectomeGRU,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    parquet: pq.ParquetFile,
    train_groups: list[int],
    batch_size: int,
    chunk_length: int,
    device: torch.device,
    epoch: int,
    seed: int,
    progress_every: int,
) -> float:
    epoch_groups = train_groups.copy()
    random.Random(seed + epoch).shuffle(epoch_groups)
    n_batches = math.ceil(len(epoch_groups) / batch_size)
    total_loss = 0.0
    total_updates = 0
    started = time.perf_counter()

    for batch_index, start in enumerate(range(0, len(epoch_groups), batch_size), start=1):
        groups = epoch_groups[start : start + batch_size]
        features, targets = read_batch(parquet, groups)
        batch_loss, updates = train_batch(
            model,
            optimizer,
            scaler,
            features,
            targets,
            chunk_length,
            device,
        )
        total_loss += batch_loss
        total_updates += updates

        if batch_index % progress_every == 0 or batch_index == n_batches:
            elapsed = time.perf_counter() - started
            rate = batch_index / elapsed
            eta = (n_batches - batch_index) / max(rate, 1e-12)
            print(
                f"  train batch {batch_index:,}/{n_batches:,} "
                f"| sequences={min(start + len(groups), len(epoch_groups)):,}/{len(epoch_groups):,} "
                f"| loss={total_loss / total_updates:.6f} "
                f"| elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}"
            )

    return total_loss / total_updates


@torch.inference_mode()
def evaluate_holdout(
    model: ConnectomeGRU,
    parquet: pq.ParquetFile,
    valid_groups: list[int],
    batch_size: int,
    chunk_length: int,
    device: torch.device,
    progress_every: int,
) -> dict:
    model.eval()
    accumulator = GlobalAccumulator()
    n_batches = math.ceil(len(valid_groups) / batch_size)
    started = time.perf_counter()
    need_prediction = np.arange(SEQUENCE_LENGTH) >= WARMUP

    for batch_index, start in enumerate(range(0, len(valid_groups), batch_size), start=1):
        groups = valid_groups[start : start + batch_size]
        features, targets = read_batch(parquet, groups)
        hidden = None
        prediction_parts = []

        for chunk_start in range(0, SEQUENCE_LENGTH, chunk_length):
            chunk_stop = min(chunk_start + chunk_length, SEQUENCE_LENGTH)
            x = torch.from_numpy(features[:, chunk_start:chunk_stop]).to(
                device, non_blocking=True
            )
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                predictions, hidden = model(x, hidden)
            prediction_parts.append(predictions.float().cpu().numpy())

        batch_predictions = np.concatenate(prediction_parts, axis=1)
        for item in range(len(groups)):
            accumulator.add(targets[item], batch_predictions[item], need_prediction)

        if batch_index % progress_every == 0 or batch_index == n_batches:
            elapsed = time.perf_counter() - started
            rate = batch_index / elapsed
            eta = (n_batches - batch_index) / max(rate, 1e-12)
            print(
                f"  valid batch {batch_index:,}/{n_batches:,} "
                f"| sequences={min(start + len(groups), len(valid_groups)):,}/{len(valid_groups):,} "
                f"| elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}"
            )

    result = accumulator.result()
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def save_checkpoint(
    path: Path,
    model: ConnectomeGRU,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model_config: ModelConfig,
    args: argparse.Namespace,
    epoch: int,
    train_loss: float,
    validation_result: dict,
    normalization_groups: list[int],
    train_groups: list[int],
    valid_groups: list[int],
) -> None:
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {
        "epoch": epoch,
        "model_config": asdict(model_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_loss": train_loss,
        "validation_result": validation_result,
        "normalization_groups": normalization_groups,
        "train_groups": train_groups,
        "valid_groups": valid_groups,
        "training_args": serializable_args,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if args.batch_size < 1 or args.chunk_length < 1:
        raise ValueError("batch-size and chunk-length must be positive")

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_groups, valid_groups, manifest = load_split(
        args.manifest,
        args.fold,
        args.max_train_sequences,
        args.max_valid_sequences,
        args.seed,
    )
    parquet = pq.ParquetFile(args.train)
    all_groups = train_groups + valid_groups
    if min(all_groups) < 0 or max(all_groups) >= parquet.num_row_groups:
        raise ValueError("fold manifest contains an invalid train row-group index")

    print("Alfa Connectome: GRU sequence-fold training")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Dataset row groups: {parquet.num_row_groups:,}")
    print(f"Manifest seed: {manifest['seed']}; holdout fold: {args.fold}")
    print(f"Train sequences: {len(train_groups):,}")
    print(f"Holdout sequences: {len(valid_groups):,}")
    print(
        f"Batch size: {args.batch_size}; chunk length: {args.chunk_length}; "
        f"epochs: {args.epochs}; patience: {args.patience}"
    )
    print(
        "Holdout WP is a train-fold proxy over every row after warm-up; "
        "it is not the official validation is_scored WP."
    )

    feature_mean, feature_std, normalization_groups = compute_normalization(
        parquet,
        train_groups,
        args.normalization_sequences,
        args.seed + args.fold,
    )
    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = ConnectomeGRU(model_config, feature_mean, feature_std).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}")

    best_wp = -math.inf
    epochs_without_improvement = 0
    training_started = time.perf_counter()
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(
            model,
            optimizer,
            scaler,
            parquet,
            train_groups,
            args.batch_size,
            args.chunk_length,
            device,
            epoch,
            args.seed,
            args.progress_every,
        )
        validation_result = evaluate_holdout(
            model,
            parquet,
            valid_groups,
            args.batch_size,
            args.chunk_length,
            device,
            args.progress_every,
        )
        wp = float(validation_result["weighted_pearson"])
        improved = wp > best_wp + 1e-6
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **validation_result,
            }
        )
        print(
            f"Epoch {epoch} result | train_loss={train_loss:.6f} "
            f"| t0={validation_result['t0']:.6f} "
            f"| t1={validation_result['t1']:.6f} | WP={wp:.6f}"
        )

        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scaler,
            model_config,
            args,
            epoch,
            train_loss,
            validation_result,
            normalization_groups,
            train_groups,
            valid_groups,
        )
        if improved:
            best_wp = wp
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scaler,
                model_config,
                args,
                epoch,
                train_loss,
                validation_result,
                normalization_groups,
                train_groups,
                valid_groups,
            )
            print(f"New best checkpoint: WP={best_wp:.6f}")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement}/{args.patience} epochs; "
                f"best WP={best_wp:.6f}"
            )

        history_path = args.output_dir / "history.json"
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break

    torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - training_started
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    print("\nTraining finished")
    print(f"Best holdout WP: {best_wp:.6f}")
    print(f"Elapsed: {format_duration(total_elapsed)}")
    print(f"Peak allocated GPU memory: {peak_memory_mb:.1f} MB")
    print(f"Best checkpoint: {args.output_dir / 'best.pt'}")
    print(f"History: {args.output_dir / 'history.json'}")


if __name__ == "__main__":
    main()
