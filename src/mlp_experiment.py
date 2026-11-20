"""Compare current-feature and causal-history MLP models."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.signal import lfilter
from torch import nn

from train_gru_fold import (
    compute_normalization,
    load_split,
    seed_everything,
    weighted_smooth_l1,
)
from utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    GlobalAccumulator,
    validate_sequence,
)

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260928


class MLP(nn.Module):
    def __init__(self, mode):
        super().__init__()
        width = len(FEATURE_COLUMNS) * (5 if mode == "history" else 1)
        self.network = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return 2.0 * torch.tanh(self.network(x))


def read_sequence(parquet, group, official=False):
    columns = [
        "seq_ix", "step_in_seq", "need_prediction",
        *FEATURE_COLUMNS, *TARGET_COLUMNS,
    ]
    if official:
        columns.append("is_scored")
    table = parquet.read_row_group(group, columns=columns, use_threads=False)
    _, need = validate_sequence(table)
    x = np.column_stack(
        [table[c].to_numpy() for c in FEATURE_COLUMNS]
    ).astype(np.float32)
    y = np.column_stack(
        [table[c].to_numpy() for c in TARGET_COLUMNS]
    ).astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Nonfinite data in group {group}")
    mask = need.copy()
    if official:
        mask &= table["is_scored"].to_numpy().astype(bool)
    return x, y, mask


def make_features(x, mean, std, mode, rows=None):
    # Statistics come only from the training split.
    z = np.clip((x - mean) / std, -10.0, 10.0)
    indices = np.arange(len(z)) if rows is None else rows
    parts = [z[indices]]

    if mode == "history":
        for lag in (1, 20):
            delta = z[indices] - z[np.maximum(indices - lag, 0)]
            delta[indices < lag] = 0.0
            parts.append(delta)

        for span in (20, 100):
            alpha = 2.0 / (span + 1.0)
            # Initialize EMA with the first observation of this sequence.
            # lfilter runs the causal recurrence efficiently on CPU.
            ema, _ = lfilter(
                [alpha],
                [1.0, -(1.0 - alpha)],
                z,
                axis=0,
                zi=((1.0 - alpha) * z[0])[None, :],
            )
            parts.append(z[indices] - ema[indices])

    return np.clip(
        np.concatenate(parts, axis=1), -10.0, 10.0
    ).astype(np.float32)


@torch.inference_mode()
def evaluate(model, parquet, groups, mean, std, mode, device, official=False):
    model.eval()
    accumulator = GlobalAccumulator()
    for position, group in enumerate(groups, 1):
        x, y, mask = read_sequence(parquet, group, official)
        features = make_features(x, mean, std, mode)
        predictions = np.empty_like(y)
        for start in range(0, len(x), 8192):
            batch = torch.from_numpy(features[start:start + 8192]).to(device)
            predictions[start:start + 8192] = model(batch).cpu().numpy()
        accumulator.add(y, predictions, mask)
        if position % 200 == 0 or position == len(groups):
            print(f"  evaluated {position}/{len(groups)} sequences", flush=True)
    return accumulator.result()


def update_model(model, optimizer, features, targets, device):
    x = torch.from_numpy(np.concatenate(features)).to(device)
    y = torch.from_numpy(np.concatenate(targets)).to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = weighted_smooth_l1(model(x), y)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    return loss.item()


def train_epoch(model, optimizer, parquet, groups, mean, std, args, epoch, device):
    model.train()
    order = np.random.default_rng(SEED + epoch).permutation(groups)
    features, targets, losses = [], [], []

    for position, group in enumerate(order, 1):
        x, y, mask = read_sequence(parquet, int(group))
        eligible = np.flatnonzero(mask)
        # Same row selection for both MLP modes, regardless of their weights.
        rng = np.random.default_rng(
            np.random.SeedSequence([SEED, epoch, int(group)])
        )
        rows = rng.choice(
            eligible, min(args.rows_per_sequence, len(eligible)), replace=False
        )
        features.append(make_features(x, mean, std, args.mode, rows))
        targets.append(y[rows])

        # Mix sampled rows from eight sequences in each optimizer update.
        if len(features) == 8 or position == len(order):
            losses.append(
                update_model(model, optimizer, features, targets, device)
            )
            features, targets = [], []

        if position % 512 == 0 or position == len(order):
            print(
                f"  train {position}/{len(order)} sequences"
                f" | mean_batch_loss={np.mean(losses):.6f}",
                flush=True,
            )
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["current", "history"], default="current")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--rows-per-sequence", type=int, default=512)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.rows_per_sequence) < 1:
        raise ValueError("Epochs, patience and rows-per-sequence must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Activate the project environment with CUDA PyTorch")

    device = torch.device("cuda")
    run = args.run if args.run.is_absolute() else ROOT / args.run
    seed_everything(SEED)

    if args.evaluate:
        checkpoint = torch.load(
            run / "best.pt", map_location="cpu", weights_only=True
        )
        if checkpoint["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError("Feature order differs from checkpoint")
        mode = checkpoint["mode"]
        model = MLP(mode).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
        print(f"Official validation: mode={mode}, epoch={checkpoint['epoch']}")
        result = evaluate(
            model, parquet, list(range(parquet.num_row_groups)),
            checkpoint["mean"].numpy(), checkpoint["std"].numpy(),
            mode, device, official=True,
        )
        result.update({"mode": mode, "checkpoint_epoch": checkpoint["epoch"]})
        (run / "official_validation.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return

    # Prevent accidental overwriting of a completed experiment.
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Use a new output directory: {run}")
    run.mkdir(parents=True, exist_ok=True)
    train_groups, valid_groups, _ = load_split(
        ROOT / "sequence_folds.json", args.fold, 0, 0, SEED
    )
    parquet = pq.ParquetFile(ROOT / "datasets/train.parquet")
    print(
        f"Mode={args.mode}; train={len(train_groups)};"
        f" holdout={len(valid_groups)}; rows/sequence={args.rows_per_sequence}",
        flush=True,
    )
    mean, std, normalization_groups = compute_normalization(
        parquet, train_groups, 256, SEED + args.fold
    )
    model = MLP(args.mode).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    best, stale, history = -float("inf"), 0, []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        loss = train_epoch(
            model, optimizer, parquet, train_groups,
            mean, std, args, epoch, device,
        )
        result = evaluate(
            model, parquet, valid_groups, mean, std, args.mode, device
        )
        wp = result["weighted_pearson"]
        history.append({"epoch": epoch, "train_loss": loss, **result})
        print(
            f"loss={loss:.6f} | t0={result['t0']:.6f}"
            f" | t1={result['t1']:.6f} | proxy_WP={wp:.6f}",
            flush=True,
        )

        if wp > best + 1e-6:
            best, stale = wp, 0
            checkpoint = {
                "mode": args.mode, "epoch": epoch, "seed": SEED,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean": torch.from_numpy(mean),
                "std": torch.from_numpy(std),
                "feature_columns": FEATURE_COLUMNS,
                "train_groups": train_groups, "valid_groups": valid_groups,
                "normalization_groups": normalization_groups,
                "rows_per_sequence": args.rows_per_sequence,
                "validation_result": result,
            }
            temporary = run / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(run / "best.pt")
            print("Saved new best checkpoint")
        else:
            stale += 1

        (run / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if stale >= args.patience:
            print("Early stopping")
            break

    print(f"Best proxy WP: {best:.6f}")
    print(f"Checkpoint: {run / 'best.pt'}")


if __name__ == "__main__":
    main()