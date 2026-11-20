"""Original fixed full-data GRU loop, decoupled from historical experiment guards."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import hashlib, json, os, random
import numpy as np
import torch
import train_gru_mse_fold as base
import train_gru_mse_ema_fold as trainer
SEED=20260928
SCHEDULE={"train":(0,4,.0003),"finetune":(4,1,.0001)}
def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def rng_state(device):
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])

def restore_rng(state, device):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved run")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])

def assert_checkpoint(saved, recipe, stage, mean, std, complete=False):
    offset, epochs, lr = SCHEDULE[stage]
    if saved.get("recipe") != recipe or saved.get("stage") != stage:
        raise ValueError("Checkpoint does not belong to this fixed full-data experiment")
    local = saved.get("local_epoch", 0)
    if not 1 <= local <= epochs or (complete and local != epochs) or saved["epoch"] != offset + local:
        raise ValueError("Wrong fixed checkpoint epoch")
    expected = dict(seed=SEED, epochs=epochs, learning_rate=lr, batch_size=16,
                    chunk_length=512, optimizer="adamw", weight_decay=.0001, ema_decay=.999,
                    loss_name="weighted_mse", initial_epoch=offset,
                    optimizer_initialization="fresh_from_ema_weights" if offset else "fresh_random_init")
    if saved.get("training_args") != expected:
        raise ValueError("Checkpoint training settings changed")
    if (saved.get("weights_source") != "ema" or saved["model_config"] != recipe["model_config"] or
            saved["train_groups"] != recipe["train_groups"] or saved["valid_groups"] != [] or
            saved["normalization_groups"] != recipe["normalization_groups"] or
            saved.get("validation_result") is not None or saved["ema_decay"] != .999):
        raise ValueError("Invalid full-data checkpoint contract")
    for state_key in ("model_state_dict", "raw_model_state_dict"):
        state = saved[state_key]
        for key, array in (("feature_mean", mean), ("feature_std", std)):
            torch.testing.assert_close(state[key].cpu().reshape(-1), torch.from_numpy(array), atol=0, rtol=0)
    if any(g["lr"] != lr or g["weight_decay"] != .0001 for g in saved["optimizer_state_dict"]["param_groups"]):
        raise ValueError("Optimizer schedule changed")

@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+b") as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            lock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise RuntimeError("Another process is already using this training directory") from exc
        try:
            yield
        finally:
            stream.seek(0)
            unlock()

def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

def run_stage(recipe, mean, std, parquet, stage, output, epochs_this_run, device, initial_path=None):
    if epochs_this_run < 1:
        raise ValueError("epochs-this-run must be positive")
    offset, epochs, lr = SCHEDULE[stage]
    initial = None
    if stage == "finetune":
        if initial_path is None:
            raise ValueError("Full-data base epoch 4 is required")
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        assert_checkpoint(initial, recipe, "train", mean, std, complete=True)
    elif initial_path is not None:
        raise ValueError("Base training must start from random weights")
    initialization = None if initial is None else dict(path=str(initial_path.resolve()), sha256=sha256(initial_path))
    contract = dict(recipe=recipe, stage=stage, initialization=initialization)
    with run_lock(output):
        plan_path = output / "run.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding="utf-8")) != contract:
                raise ValueError("Existing run has a different recipe/source; refusing to overwrite")
        else:
            if any(p.name not in ("run.lock", "run.json.partial") for p in output.iterdir()):
                raise ValueError("Refusing to use a nonempty, unidentified output directory")
            atomic_json(plan_path, contract)
        paths = [output / f"epoch_{offset + i:02d}.pt" for i in range(1, epochs + 1)]
        existing = [p for p in paths if p.exists()]
        if existing != paths[:len(existing)] or set(output.glob("*.pt")) != set(existing):
            raise ValueError("Unexpected checkpoint or gap in epoch history")
        completed = len(existing)
        if completed == epochs:
            assert_checkpoint(torch.load(paths[-1], map_location="cpu", weights_only=True),
                              recipe, stage, mean, std, complete=True)
            print(f"Stage already complete: {paths[-1]}; no additional epochs.", flush=True)
            return paths[-1]
        previous = torch.load(existing[-1], map_location="cpu", weights_only=True) if existing else None
        if previous is not None:
            assert_checkpoint(previous, recipe, stage, mean, std)
            if previous["local_epoch"] != completed or previous["initialization"] != initialization:
                raise ValueError("Resume checkpoint/source mismatch")
        # Fine-tuning restarts RNG, optimizer, scaler and EMA, as in the reference recipe.
        base.seed_everything(SEED)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        model = base.ConnectomeGRU(base.ModelConfig(**recipe["model_config"]), mean, std).to(device)
        if initial is not None:
            model.load_state_dict(initial["model_state_dict"], strict=True)
        args = SimpleNamespace(optimizer="adamw", learning_rate=lr, weight_decay=.0001)
        optimizer = trainer.make_optimizer(model, args)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        ema = trainer.EMAWeights(model, .999)
        if previous is not None:
            model.load_state_dict(previous["raw_model_state_dict"], strict=True)
            ema.model.load_state_dict(previous["model_state_dict"], strict=True)
            ema.updates = previous["ema_updates"]
            optimizer.load_state_dict(previous["optimizer_state_dict"])
            scaler.load_state_dict(previous["scaler_state_dict"])
            restore_rng(previous["rng_state"], device)
        hook = ema.attach(optimizer, model)
        try:
            for local_epoch in range(completed + 1, min(epochs, completed + epochs_this_run) + 1):
                epoch = offset + local_epoch
                print(f"\n{stage}: fixed epoch {epoch}/5, stage {local_epoch}/{epochs}, "
                      f"LR={lr}, sequences={len(recipe['train_groups'])}; no validation selection", flush=True)
                loss = base.train_epoch(model=model, optimizer=optimizer, scaler=scaler, parquet=parquet,
                                        train_groups=recipe["train_groups"], batch_size=16, chunk_length=512,
                                        device=device, epoch=epoch, seed=SEED, progress_every=20)
                if not np.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss; checkpoint not committed")
                saved = dict(recipe=recipe, stage=stage, initialization=initialization,
                             epoch=epoch, local_epoch=local_epoch, model_config=recipe["model_config"],
                             model_state_dict=ema.model.state_dict(), weights_source="ema",
                             raw_model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                             scaler_state_dict=scaler.state_dict(), ema_updates=ema.updates, ema_decay=.999,
                             rng_state=rng_state(device), train_loss=float(loss), validation_result=None,
                             train_groups=recipe["train_groups"], valid_groups=[],
                             normalization_groups=recipe["normalization_groups"],
                             training_args=dict(seed=SEED, epochs=epochs, learning_rate=lr, batch_size=16,
                                                chunk_length=512, optimizer="adamw", weight_decay=.0001,
                                                ema_decay=.999, loss_name="weighted_mse", initial_epoch=offset,
                                                optimizer_initialization="fresh_from_ema_weights" if offset else "fresh_random_init"))
                assert_checkpoint(saved, recipe, stage, mean, std)
                path = paths[local_epoch - 1]
                if path.exists():
                    raise FileExistsError(path)
                temporary = path.with_suffix(".pt.partial")
                torch.save(saved, temporary)
                temporary.replace(path)
                print(f"Saved complete epoch: {path}; EMA updates={ema.updates}", flush=True)
        finally:
            hook.remove()
        print("Stopped at epoch boundary. No next stage, validation, export or submission started.", flush=True)
        return path
