import argparse
import math
import os
from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None

from .dataset import EventSequenceDataset
from .losses import compute_losses
from .model import EventDrivenAPCLSTM
from .plot_utils import make_time_domain_plots, collect_predictions
from .utils import ensure_dirs_from_config, load_config, load_json, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def mean_dict(dicts):
    out = {}
    keys = dicts[0].keys()
    for k in keys:
        vals = [float(d[k]) for d in dicts]
        out[k] = sum(vals) / max(len(vals), 1)
    return out


def run_epoch(model, loader, optimizer, scaler, device, cfg, train: bool):
    model.train(train)
    epoch_stats = []
    pbar = tqdm(loader, desc="train" if train else "val")
    for batch in pbar:
        batch = to_device(batch, device)
        with torch.set_grad_enabled(train):
            with autocast(enabled=(device.type == "cuda" and bool(cfg["train"]["amp"]))):
                outputs = model(batch)
                stats = compute_losses(
                    outputs=outputs,
                    batch=batch,
                    horizon_weights=cfg["train"]["horizon_loss_weights"],
                    cls_loss_weight=float(cfg["train"]["cls_loss_weight"]),
                )
                loss = stats["loss"]

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()

        detached = {k: float(v.detach().cpu().item()) for k, v in stats.items()}
        epoch_stats.append(detached)
        pbar.set_postfix({
            "loss": f"{detached['loss']:.4f}",
            "mae": f"{detached['mae']:.4f}",
            "acc": f"{detached['change_acc']:.4f}",
        })
    return mean_dict(epoch_stats)


def init_wandb(cfg):
    if not cfg["wandb"]["enabled"]:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Please `pip install wandb` or set wandb.enabled=false.")
    api_key = os.getenv("WANDB_API_KEY", "")
    if api_key:
        wandb.login(key=api_key, relogin=True)
    else:
        wandb.login()
    return wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        name=cfg["wandb"]["run_name"],
        config=cfg,
    )


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, cfg):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": cfg,
    }
    torch.save(payload, path)


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    ensure_dirs_from_config(cfg)
    seed_everything(cfg["seed"])

    root = Path(".")
    manifest_dir = root / cfg["paths"]["manifest_dir"]
    meta_dir = root / cfg["paths"]["meta_dir"]
    ckpt_dir = root / cfg["paths"]["ckpt_dir"]
    plot_dir = root / cfg["paths"]["plot_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    schema = load_json(str(meta_dir / "column_schema.json"))
    recipe_map = load_json(str(meta_dir / "recipe_map.json"))

    train_manifest_path = None
    val_manifest_path = None
    for name in ["train_manifest.parquet", "train_manifest.pkl"]:
        p = manifest_dir / name
        if p.exists():
            train_manifest_path = str(p)
            break
    for name in ["val_manifest.parquet", "val_manifest.pkl"]:
        p = manifest_dir / name
        if p.exists():
            val_manifest_path = str(p)
            break
    if train_manifest_path is None or val_manifest_path is None:
        raise FileNotFoundError("Manifest not found. Please run preprocess and build_manifest first.")

    threshold_path = str(meta_dir / "threshold_tau.npy")
    train_ds = EventSequenceDataset(train_manifest_path, str(meta_dir / "column_schema.json"), threshold_path, int(cfg["data"]["max_files_in_memory"]))
    val_ds = EventSequenceDataset(val_manifest_path, str(meta_dir / "column_schema.json"), threshold_path, int(cfg["data"]["max_files_in_memory"]))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["data"]["loader"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["loader"]["num_workers"]),
        pin_memory=bool(cfg["data"]["loader"]["pin_memory"]),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["data"]["loader"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["loader"]["num_workers"]),
        pin_memory=bool(cfg["data"]["loader"]["pin_memory"]),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventDrivenAPCLSTM(
        n_io=int(schema["n_io"]),
        n_apc=int(schema["n_apc"]),
        n_recipes=len(recipe_map),
        recipe_emb_dim=int(cfg["model"]["recipe_emb_dim"]),
        event_hidden_dim=int(cfg["model"]["event_hidden_dim"]),
        state_hidden_dim=int(cfg["model"]["state_hidden_dim"]),
        lstm_hidden_dim=int(cfg["model"]["lstm_hidden_dim"]),
        lstm_layers=int(cfg["model"]["lstm_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        tail_pool_k=int(cfg["model"]["tail_pool_k"]),
        num_horizons=len(cfg["data"]["horizons_events"]),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda" and bool(cfg["train"]["amp"])))
    run = init_wandb(cfg)

    best_val_loss = math.inf
    best_epoch = -1
    patience = int(cfg["train"]["early_stopping_patience"])

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_stats = run_epoch(model, train_loader, optimizer, scaler, device, cfg, train=True)
        val_stats = run_epoch(model, val_loader, optimizer, scaler, device, cfg, train=False)
        scheduler.step(val_stats["loss"])

        lr_now = optimizer.param_groups[0]["lr"]
        log_dict = {"epoch": epoch, "lr": lr_now}
        log_dict.update({f"train/{k}": v for k, v in train_stats.items()})
        log_dict.update({f"val/{k}": v for k, v in val_stats.items()})
        print(log_dict)

        if run is not None:
            wandb.log(log_dict, step=epoch)

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, best_val_loss, cfg)
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, best_val_loss, cfg)

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    bundle = collect_predictions(model, val_loader, device=device)
    plot_paths = make_time_domain_plots(
        bundle=bundle,
        horizons_events=[int(x) for x in cfg["data"]["horizons_events"]],
        save_dir=str(plot_dir),
        top_k_channels=int(cfg["plots"]["top_k_channels"]),
        max_points_per_plot=int(cfg["plots"]["max_points_per_plot"]),
        selected_channels=cfg["plots"]["selected_channels"],
    )

    if run is not None:
        for path in plot_paths:
            wandb.log({f"plot/{Path(path).stem}": wandb.Image(path)})
        wandb.finish()

    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Plots saved under: {plot_dir}")


if __name__ == "__main__":
    main()
