import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import EventSequenceDataset
from .model import EventDrivenAPCLSTM
from .utils import load_config, load_json

EPS = 1e-6
ERROR_PASS_THRESHOLD = 0.10
LATENCY_PASS_THRESHOLD_MS = 100.0
TOP_K_SUMMARY = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--target_horizon", type=int, default=1)
    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--timing_iters", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_channels")
    return parser


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is None or device_arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")
    return device


def torch_load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def add_batch_dim(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    batched = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batched[key] = value.unsqueeze(0)
        else:
            batched[key] = value
    return batched


def build_model(cfg: dict, schema: dict, recipe_map: dict) -> EventDrivenAPCLSTM:
    return EventDrivenAPCLSTM(
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
    )


def resolve_horizon_index(horizons_events: List[int], target_horizon: int) -> int:
    if target_horizon not in horizons_events:
        raise ValueError(
            f"target_horizon={target_horizon} is not in config.data.horizons_events={horizons_events}"
        )
    return horizons_events.index(target_horizon)


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup_iters < 0:
        raise ValueError("--warmup_iters must be >= 0")
    if args.timing_iters <= 0:
        raise ValueError("--timing_iters must be > 0")


@torch.inference_mode()
def collect_target_horizon_predictions(
    model: EventDrivenAPCLSTM,
    loader: DataLoader,
    device: torch.device,
    horizon_idx: int,
) -> Dict[str, np.ndarray]:
    model.eval()
    collected = {
        "delta_pred": [],
        "y_delta": [],
        "y_valid": [],
        "y_apc_now": [],
    }

    for batch in tqdm(loader, desc="eval_channels"):
        batch_dev = to_device(batch, device)
        outputs = model(batch_dev)

        collected["delta_pred"].append(outputs["delta_pred"][:, horizon_idx, :].detach().cpu().numpy())
        collected["y_delta"].append(batch["y_delta"][:, horizon_idx, :].cpu().numpy())
        collected["y_valid"].append(batch["y_valid"][:, horizon_idx, :].cpu().numpy())
        collected["y_apc_now"].append(batch["y_apc_now"].cpu().numpy())

    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def compute_channel_metrics(bundle: Dict[str, np.ndarray], apc_cols: List[str]) -> pd.DataFrame:
    abs_pred = bundle["y_apc_now"] + bundle["delta_pred"]
    abs_true = bundle["y_apc_now"] + bundle["y_delta"]
    rel_err = np.abs(abs_pred - abs_true) / np.maximum(np.abs(abs_true), EPS)
    valid = bundle["y_valid"] > 0.5

    records = []
    for ch_idx, ch_name in enumerate(apc_cols):
        valid_mask = valid[:, ch_idx]
        count = int(valid_mask.sum())

        if count == 0:
            mean_rel_err = np.nan
            p90_rel_err = np.nan
            max_rel_err = np.nan
            passed = False
        else:
            errs = rel_err[valid_mask, ch_idx]
            mean_rel_err = float(np.mean(errs))
            p90_rel_err = float(np.percentile(errs, 90))
            max_rel_err = float(np.max(errs))
            passed = mean_rel_err < ERROR_PASS_THRESHOLD

        records.append(
            {
                "channel_idx": ch_idx + 1,
                "channel_name": str(ch_name),
                "valid_count": count,
                "mean_rel_err": mean_rel_err,
                "p90_rel_err": p90_rel_err,
                "max_rel_err": max_rel_err,
                "pass_lt_10pct": passed,
            }
        )

    return pd.DataFrame.from_records(records)


def maybe_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure_forward_latency(
    model: EventDrivenAPCLSTM,
    sample: Dict[str, torch.Tensor],
    device: torch.device,
    warmup_iters: int,
    timing_iters: int,
) -> Tuple[pd.DataFrame, dict]:
    model.eval()
    batch = to_device(add_batch_dim(sample), device)

    for _ in range(warmup_iters):
        _ = model(batch)
    maybe_cuda_sync(device)

    records = []
    for iter_idx in range(1, timing_iters + 1):
        maybe_cuda_sync(device)
        start = time.perf_counter()
        _ = model(batch)
        maybe_cuda_sync(device)
        forward_ms = (time.perf_counter() - start) * 1000.0
        records.append(
            {
                "iter": iter_idx,
                "forward_ms": float(forward_ms),
                "pass_lt_100ms": bool(forward_ms < LATENCY_PASS_THRESHOLD_MS),
            }
        )

    latency_df = pd.DataFrame.from_records(records)
    latency_vals = latency_df["forward_ms"].to_numpy(dtype=np.float64)
    summary = {
        "device": str(device),
        "warmup_iters": int(warmup_iters),
        "timing_iters": int(timing_iters),
        "mean_ms": float(np.mean(latency_vals)),
        "p50_ms": float(np.percentile(latency_vals, 50)),
        "p90_ms": float(np.percentile(latency_vals, 90)),
        "max_ms": float(np.max(latency_vals)),
        "all_lt_100ms": bool(np.all(latency_vals < LATENCY_PASS_THRESHOLD_MS)),
    }
    return latency_df, summary


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.2f}%"


def format_ms(value: float) -> str:
    return f"{value:.3f} ms"


def render_channel_lines(df: pd.DataFrame, top_k: int = TOP_K_SUMMARY) -> List[str]:
    if df.empty:
        return ["- none"]

    lines = []
    for _, row in df.head(top_k).iterrows():
        lines.append(
            "- ch {channel_idx:03d} ({channel_name}): valid_count={valid_count}, "
            "mean={mean_rel_err}, p90={p90_rel_err}, max={max_rel_err}".format(
                channel_idx=int(row["channel_idx"]),
                channel_name=row["channel_name"],
                valid_count=int(row["valid_count"]),
                mean_rel_err=format_pct(row["mean_rel_err"]),
                p90_rel_err=format_pct(row["p90_rel_err"]),
                max_rel_err=format_pct(row["max_rel_err"]),
            )
        )
    return lines


def build_summary_markdown(
    metrics_df: pd.DataFrame,
    latency_summary: dict,
    target_horizon: int,
    horizon_idx: int,
    device: torch.device,
    output_dir: Path,
    plot_generated: bool,
) -> str:
    total_channels = int(len(metrics_df))
    passed_count = int(metrics_df["pass_lt_10pct"].sum())
    failed_df = metrics_df.loc[~metrics_df["pass_lt_10pct"]].copy()

    worst_mean_df = metrics_df.sort_values("mean_rel_err", ascending=False, na_position="last")
    p90_watch_df = metrics_df.loc[metrics_df["p90_rel_err"] >= ERROR_PASS_THRESHOLD].sort_values(
        "p90_rel_err", ascending=False, na_position="last"
    )
    max_watch_df = metrics_df.sort_values("max_rel_err", ascending=False, na_position="last")

    latency_gate = "PASS" if latency_summary["all_lt_100ms"] else "FAIL"
    channel_ratio = passed_count / max(total_channels, 1)

    lines = [
        "# Single-Horizon Evaluation Summary",
        "",
        "## Overview",
        f"- target_horizon: +{target_horizon} event(s)",
        f"- horizon_idx: {horizon_idx}",
        f"- device: {device}",
        f"- channel_pass: {passed_count}/{total_channels} ({channel_ratio * 100:.2f}%)",
        (
            f"- latency_gate: {latency_gate} "
            f"(max_ms={format_ms(latency_summary['max_ms'])}, threshold={LATENCY_PASS_THRESHOLD_MS:.0f} ms)"
        ),
        f"- latency_mean_p50_p90: {format_ms(latency_summary['mean_ms'])} / "
        f"{format_ms(latency_summary['p50_ms'])} / {format_ms(latency_summary['p90_ms'])}",
        "",
        f"## Failed Channels ({len(failed_df)})",
        *render_channel_lines(failed_df.sort_values('mean_rel_err', ascending=False, na_position='last')),
        "",
        f"## Worst Channels By mean_rel_err (top {TOP_K_SUMMARY})",
        *render_channel_lines(worst_mean_df),
        "",
        f"## Tail Watchlist By p90_rel_err >= {ERROR_PASS_THRESHOLD * 100:.0f}% ({len(p90_watch_df)})",
        *render_channel_lines(p90_watch_df),
        "",
        f"## Highest max_rel_err (top {TOP_K_SUMMARY})",
        *render_channel_lines(max_watch_df),
        "",
        "## Outputs",
        f"- channel_metrics.csv: {output_dir / 'channel_metrics.csv'}",
        f"- latency_samples.csv: {output_dir / 'latency_samples.csv'}",
        f"- latency_summary.json: {output_dir / 'latency_summary.json'}",
        f"- summary.md: {output_dir / 'summary.md'}",
        f"- channel_error.png: {output_dir / 'channel_error.png' if plot_generated else 'not generated'}",
    ]
    return "\n".join(lines) + "\n"


def maybe_make_channel_plot(metrics_df: pd.DataFrame, output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    plot_df = metrics_df.sort_values("mean_rel_err", ascending=False, na_position="last").copy()
    if plot_df.empty:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(plot_df))
    y = plot_df["mean_rel_err"].fillna(0.0).to_numpy(dtype=np.float64)
    labels = plot_df["channel_name"].astype(str).tolist()
    colors = ["#c0392b" if not bool(passed) else "#2e86c1" for passed in plot_df["pass_lt_10pct"].tolist()]

    plt.figure(figsize=(max(12, len(plot_df) * 0.18), 5.5))
    plt.bar(x, y, color=colors)
    plt.axhline(ERROR_PASS_THRESHOLD, color="#111111", linestyle="--", linewidth=1.2, label="10% threshold")
    plt.xlabel("APC channel")
    plt.ylabel("mean relative error")
    plt.title("Single-horizon APC channel mean relative error")
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def save_outputs(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    latency_summary: dict,
    summary_md: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "channel_metrics.csv", index=False)
    latency_df.to_csv(output_dir / "latency_samples.csv", index=False)
    with open(output_dir / "latency_summary.json", "w", encoding="utf-8") as f:
        json.dump(latency_summary, f, ensure_ascii=False, indent=2)
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    cfg = load_config(args.config)
    device = resolve_device(args.device)

    meta_dir = Path(cfg["paths"]["meta_dir"])
    schema_path = meta_dir / "column_schema.json"
    threshold_path = meta_dir / "threshold_tau.npy"
    recipe_map_path = meta_dir / "recipe_map.json"

    schema = load_json(str(schema_path))
    recipe_map = load_json(str(recipe_map_path))
    horizons_events = [int(x) for x in cfg["data"]["horizons_events"]]
    horizon_idx = resolve_horizon_index(horizons_events, args.target_horizon)

    dataset = EventSequenceDataset(
        manifest_path=args.manifest,
        schema_path=str(schema_path),
        threshold_path=str(threshold_path),
        max_files_in_memory=int(cfg["data"]["max_files_in_memory"]),
    )
    if len(dataset) == 0:
        raise ValueError("Evaluation dataset is empty.")

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["data"]["loader"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(cfg["data"]["loader"]["pin_memory"]) and device.type == "cuda",
        drop_last=False,
    )

    model = build_model(cfg, schema, recipe_map).to(device)
    checkpoint = torch_load_checkpoint(args.ckpt, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    bundle = collect_target_horizon_predictions(model, loader, device, horizon_idx)
    metrics_df = compute_channel_metrics(bundle, schema["apc_cols"])

    sample = dataset[0]
    latency_df, latency_summary = measure_forward_latency(
        model=model,
        sample=sample,
        device=device,
        warmup_iters=args.warmup_iters,
        timing_iters=args.timing_iters,
    )

    output_dir = Path(args.output_dir)
    plot_generated = maybe_make_channel_plot(metrics_df, output_dir / "channel_error.png")
    summary_md = build_summary_markdown(
        metrics_df=metrics_df,
        latency_summary=latency_summary,
        target_horizon=args.target_horizon,
        horizon_idx=horizon_idx,
        device=device,
        output_dir=output_dir,
        plot_generated=plot_generated,
    )
    save_outputs(
        output_dir=output_dir,
        metrics_df=metrics_df,
        latency_df=latency_df,
        latency_summary=latency_summary,
        summary_md=summary_md,
    )

    print(f"Output directory: {output_dir}")
    print(f"Channel pass: {int(metrics_df['pass_lt_10pct'].sum())}/{len(metrics_df)}")
    print(
        "Latency gate: "
        f"{'PASS' if latency_summary['all_lt_100ms'] else 'FAIL'} "
        f"(max_ms={latency_summary['max_ms']:.3f})"
    )


if __name__ == "__main__":
    main()
