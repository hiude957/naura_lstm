from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    collected = {
        "delta_pred": [],
        "y_delta": [],
        "y_valid": [],
        "y_apc_now": [],
        "future_ts": [],
    }

    for batch in tqdm(loader, desc="collect_pred"):
        batch_dev = to_device(batch, device)
        outputs = model(batch_dev)

        collected["delta_pred"].append(outputs["delta_pred"].detach().cpu().numpy())
        collected["y_delta"].append(batch["y_delta"].cpu().numpy())
        collected["y_valid"].append(batch["y_valid"].cpu().numpy())
        collected["y_apc_now"].append(batch["y_apc_now"].cpu().numpy())
        collected["future_ts"].append(batch["future_ts"].cpu().numpy())

    for k in collected:
        collected[k] = np.concatenate(collected[k], axis=0)

    collected["abs_pred"] = collected["y_apc_now"][:, None, :] + collected["delta_pred"]
    collected["abs_true"] = collected["y_apc_now"][:, None, :] + collected["y_delta"]
    return collected


def select_channels(bundle: Dict[str, np.ndarray], top_k: int, selected_channels: Optional[List[int]] = None) -> List[int]:
    if selected_channels:
        return [int(c) - 1 for c in selected_channels]
    valid = bundle["y_valid"] > 0.5
    delta = np.abs(bundle["y_delta"]) * valid
    score = delta.sum(axis=(0, 1)) / np.maximum(valid.sum(axis=(0, 1)), 1.0)
    order = np.argsort(-score)
    return order[:top_k].tolist()


def downsample_for_plot(x, y1, y2, max_points: int):
    n = len(x)
    if n <= max_points:
        return x, y1, y2
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return x[idx], y1[idx], y2[idx]


def make_time_domain_plots(
    bundle: Dict[str, np.ndarray],
    horizons_events: List[int],
    save_dir: str,
    top_k_channels: int = 6,
    max_points_per_plot: int = 800,
    selected_channels: Optional[List[int]] = None,
) -> List[str]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    channels = select_channels(bundle, top_k=top_k_channels, selected_channels=selected_channels)
    saved_paths = []

    for h_idx, horizon in enumerate(horizons_events):
        for ch in channels:
            valid = bundle["y_valid"][:, h_idx, ch] > 0.5
            if valid.sum() < 3:
                continue

            ts = pd.to_datetime(bundle["future_ts"][:, h_idx][valid])
            pred = bundle["abs_pred"][:, h_idx, ch][valid]
            true = bundle["abs_true"][:, h_idx, ch][valid]

            order = np.argsort(ts.values.astype("int64"))
            ts = ts.values[order]
            pred = pred[order]
            true = true[order]

            ts, pred, true = downsample_for_plot(ts, pred, true, max_points=max_points_per_plot)

            plt.figure(figsize=(12, 4.5))
            plt.plot(ts, true, label="true")
            plt.plot(ts, pred, label="pred")
            plt.xlabel("future event timestamp")
            plt.ylabel("APC absolute value")
            plt.title(f"APC channel {ch + 1} | horizon +{horizon} events")
            plt.legend()
            plt.tight_layout()

            out_path = save_dir / f"apc_ch_{ch+1:03d}_horizon_{horizon}.png"
            plt.savefig(out_path, dpi=180)
            plt.close()
            saved_paths.append(str(out_path))

    return saved_paths
