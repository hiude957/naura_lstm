import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import (
    dataframe_load,
    dataframe_save,
    ensure_dirs_from_config,
    extract_day_id,
    list_processed_files,
    load_config,
    load_json,
    resolve_train_val_days,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser


def build_manifest_rows(
    df: pd.DataFrame,
    processed_path: str,
    day_id: str,
    seq_len: int,
    horizons: List[int],
) -> List[Dict]:
    rows = []
    max_h = max(horizons)
    for session_id, session_df in df.groupby("session_id", sort=False):
        idx = session_df.index.to_numpy()
        if len(idx) < seq_len + max_h:
            continue
        for local_anchor in range(seq_len - 1, len(idx) - max_h):
            anchor_idx = int(idx[local_anchor])
            hist_start = int(idx[local_anchor - seq_len + 1])
            hist_end = anchor_idx
            future_idxs = [int(idx[local_anchor + h]) for h in horizons]
            row = {
                "processed_path": str(Path(processed_path).resolve()),
                "day_id": str(day_id),
                "session_id": int(session_id),
                "hist_start_idx": hist_start,
                "hist_end_idx": hist_end,
                "anchor_idx": anchor_idx,
            }
            for j, fidx in enumerate(future_idxs):
                row[f"future_idx_{j}"] = fidx
            rows.append(row)
    return rows


def compute_change_thresholds(
    processed_files: List[Path],
    train_days: List[str],
    apc_cols: List[str],
    mask_cols: List[str],
    horizons: List[int],
    quantile: float,
    min_threshold: float,
) -> np.ndarray:
    abs_values = [[] for _ in apc_cols]
    for path in tqdm(processed_files, desc="thresholds"):
        day_id = extract_day_id(path)
        if day_id not in train_days:
            continue
        df = dataframe_load(str(path))
        for _, session_df in df.groupby("session_id", sort=False):
            apc = session_df[apc_cols].to_numpy(dtype=np.float32)
            mask = session_df[mask_cols].to_numpy(dtype=np.float32)
            n = len(session_df)
            if n <= max(horizons):
                continue
            for h in horizons:
                if n <= h:
                    continue
                delta_abs = np.abs(apc[h:] - apc[:-h])
                valid = (mask[h:] * mask[:-h]) > 0.5
                for c in range(len(apc_cols)):
                    vals = delta_abs[:, c][valid[:, c]]
                    if vals.size > 0:
                        abs_values[c].append(vals)
    taus = np.full(len(apc_cols), min_threshold, dtype=np.float32)
    for c in range(len(apc_cols)):
        if abs_values[c]:
            merged = np.concatenate(abs_values[c], axis=0)
            taus[c] = max(float(np.quantile(merged, quantile)), min_threshold)
    return taus


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    ensure_dirs_from_config(cfg)
    processed_files = list_processed_files(cfg["paths"]["processed_dir"])
    manifest_dir = Path(cfg["paths"]["manifest_dir"])
    meta_dir = Path(cfg["paths"]["meta_dir"])

    schema = load_json(str(meta_dir / "column_schema.json"))
    train_days, val_days = resolve_train_val_days(
        processed_files,
        cfg["data"]["split"]["val_days"],
    )
    split_info = {"train_days": train_days, "val_days": val_days}
    save_json(split_info, str(meta_dir / "split_info.json"))

    seq_len = int(cfg["data"]["seq_len"])
    horizons = [int(x) for x in cfg["data"]["horizons_events"]]

    train_rows, val_rows = [], []
    for path in tqdm(processed_files, desc="manifest"):
        day_id = extract_day_id(path)
        df = dataframe_load(str(path))
        rows = build_manifest_rows(
            df=df,
            processed_path=str(path),
            day_id=day_id,
            seq_len=seq_len,
            horizons=horizons,
        )
        if day_id in train_days:
            train_rows.extend(rows)
        elif day_id in val_days:
            val_rows.extend(rows)

    if not train_rows:
        raise ValueError("No training samples were created")
    if not val_rows:
        raise ValueError("No validation samples were created")

    train_manifest = pd.DataFrame(train_rows)
    val_manifest = pd.DataFrame(val_rows)
    train_path = dataframe_save(train_manifest, str(manifest_dir / "train_manifest"))
    val_path = dataframe_save(val_manifest, str(manifest_dir / "val_manifest"))

    taus = compute_change_thresholds(
        processed_files=processed_files,
        train_days=train_days,
        apc_cols=schema["apc_cols"],
        mask_cols=schema["mask_cols"],
        horizons=horizons,
        quantile=float(cfg["data"]["thresholds"]["change_quantile"]),
        min_threshold=float(cfg["data"]["thresholds"]["min_threshold"]),
    )
    np.save(meta_dir / "threshold_tau.npy", taus)

    print(f"Train manifest: {train_path}")
    print(f"Val manifest:   {val_path}")
    print(f"Thresholds:     {meta_dir / 'threshold_tau.npy'}")


if __name__ == "__main__":
    main()
