import json
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs_from_config(cfg: dict) -> None:
    root = Path(".")
    keys = [
        "raw_dir",
        "processed_dir",
        "manifest_dir",
        "meta_dir",
        "output_dir",
        "ckpt_dir",
        "plot_dir",
    ]
    for k in keys:
        path = root / cfg["paths"][k]
        path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_day_id(path: Path) -> str:
    match = re.search(r"(\d{8})", path.name)
    if not match:
        raise ValueError(f"Cannot extract day id from filename: {path.name}")
    return match.group(1)


def discover_raw_csvs(raw_dir: str) -> List[Path]:
    paths = sorted(Path(raw_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    return paths


def detect_column_groups(columns: List[str]) -> Dict[str, List[str]]:
    io_flag_cols = []
    io_value_cols = []
    apc_cols = []
    mask_cols = []

    for c in columns:
        c_str = str(c)
        if re.fullmatch(r"io_\d{3}_flag", c_str):
            io_flag_cols.append(c_str)
        elif re.fullmatch(r"io_\d{3}_value", c_str):
            io_value_cols.append(c_str)
        elif re.fullmatch(r"mask_\d+", c_str):
            mask_cols.append(c_str)
        elif re.fullmatch(r"\d+", c_str):
            apc_cols.append(c_str)

    io_flag_cols = sorted(io_flag_cols, key=lambda x: int(re.search(r"(\d+)", x).group(1)))
    io_value_cols = sorted(io_value_cols, key=lambda x: int(re.search(r"(\d+)", x).group(1)))
    apc_cols = sorted(apc_cols, key=lambda x: int(x))
    mask_cols = sorted(mask_cols, key=lambda x: int(re.search(r"(\d+)", x).group(1)))

    if len(io_flag_cols) != len(io_value_cols):
        raise ValueError("io_flag count != io_value count")

    if not io_flag_cols or not io_value_cols or not apc_cols or not mask_cols:
        raise ValueError("Failed to detect column groups correctly")

    return {
        "io_flag_cols": io_flag_cols,
        "io_value_cols": io_value_cols,
        "apc_cols": apc_cols,
        "mask_cols": mask_cols,
    }


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dataframe_save(df: pd.DataFrame, path_no_ext: str) -> str:
    parquet_path = f"{path_no_ext}.parquet"
    pkl_path = f"{path_no_ext}.pkl"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        df.to_pickle(pkl_path)
        return pkl_path


def dataframe_load(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported dataframe format: {path}")


def list_processed_files(processed_dir: str) -> List[Path]:
    parquet_files = sorted(Path(processed_dir).glob("*.parquet"))
    pkl_files = sorted(Path(processed_dir).glob("*.pkl"))
    files = parquet_files if parquet_files else pkl_files
    if not files:
        raise FileNotFoundError(f"No processed files found under {processed_dir}")
    return files


def resolve_train_val_days(processed_files: List[Path], val_days_cfg: List[str]) -> Tuple[List[str], List[str]]:
    all_days = [extract_day_id(p) for p in processed_files]
    unique_days = sorted(set(all_days))
    if not unique_days:
        raise ValueError("No available day ids found")

    if val_days_cfg:
        val_days = sorted(set(val_days_cfg))
    else:
        val_days = [unique_days[-1]]

    train_days = [d for d in unique_days if d not in val_days]
    if not train_days:
        raise ValueError("Train days are empty after split")

    return train_days, val_days


def mask_array_to_keys(mask_array: np.ndarray) -> List[str]:
    packed = np.packbits(mask_array.astype(np.uint8), axis=1)
    return [row.tobytes().hex() for row in packed]


class DataFrameLRUCache:
    def __init__(self, max_size: int = 4):
        self.max_size = max_size
        self.cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def get(self, path: str) -> pd.DataFrame:
        path = str(path)
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        df = dataframe_load(path)
        self.cache[path] = df
        self.cache.move_to_end(path)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        return df
