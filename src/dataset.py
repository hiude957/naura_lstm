from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import DataFrameLRUCache, dataframe_load, load_json


class EventSequenceDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        schema_path: str,
        threshold_path: str,
        max_files_in_memory: int = 4,
    ) -> None:
        self.manifest = dataframe_load(manifest_path).reset_index(drop=True)
        self.schema = load_json(schema_path)
        self.taus = np.load(threshold_path).astype(np.float32)

        self.io_flag_cols = self.schema["io_flag_cols"]
        self.io_value_cols = self.schema["io_value_cols"]
        self.apc_cols = self.schema["apc_cols"]
        self.mask_cols = self.schema["mask_cols"]

        self.cache = DataFrameLRUCache(max_size=max_files_in_memory)
        self.num_horizons = len([c for c in self.manifest.columns if c.startswith("future_idx_")])

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.manifest.iloc[idx]
        df = self.cache.get(row["processed_path"])

        hist = df.iloc[int(row["hist_start_idx"]): int(row["hist_end_idx"]) + 1]
        anchor = df.iloc[int(row["anchor_idx"])]
        future_indices = [int(row[f"future_idx_{j}"]) for j in range(self.num_horizons)]
        future = df.iloc[future_indices]

        x_event = hist[self.io_flag_cols].to_numpy(dtype=np.float32)
        x_state = hist[self.io_value_cols].to_numpy(dtype=np.float32)
        x_apc = hist[self.apc_cols].to_numpy(dtype=np.float32)
        x_mask = hist[self.mask_cols].to_numpy(dtype=np.float32)
        x_apc = x_apc * x_mask

        x_time = hist[["dt_prev_norm", "dt_prev_log", "recipe_change_flag", "new_session_flag"]].to_numpy(dtype=np.float32)
        x_recipe_id = hist["recipe_id"].to_numpy(dtype=np.int64)

        apc_now = anchor[self.apc_cols].to_numpy(dtype=np.float32)
        mask_now = anchor[self.mask_cols].to_numpy(dtype=np.float32)
        future_apc = future[self.apc_cols].to_numpy(dtype=np.float32)
        future_mask = future[self.mask_cols].to_numpy(dtype=np.float32)

        y_delta = future_apc - apc_now[None, :]
        y_valid = (future_mask * mask_now[None, :]).astype(np.float32)
        y_change = (np.abs(y_delta) > self.taus[None, :]).astype(np.float32)

        future_ts = future["timestamp_ns"].to_numpy(dtype=np.int64)
        anchor_ts = np.int64(anchor["timestamp_ns"])
        day_id = np.int64(int(str(anchor["day_id"])))

        return {
            "x_event": torch.from_numpy(x_event),
            "x_state": torch.from_numpy(x_state),
            "x_apc": torch.from_numpy(x_apc),
            "x_mask": torch.from_numpy(x_mask),
            "x_time": torch.from_numpy(x_time),
            "x_recipe_id": torch.from_numpy(x_recipe_id),
            "y_delta": torch.from_numpy(y_delta),
            "y_valid": torch.from_numpy(y_valid),
            "y_change": torch.from_numpy(y_change),
            "y_apc_now": torch.from_numpy(apc_now.astype(np.float32)),
            "future_ts": torch.from_numpy(future_ts),
            "anchor_ts": torch.tensor(anchor_ts, dtype=torch.int64),
            "day_id": torch.tensor(day_id, dtype=torch.int64),
        }
