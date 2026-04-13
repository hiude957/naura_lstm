import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import (
    dataframe_save,
    detect_column_groups,
    discover_raw_csvs,
    ensure_dirs_from_config,
    extract_day_id,
    load_config,
    mask_array_to_keys,
    save_json,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    ensure_dirs_from_config(cfg)
    seed_everything(cfg["seed"])

    raw_files = discover_raw_csvs(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    meta_dir = Path(cfg["paths"]["meta_dir"])

    schema = None
    recipe_map = {}
    next_recipe_id = 0

    for raw_path in tqdm(raw_files, desc="preprocess"):
        day_id = extract_day_id(raw_path)
        df = pd.read_csv(raw_path)
        if "timestamp" not in df.columns:
            raise ValueError(f"'timestamp' column not found in {raw_path}")

        if schema is None:
            groups = detect_column_groups(list(df.columns))
            schema = {
                "timestamp_col": "timestamp",
                **groups,
                "n_io": len(groups["io_flag_cols"]),
                "n_apc": len(groups["apc_cols"]),
            }
            save_json(schema, str(meta_dir / "column_schema.json"))
        else:
            groups = {
                "io_flag_cols": schema["io_flag_cols"],
                "io_value_cols": schema["io_value_cols"],
                "apc_cols": schema["apc_cols"],
                "mask_cols": schema["mask_cols"],
            }

        io_flag_cols = groups["io_flag_cols"]
        io_value_cols = groups["io_value_cols"]
        apc_cols = groups["apc_cols"]
        mask_cols = groups["mask_cols"]

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            continue

        df[io_flag_cols] = df[io_flag_cols].astype(np.float32)
        df[io_value_cols] = df[io_value_cols].astype(np.float32)
        df[apc_cols] = df[apc_cols].astype(np.float32)
        df[mask_cols] = df[mask_cols].astype(np.float32)

        dt_prev_sec = df["timestamp"].diff().dt.total_seconds().fillna(0.0).astype(np.float32)
        recipe_change_flag = (
            df[mask_cols].ne(df[mask_cols].shift(1)).any(axis=1).fillna(True).astype(np.int8)
        )
        recipe_change_flag.iloc[0] = 1

        new_session_flag = ((dt_prev_sec > float(cfg["data"]["session_gap_sec"])) | (recipe_change_flag == 1)).astype(np.int8)
        new_session_flag.iloc[0] = 1
        session_id = new_session_flag.cumsum().astype(np.int32) - 1

        dt_clip_sec = float(cfg["data"]["dt_clip_sec"])
        dt_prev_clipped = np.minimum(dt_prev_sec.to_numpy(dtype=np.float32), dt_clip_sec).astype(np.float32)
        dt_prev_norm = (dt_prev_clipped / max(dt_clip_sec, 1e-6)).astype(np.float32)
        dt_prev_log = np.log1p(dt_prev_clipped).astype(np.float32)

        mask_keys = mask_array_to_keys(df[mask_cols].to_numpy(dtype=np.uint8))
        recipe_ids = []
        for key in mask_keys:
            if key not in recipe_map:
                recipe_map[key] = next_recipe_id
                next_recipe_id += 1
            recipe_ids.append(recipe_map[key])

        processed = pd.concat(
            [
                pd.DataFrame({
                    "timestamp": df["timestamp"],
                    "timestamp_ns": df["timestamp"].astype("int64"),
                    "day_id": day_id,
                    "session_id": session_id,
                    "recipe_id": np.asarray(recipe_ids, dtype=np.int32),
                    "dt_prev_sec": dt_prev_sec.to_numpy(dtype=np.float32),
                    "dt_prev_norm": dt_prev_norm,
                    "dt_prev_log": dt_prev_log,
                    "recipe_change_flag": recipe_change_flag.to_numpy(dtype=np.int8),
                    "new_session_flag": new_session_flag.to_numpy(dtype=np.int8),
                }),
                df[io_flag_cols].reset_index(drop=True),
                df[io_value_cols].reset_index(drop=True),
                df[apc_cols].reset_index(drop=True),
                df[mask_cols].reset_index(drop=True),
            ],
            axis=1,
        )

        out_path = processed_dir / f"{day_id}_processed"
        dataframe_save(processed, str(out_path))

    save_json(recipe_map, str(meta_dir / "recipe_map.json"))
    print(f"Done. Processed files saved under: {processed_dir}")


if __name__ == "__main__":
    main()
