#!/usr/bin/env bash
set -e

export WANDB_API_KEY="${WANDB_API_KEY:-}"

python -m src.preprocess --config configs/config.yaml
python -m src.build_manifest --config configs/config.yaml
python -m src.train --config configs/config.yaml
