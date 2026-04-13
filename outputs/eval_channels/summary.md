# Single-Horizon Evaluation Summary

## Overview
- target_horizon: +1 event(s)
- horizon_idx: 0
- device: cpu
- channel_pass: 80/150 (53.33%)
- latency_gate: PASS (max_ms=95.608 ms, threshold=100 ms)
- latency_mean_p50_p90: 41.932 ms / 7.527 ms / 90.119 ms

## Failed Channels (70)
- ch 101 (101): valid_count=441, mean=731416.46%, p90=293272.24%, max=100293218.75%
- ch 010 (10): valid_count=441, mean=619723.97%, p90=619775.88%, max=619823.10%
- ch 017 (17): valid_count=441, mean=611549.61%, p90=611620.12%, max=611698.14%
- ch 032 (32): valid_count=441, mean=574157.03%, p90=574210.45%, max=574314.11%
- ch 099 (99): valid_count=441, mean=562456.54%, p90=562543.26%, max=562588.92%
- ch 034 (34): valid_count=441, mean=509878.08%, p90=509948.68%, max=510046.14%
- ch 080 (80): valid_count=441, mean=467679.25%, p90=467705.86%, max=467744.09%
- ch 022 (22): valid_count=441, mean=451783.74%, p90=451868.07%, max=451966.55%
- ch 037 (37): valid_count=441, mean=445224.02%, p90=546779.54%, max=546870.02%
- ch 002 (2): valid_count=441, mean=395093.55%, p90=395183.64%, max=395293.26%

## Worst Channels By mean_rel_err (top 10)
- ch 101 (101): valid_count=441, mean=731416.46%, p90=293272.24%, max=100293218.75%
- ch 010 (10): valid_count=441, mean=619723.97%, p90=619775.88%, max=619823.10%
- ch 017 (17): valid_count=441, mean=611549.61%, p90=611620.12%, max=611698.14%
- ch 032 (32): valid_count=441, mean=574157.03%, p90=574210.45%, max=574314.11%
- ch 099 (99): valid_count=441, mean=562456.54%, p90=562543.26%, max=562588.92%
- ch 034 (34): valid_count=441, mean=509878.08%, p90=509948.68%, max=510046.14%
- ch 080 (80): valid_count=441, mean=467679.25%, p90=467705.86%, max=467744.09%
- ch 022 (22): valid_count=441, mean=451783.74%, p90=451868.07%, max=451966.55%
- ch 037 (37): valid_count=441, mean=445224.02%, p90=546779.54%, max=546870.02%
- ch 002 (2): valid_count=441, mean=395093.55%, p90=395183.64%, max=395293.26%

## Tail Watchlist By p90_rel_err >= 10% (65)
- ch 010 (10): valid_count=441, mean=619723.97%, p90=619775.88%, max=619823.10%
- ch 017 (17): valid_count=441, mean=611549.61%, p90=611620.12%, max=611698.14%
- ch 032 (32): valid_count=441, mean=574157.03%, p90=574210.45%, max=574314.11%
- ch 099 (99): valid_count=441, mean=562456.54%, p90=562543.26%, max=562588.92%
- ch 037 (37): valid_count=441, mean=445224.02%, p90=546779.54%, max=546870.02%
- ch 034 (34): valid_count=441, mean=509878.08%, p90=509948.68%, max=510046.14%
- ch 004 (4): valid_count=441, mean=338657.01%, p90=490621.88%, max=490676.12%
- ch 035 (35): valid_count=441, mean=356584.69%, p90=485988.72%, max=486063.48%
- ch 080 (80): valid_count=441, mean=467679.25%, p90=467705.86%, max=467744.09%
- ch 022 (22): valid_count=441, mean=451783.74%, p90=451868.07%, max=451966.55%

## Highest max_rel_err (top 10)
- ch 101 (101): valid_count=441, mean=731416.46%, p90=293272.24%, max=100293218.75%
- ch 020 (20): valid_count=441, mean=237952.27%, p90=1.10%, max=50292928.12%
- ch 021 (21): valid_count=441, mean=230052.17%, p90=0.58%, max=49880834.38%
- ch 005 (5): valid_count=441, mean=105594.12%, p90=26473.11%, max=36012834.38%
- ch 027 (27): valid_count=441, mean=214178.88%, p90=146575.93%, max=35839800.00%
- ch 019 (19): valid_count=441, mean=39126.04%, p90=218.93%, max=1090847.56%
- ch 010 (10): valid_count=441, mean=619723.97%, p90=619775.88%, max=619823.10%
- ch 017 (17): valid_count=441, mean=611549.61%, p90=611620.12%, max=611698.14%
- ch 032 (32): valid_count=441, mean=574157.03%, p90=574210.45%, max=574314.11%
- ch 099 (99): valid_count=441, mean=562456.54%, p90=562543.26%, max=562588.92%

## Outputs
- channel_metrics.csv: outputs/eval_channels/channel_metrics.csv
- latency_samples.csv: outputs/eval_channels/latency_samples.csv
- latency_summary.json: outputs/eval_channels/latency_summary.json
- summary.md: outputs/eval_channels/summary.md
- channel_error.png: outputs/eval_channels/channel_error.png
