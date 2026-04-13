# Event-driven APC LSTM

这套代码按事件驱动 LSTM 来训练：
- 每一行就是一个 io_event 事件步
- 不做固定 100ms 重采样
- train / val 两部分
- 用 W&B 记录 loss 曲线
- 训练结束后自动导出 pred vs true 的时域对比图

## 目录

```text
configs/config.yaml
requirements.txt
src/
```

## 你的数据放这里

```bash
mkdir -p data/raw
# 把每天的 merged.csv 放进 data/raw/
```

## 运行

```bash
pip install -r requirements.txt
export WANDB_API_KEY="你的apikey"
python -m src.preprocess --config configs/config.yaml
python -m src.build_manifest --config configs/config.yaml
python -m src.train --config configs/config.yaml
```

## 验证集

在 `configs/config.yaml` 里修改：

```yaml
data:
  split:
    val_days: ["20260320"]
```

如果留空，就默认使用最新一天作为验证集。

## horizon 说明

```yaml
data:
  horizons_events: [1, 3, 5]
```

表示预测未来第 1 / 3 / 5 个事件的 APC 变化，不是秒。
