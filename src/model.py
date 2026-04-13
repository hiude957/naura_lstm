from typing import Dict

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EventDrivenAPCLSTM(nn.Module):
    def __init__(
        self,
        n_io: int,
        n_apc: int,
        n_recipes: int,
        recipe_emb_dim: int,
        event_hidden_dim: int,
        state_hidden_dim: int,
        lstm_hidden_dim: int,
        lstm_layers: int,
        dropout: float,
        tail_pool_k: int,
        num_horizons: int,
    ) -> None:
        super().__init__()
        self.n_apc = n_apc
        self.num_horizons = num_horizons
        self.tail_pool_k = tail_pool_k

        self.recipe_emb = nn.Embedding(max(n_recipes, 1) + 1, recipe_emb_dim)
        event_in_dim = n_io + 4
        state_in_dim = n_io + n_apc + n_apc + recipe_emb_dim
        self.event_proj = MLPBlock(event_in_dim, event_hidden_dim, dropout)
        self.state_proj = MLPBlock(state_in_dim, state_hidden_dim, dropout)

        lstm_input_dim = event_hidden_dim + state_hidden_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        context_dim = lstm_hidden_dim * 3
        self.reg_head = nn.Sequential(
            nn.Linear(context_dim, lstm_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, num_horizons * n_apc),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(context_dim, lstm_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, num_horizons * n_apc),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_event = batch["x_event"]
        x_state = batch["x_state"]
        x_apc = batch["x_apc"]
        x_mask = batch["x_mask"]
        x_time = batch["x_time"]
        x_recipe_id = batch["x_recipe_id"]

        recipe_emb = self.recipe_emb(x_recipe_id)
        event_feat = torch.cat([x_event, x_time], dim=-1)
        state_feat = torch.cat([x_state, x_apc, x_mask, recipe_emb], dim=-1)

        event_z = self.event_proj(event_feat)
        state_z = self.state_proj(state_feat)
        x = torch.cat([event_z, state_z], dim=-1)

        out, _ = self.lstm(x)
        last = out[:, -1, :]
        k = min(self.tail_pool_k, out.shape[1])
        tail = out[:, -k:, :]
        tail_mean = tail.mean(dim=1)
        tail_max = tail.max(dim=1).values
        context = torch.cat([last, tail_mean, tail_max], dim=-1)

        delta_pred = self.reg_head(context).view(-1, self.num_horizons, self.n_apc)
        change_logit = self.cls_head(context).view(-1, self.num_horizons, self.n_apc)

        return {
            "delta_pred": delta_pred,
            "change_logit": change_logit,
        }
