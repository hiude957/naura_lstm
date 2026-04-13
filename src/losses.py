from typing import Dict, List

import torch
import torch.nn.functional as F


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def masked_bce_with_logits(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def masked_change_accuracy(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = (torch.sigmoid(logit) > 0.5).float()
    correct = (pred == target).float() * mask
    denom = mask.sum().clamp_min(1.0)
    return correct.sum() / denom


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    horizon_weights: List[float],
    cls_loss_weight: float,
) -> Dict[str, torch.Tensor]:
    delta_pred = outputs["delta_pred"]
    change_logit = outputs["change_logit"]
    y_delta = batch["y_delta"]
    y_valid = batch["y_valid"]
    y_change = batch["y_change"]

    device = delta_pred.device
    weights = torch.tensor(horizon_weights, dtype=delta_pred.dtype, device=device)
    weights = weights / weights.sum().clamp_min(1e-6)

    reg_losses, cls_losses, maes, accs = [], [], [], []
    for h in range(delta_pred.shape[1]):
        mask_h = y_valid[:, h, :]
        reg_h = masked_smooth_l1(delta_pred[:, h, :], y_delta[:, h, :], mask_h)
        cls_h = masked_bce_with_logits(change_logit[:, h, :], y_change[:, h, :], mask_h)
        mae_h = masked_mae(delta_pred[:, h, :], y_delta[:, h, :], mask_h)
        acc_h = masked_change_accuracy(change_logit[:, h, :], y_change[:, h, :], mask_h)
        reg_losses.append(reg_h)
        cls_losses.append(cls_h)
        maes.append(mae_h)
        accs.append(acc_h)

    reg_loss = sum(w * l for w, l in zip(weights, reg_losses))
    cls_loss = sum(w * l for w, l in zip(weights, cls_losses))
    mae = sum(w * l for w, l in zip(weights, maes))
    change_acc = sum(w * l for w, l in zip(weights, accs))
    total_loss = reg_loss + cls_loss_weight * cls_loss

    out = {
        "loss": total_loss,
        "reg_loss": reg_loss.detach(),
        "cls_loss": cls_loss.detach(),
        "mae": mae.detach(),
        "change_acc": change_acc.detach(),
    }
    for h, (r, c, m, a) in enumerate(zip(reg_losses, cls_losses, maes, accs)):
        out[f"reg_h{h}"] = r.detach()
        out[f"cls_h{h}"] = c.detach()
        out[f"mae_h{h}"] = m.detach()
        out[f"change_acc_h{h}"] = a.detach()
    return out
