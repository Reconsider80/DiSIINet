"""Evaluation metrics for enhancement and segmentation."""

import numpy as np
import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val**2 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return 0.0

    p = pred.squeeze().cpu().numpy()
    t = target.squeeze().cpu().numpy()
    if p.ndim == 3:
        return float(
            np.mean([structural_similarity(p[i], t[i], data_range=1.0) for i in range(p.shape[0])])
        )
    return float(structural_similarity(p, t, data_range=1.0))


def dice_coefficient(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> float:
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    if union == 0:
        return 1.0
    return (2.0 * intersection / union).item()


def mean_iou(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, threshold: float = 0.5
) -> float:
    ious = []
    pred_bin = (pred > threshold).float()
    for c in range(1, num_classes):
        p = pred_bin[:, c]
        t = target[:, c]
        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection
        if union > 0:
            ious.append((intersection / union).item())
    return float(np.mean(ious)) if ious else 0.0


def evaluate_batch(
    outputs: dict[str, torch.Tensor],
    x_gt: torch.Tensor,
    mask_gt: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    return {
        "psnr": psnr(outputs["ip"], x_gt),
        "ssim": ssim(outputs["ip"], x_gt),
        "dice": dice_coefficient(outputs["sp"][:, 1:], mask_gt[:, 1:]),
        "miou": mean_iou(outputs["sp"], mask_gt, num_classes),
    }
