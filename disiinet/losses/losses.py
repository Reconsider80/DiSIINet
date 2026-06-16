"""Joint training loss functions (Sec. 3.3)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiSIINetLoss(nn.Module):
    """
    L_overall = L_DDIM_DiEnh + beta * L_DDIM_DiSeg + L_enh + lambda * L_seg
    """

    def __init__(self, beta: float = 1.0, lambda_seg: float = 0.5):
        super().__init__()
        self.beta = beta
        self.lambda_seg = lambda_seg

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        x_gt: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l_ddim_enh = F.mse_loss(
            outputs["eps_enh_pred"], outputs["eps_enh_target"]
        )
        l_ddim_seg = F.mse_loss(
            outputs["eps_seg_pred"], outputs["eps_seg_target"]
        )
        l_enh = F.mse_loss(outputs["ip"], x_gt)

        # Weighted BCE for segmentation (Eq. 20)
        pos_weight = self._compute_pos_weight(mask_gt).view(-1, 1, 1)
        l_seg = F.binary_cross_entropy_with_logits(
            outputs["sp_logits"],
            mask_gt,
            pos_weight=pos_weight,
        )

        l_total = (
            l_ddim_enh
            + self.beta * l_ddim_seg
            + l_enh
            + self.lambda_seg * l_seg
        )

        return {
            "loss": l_total,
            "ddim_enh": l_ddim_enh,
            "ddim_seg": l_ddim_seg,
            "enh": l_enh,
            "seg": l_seg,
        }

    @staticmethod
    def _compute_pos_weight(mask: torch.Tensor) -> torch.Tensor:
        """Per-class positive weights for imbalanced segmentation."""
        pos = mask.sum(dim=(0, 2, 3))
        neg = mask.numel() // mask.shape[1] - pos
        weight = neg / (pos + 1e-6)
        return weight.clamp(max=10.0)
