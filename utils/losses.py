import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Cast to float32 to prevent FP16 overflow during summation.
    # An image is 256x256 = 65536 pixels. FP16 max value is 65504.
    # If the model predicts mostly 1s, sum() will overflow to inf in FP16!
    probs = torch.sigmoid(logits).to(torch.float32)
    probs = probs.contiguous()
    targets = targets.to(torch.float32).contiguous()

    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    
    # Cast back to the original dtype (e.g. float16 if using AMP) to maintain type consistency
    return (1.0 - dice.mean()).to(logits.dtype)


class FocalLoss(nn.Module):
    """Focal Loss for dense binary classification (Lin et al., 2017).

    Down-weights well-classified pixels so the network focuses on hard examples
    (boundary pixels, small change regions). This directly addresses the
    boundary-blur problem by making the loss boundary-aware.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Balancing factor for positive class. Default 0.25.
        gamma: Focusing parameter. Higher = more focus on hard examples. Default 2.0.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.clamp(torch.exp(-bce_loss), min=1e-6, max=1.0 - 1e-6)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice + Focal Loss.

    - BCE: pixel-level cross entropy, provides stable gradients everywhere.
    - Dice: region-level overlap, handles class imbalance.
    - Focal: focuses on hard boundary pixels, prevents over-smooth predictions.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, focal_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        d = dice_loss_from_logits(logits, targets)
        f = self.focal(logits, targets)
        return self.bce_weight * bce + self.dice_weight * d + self.focal_weight * f

