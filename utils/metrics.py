from dataclasses import dataclass, field
from typing import Tuple

import torch


@torch.no_grad()
def batch_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float, float]:
    """Compute IoU, F1, OA for a single batch (used for progress bar display only)."""
    preds = (torch.sigmoid(logits) > threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    oa = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return float(iou), float(f1), float(oa)


@torch.no_grad()
def batch_counts_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float, float, float]:
    """Return raw TP, FP, FN, TN counts for global accumulation."""
    preds = (torch.sigmoid(logits) > threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

    return tp, fp, fn, tn


@dataclass
class MetricTracker:
    """Global metric tracker that accumulates raw TP/FP/FN/TN counts.
    
    Computes IoU, F1, OA globally over the entire dataset — matching
    how SOTA papers (BIT, ChangeFormer, SNUNet) report their metrics.
    """
    loss: float = 0.0
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0
    n: int = 0

    def update(self, loss: float, tp: float, fp: float, fn: float, tn: float, batch_size: int) -> None:
        self.loss += float(loss) * batch_size
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.n += int(batch_size)

    def compute(self) -> Tuple[float, float, float, float]:
        if self.n == 0:
            return 0.0, 0.0, 0.0, 0.0
        avg_loss = self.loss / self.n
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-8)
        f1 = 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn + 1e-8)
        oa = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + 1e-8)
        return avg_loss, iou, f1, oa

