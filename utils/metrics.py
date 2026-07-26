from dataclasses import dataclass
from typing import Tuple

import torch


@torch.no_grad()
def batch_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float, float]:
    preds = (torch.sigmoid(logits) > threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    oa = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return float(iou), float(f1), float(oa)


@dataclass
class MetricTracker:
    loss: float = 0.0
    iou: float = 0.0
    f1: float = 0.0
    oa: float = 0.0
    n: int = 0

    def update(self, loss: float, iou: float, f1: float, oa: float, batch_size: int) -> None:
        self.loss += float(loss) * batch_size
        self.iou += float(iou) * batch_size
        self.f1 += float(f1) * batch_size
        self.oa += float(oa) * batch_size
        self.n += int(batch_size)

    def compute(self) -> Tuple[float, float, float, float]:
        if self.n == 0:
            return 0.0, 0.0, 0.0, 0.0
        return self.loss / self.n, self.iou / self.n, self.f1 / self.n, self.oa / self.n
