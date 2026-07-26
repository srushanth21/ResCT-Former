import os
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.levir_loader import LEVIRCDDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(
    dataset_root: str,
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = LEVIRCDDataset(dataset_root, split="train", img_size=img_size, augment=True)
    val_ds = LEVIRCDDataset(dataset_root, split="val", img_size=img_size, augment=False)
    test_ds = LEVIRCDDataset(dataset_root, split="test", img_size=img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_iou: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_iou": float(best_iou),
        },
        path,
    )


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def _to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().clamp(0, 1).numpy()
    x = (x * 255.0).astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))
    return x


def _to_uint8_mask(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().numpy()
    x = (x > 0.5).astype(np.uint8) * 255
    return x


def save_change_overlay(
    pre: torch.Tensor,
    post: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_mask: Optional[torch.Tensor],
    out_path: str,
    alpha: float = 0.5,
) -> None:
    """Save a 5-panel visualization: pre, post, gt, pred, overlay."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pre_img = _to_uint8_rgb(pre)
    post_img = _to_uint8_rgb(post)

    pred = _to_uint8_mask(pred_mask.squeeze(0))
    if gt_mask is not None:
        gt = _to_uint8_mask(gt_mask.squeeze(0))
    else:
        gt = np.zeros_like(pred)

    # overlay on post
    overlay = post_img.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255

    pred_bool = pred > 0
    overlay[pred_bool] = (alpha * red[pred_bool] + (1 - alpha) * overlay[pred_bool]).astype(np.uint8)

    # Create a simple horizontal montage using cv2
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
    pred_rgb = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)

    montage = np.concatenate([pre_img, post_img, gt_rgb, pred_rgb, overlay], axis=1)
    montage = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, montage)
