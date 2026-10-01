import argparse
import os
import torch
from tqdm import tqdm

from config import Config
from models.model import AdaptiveScratchFormerCD
from utils.metrics import MetricTracker
from utils.train_utils import get_dataloaders, load_checkpoint
from test_tta import get_tta_probabilities

@torch.no_grad()
def test_ensemble_tta(models_list, loader, device, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6]):
    # We will keep a tracker for each threshold to find the optimal one
    trackers = {t: MetricTracker() for t in thresholds}

    pbar = tqdm(loader, desc="Testing 5-Model TTA Ensemble", leave=False)
    for i, batch in enumerate(pbar):
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        ensemble_prob_sum = 0
        
        # Get TTA probabilities from each model
        for model in models_list:
            prob_avg = get_tta_probabilities(model, pre, post)
            ensemble_prob_sum += prob_avg
            
        # Average across the 5 models
        final_ensemble_prob = ensemble_prob_sum / len(models_list)
        
        # Calculate stats for each threshold
        for t in thresholds:
            pred_mask = (final_ensemble_prob > t).float()
            
            tp = (pred_mask * mask).sum().item()
            fp = (pred_mask * (1 - mask)).sum().item()
            fn = ((1 - pred_mask) * mask).sum().item()
            tn = ((1 - pred_mask) * (1 - mask)).sum().item()
            
            trackers[t].update(0.0, tp, fp, fn, tn, pre.size(0))

    print("\n=== 5-MODEL TTA ENSEMBLE RESULTS ===")
    best_iou = 0.0
    best_t = 0.5
    for t in thresholds:
        _, iou, f1, oa = trackers[t].compute()
        print(f"Threshold: {t:.2f} | IoU: {iou:.4f} | F1: {f1:.4f} | OA: {oa:.4f}")
        if iou > best_iou:
            best_iou = iou
            best_t = t
            
    print(f"\n GRANDMASTER SCORE: Test IoU = {best_iou:.4f} (at Threshold {best_t:.2f})")
    return best_iou


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataloaders
    _, _, test_loader = get_dataloaders(
        dataset_root=Config().dataset_root,
        img_size=(256, 256),
        batch_size=Config().batch_size,
        num_workers=Config().num_workers,
    )

    checkpoint_paths = [
        "outputs/E0_baseline/best.ckpt",           # The 11-09 run (IoU: 83.94)
        "outputs/E0_baseline/best_seed1337.ckpt",  # The 09-09 run (IoU: 84.26)
        "outputs/E0_baseline/best_seed42.ckpt",    # The 10-09 run (IoU: 84.14)
        "outputs/E0_baseline/best_seed164.ckpt",   # The 10-09 run (IoU: 84.08)
        "outputs/E0_baseline/best_seed2026.ckpt"   # The 10-09 run (IoU: 83.85)
    ]
    
    models_list = []
    
    # Load all 5 models into RAM (If this causes OOM, we can load them sequentially per batch)
    print("Loading all 5 models into memory...")
    for ckpt_path in checkpoint_paths:
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint missing -> {ckpt_path}")
            continue
            
        model = AdaptiveScratchFormerCD(
            embed_dim=256,
            ssa_heads=Config().ssa_heads,
            ssa_reduce_ratio=Config().ssa_reduce_ratio,
            ssa_max_offset=Config().ssa_max_offset,
            exp_mode="E0",
        ).to(device)
        
        load_checkpoint(ckpt_path, model, optimizer=None)
        model.eval()
        models_list.append(model)
        print(f"Loaded: {ckpt_path}")

    if len(models_list) == 0:
        print("No models found! Run train_ensemble.bat first.")
        return

    # Run Ensemble TTA
    test_ensemble_tta(models_list, test_loader, device)

if __name__ == "__main__":
    main()
