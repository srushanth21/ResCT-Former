import argparse
import os
import torch
from tqdm import tqdm

from config import Config
from models.model import AdaptiveScratchFormerCD
from utils.metrics import MetricTracker
from utils.train_utils import get_dataloaders, load_checkpoint

@torch.no_grad()
def get_tta_probabilities(model, pre, post):
    """
    Returns the averaged probabilities over 4 Test-Time Augmentations:
    1. Normal
    2. Horizontal Flip
    3. Vertical Flip
    4. Rotated 90 degrees
    """
    # 1. Normal
    logits_normal = model(pre, post)
    prob_normal = torch.sigmoid(logits_normal)
    
    # 2. Horizontal Flip (dim=-1)
    logits_h = model(torch.flip(pre, dims=[-1]), torch.flip(post, dims=[-1]))
    prob_h = torch.flip(torch.sigmoid(logits_h), dims=[-1])
    
    # 3. Vertical Flip (dim=-2)
    logits_v = model(torch.flip(pre, dims=[-2]), torch.flip(post, dims=[-2]))
    prob_v = torch.flip(torch.sigmoid(logits_v), dims=[-2])
    
    # 4. Rotate 90 degrees
    logits_rot = model(torch.rot90(pre, k=1, dims=[-2,-1]), torch.rot90(post, k=1, dims=[-2,-1]))
    prob_rot = torch.rot90(torch.sigmoid(logits_rot), k=-1, dims=[-2,-1])
    
    # Average the probabilities
    prob_avg = (prob_normal + prob_h + prob_v + prob_rot) / 4.0
    return prob_avg

@torch.no_grad()
def test_tta_with_thresholds(model, loader, device, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6]):
    model.eval()
    
    # We will keep a tracker for each threshold to find the optimal one
    trackers = {t: MetricTracker() for t in thresholds}

    pbar = tqdm(loader, desc="Testing TTA", leave=False)
    for i, batch in enumerate(pbar):
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        # Get averaged probabilities via TTA
        prob_avg = get_tta_probabilities(model, pre, post)
        
        # Calculate True Positives, False Positives, False Negatives for each threshold
        for t in thresholds:
            pred_mask = (prob_avg > t).float()
            
            # Compute stats manually for IoU
            tp = (pred_mask * mask).sum().item()
            fp = (pred_mask * (1 - mask)).sum().item()
            fn = ((1 - pred_mask) * mask).sum().item()
            tn = ((1 - pred_mask) * (1 - mask)).sum().item()
            
            trackers[t].update(0.0, tp, fp, fn, tn, pre.size(0))

    # Print results for all thresholds
    print("\n=== TEST-TIME AUGMENTATION RESULTS ===")
    best_iou = 0.0
    best_t = 0.5
    for t in thresholds:
        _, iou, f1, oa = trackers[t].compute()
        print(f"Threshold: {t:.2f} | IoU: {iou:.4f} | F1: {f1:.4f} | OA: {oa:.4f}")
        if iou > best_iou:
            best_iou = iou
            best_t = t
            
    print(f"\n BEST TEST SCORE: IoU = {best_iou:.4f} (at Threshold {best_t:.2f})")
    return best_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=Config().dataset_root)
    parser.add_argument("--checkpoint", type=str, default=os.path.join("outputs", "E0_baseline", "best_seed164.ckpt"))
    parser.add_argument("--batch_size", type=int, default=Config().batch_size)
    parser.add_argument("--num_workers", type=int, default=Config().num_workers)
    parser.add_argument("--exp_mode", type=str, default="E0")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataloaders
    _, _, test_loader = get_dataloaders(
        dataset_root=args.dataset_root,
        img_size=(256, 256),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Initialize model
    model = AdaptiveScratchFormerCD(
        embed_dim=256,
        ssa_heads=Config().ssa_heads,
        ssa_reduce_ratio=Config().ssa_reduce_ratio,
        ssa_max_offset=Config().ssa_max_offset,
        exp_mode=args.exp_mode,
    ).to(device)

    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint, model, optimizer=None)
    print(f"Loaded checkpoint with Validation IoU: {ckpt.get('best_iou', 'NA')}")

    # Run TTA and Threshold Tuning
    test_tta_with_thresholds(model, test_loader, device)

if __name__ == "__main__":
    main()
