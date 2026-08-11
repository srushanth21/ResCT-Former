import argparse
import os

import torch
from tqdm import tqdm

from config import Config
from models.model import AdaptiveScratchFormerCD
from utils.losses import BCEDiceLoss
from utils.metrics import MetricTracker, batch_metrics_from_logits
from utils.train_utils import get_dataloaders, load_checkpoint, save_change_overlay


@torch.no_grad()
def test(model, loader, criterion, device, save_vis_dir: str = "") -> MetricTracker:
    model.eval()
    tracker = MetricTracker()

    if save_vis_dir:
        os.makedirs(save_vis_dir, exist_ok=True)

    pbar = tqdm(loader, desc="test", leave=False)
    for i, batch in enumerate(pbar):
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        logits = model(pre, post)
        loss = criterion(logits, mask)
        iou, f1, oa = batch_metrics_from_logits(logits, mask)

        bs = pre.size(0)
        tracker.update(loss.item(), iou, f1, oa, bs)

        if save_vis_dir and i < 10:
            pred = (torch.sigmoid(logits[0]) > 0.5).float().cpu()
            save_change_overlay(
                pre=pre[0].cpu(),
                post=post[0].cpu(),
                pred_mask=pred,
                gt_mask=mask[0].cpu(),
                out_path=os.path.join(save_vis_dir, f"sample_{i:03d}.png"),
            )

    return tracker


def main():
    # !pip install -r requirements.txt
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=Config().dataset_root)
    parser.add_argument("--checkpoint", type=str, default=os.path.join("outputs", "best.ckpt"))
    parser.add_argument("--batch_size", type=int, default=Config().batch_size)
    parser.add_argument("--num_workers", type=int, default=Config().num_workers)
    parser.add_argument("--img_size", type=int, nargs=2, default=list(Config().img_size))
    parser.add_argument("--save_vis_dir", type=str, default=os.path.join("outputs", "vis"))

    parser.add_argument("--ssa_heads", type=int, default=Config().ssa_heads)
    parser.add_argument("--ssa_reduce_ratio", type=int, default=Config().ssa_reduce_ratio)
    parser.add_argument("--ssa_max_offset", type=float, default=Config().ssa_max_offset)
    parser.add_argument("--exp_mode", type=str, default="E0", help="Ablation experiment mode: E0, E1, E2, E3, E4, E5")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    _, _, test_loader = get_dataloaders(
        dataset_root=args.dataset_root,
        img_size=(args.img_size[0], args.img_size[1]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = AdaptiveScratchFormerCD(
        embed_dim=256,
        ssa_heads=args.ssa_heads,
        ssa_reduce_ratio=args.ssa_reduce_ratio,
        ssa_max_offset=args.ssa_max_offset,
        exp_mode=args.exp_mode,
    ).to(device)

    criterion = BCEDiceLoss()

    ckpt = load_checkpoint(args.checkpoint, model, optimizer=None)
    print(f"loaded_checkpoint_epoch={ckpt.get('epoch', 'NA')} best_iou={ckpt.get('best_iou', 'NA')}")

    tr = test(model, test_loader, criterion, device, save_vis_dir=args.save_vis_dir)
    loss, iou, f1, oa = tr.compute()

    print(f"test(loss={loss:.4f}, iou={iou:.4f}, f1={f1:.4f}, oa={oa:.4f})")


if __name__ == "__main__":
    main()
