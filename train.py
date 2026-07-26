import argparse
import gc
import math
import os

import torch
from tqdm import tqdm

from config import Config
from models.model import AdaptiveScratchFormerCD
from utils.losses import BCEDiceLoss
from utils.metrics import MetricTracker, batch_metrics_from_logits
from utils.train_utils import get_dataloaders, set_seed


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str):
    model.eval()
    tracker = MetricTracker()

    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        logits = model(pre, post)
        loss = criterion(logits, mask)

        iou, f1, oa = batch_metrics_from_logits(logits, mask)

        bs = pre.size(0)
        tracker.update(loss.item(), iou, f1, oa, bs)

    return tracker


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, default=Config().dataset_root)
    parser.add_argument("--epochs", type=int, default=Config().epochs)
    parser.add_argument("--batch_size", type=int, default=Config().batch_size)
    parser.add_argument("--num_workers", type=int, default=Config().num_workers)
    parser.add_argument("--img_size", type=int, nargs=2, default=list(Config().img_size))

    parser.add_argument("--lr", type=float, default=Config().lr)
    parser.add_argument("--weight_decay", type=float, default=Config().weight_decay)
    parser.add_argument("--grad_clip_norm", type=float, default=Config().grad_clip_norm)

    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--lambda_offset", type=float, default=Config().lambda_offset)
    parser.add_argument("--warmup_epochs", type=int, default=Config().warmup_epochs)

    parser.add_argument("--amp", action="store_true", default=True)

    parser.add_argument("--ssa_heads", type=int, default=Config().ssa_heads)
    parser.add_argument("--ssa_reduce_ratio", type=int, default=Config().ssa_reduce_ratio)
    parser.add_argument("--ssa_max_offset", type=float, default=Config().ssa_max_offset)

    parser.add_argument("--seed", type=int, default=Config().seed)

    args = parser.parse_args()

    # -------------------------
    # Setup
    # -------------------------

    set_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device={device}")

    # -------------------------
    # Data
    # -------------------------

    train_loader, val_loader, _ = get_dataloaders(
        dataset_root=args.dataset_root,
        img_size=(args.img_size[0], args.img_size[1]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # -------------------------
    # Model
    # -------------------------

    model = AdaptiveScratchFormerCD(
        embed_dim=256,
        ssa_heads=args.ssa_heads,
        ssa_reduce_ratio=args.ssa_reduce_ratio,
        ssa_max_offset=args.ssa_max_offset,
    ).to(device)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # -------------------------
    # Scheduler
    # -------------------------

    warmup_epochs = max(int(args.warmup_epochs), 0)
    total_epochs = int(args.epochs)

    cosine_epochs = max(total_epochs - warmup_epochs, 1)

    def lr_lambda(epoch: int):

        # Warmup
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        # Cosine
        t = float(epoch - warmup_epochs)
        t = max(min(t, float(cosine_epochs)), 0.0)

        return 0.5 * (1.0 + math.cos(math.pi * t / float(cosine_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

    # -------------------------
    # AMP
    # -------------------------

    use_amp = bool(args.amp) and device.type == "cuda"

    # torch.cuda.amp.GradScaler is deprecated in newer PyTorch versions.
    if hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -------------------------
    # Checkpoints
    # -------------------------

    os.makedirs(args.output_dir, exist_ok=True)

    best_path = os.path.join(args.output_dir, "best.ckpt")
    last_path = os.path.join(args.output_dir, "last.ckpt")

    best_iou = -1.0
    start_epoch = 1

    # -------------------------
    # Resume Training
    # -------------------------

    if args.resume is not None and os.path.exists(args.resume):

        print(f"Loading checkpoint: {args.resume}")

        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False
        )

        # -------------------------------------------------
        # CASE 1:
        # New checkpoint format
        # -------------------------------------------------

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

            print("Detected NEW checkpoint format")

            model.load_state_dict(checkpoint["model_state_dict"])

            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            start_epoch = int(checkpoint.get("epoch", 0)) + 1

            best_iou = float(checkpoint.get("best_iou", -1.0))

        # -------------------------------------------------
        # CASE 2:
        # OLD checkpoint format
        # -------------------------------------------------

        elif isinstance(checkpoint, dict) and "model" in checkpoint:

            print("Detected OLD checkpoint format")

            model.load_state_dict(checkpoint["model"])

            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])

            start_epoch = int(checkpoint.get("epoch", 0)) + 1

            best_iou = float(checkpoint.get("best_iou", -1.0))

        # -------------------------------------------------
        # CASE 3:
        # Raw state_dict only
        # -------------------------------------------------

        else:

            print("Detected raw model state_dict")

            model.load_state_dict(checkpoint)

            start_epoch = 1

            best_iou = -1.0

        print(f"Resumed from epoch {start_epoch}")
        print(f"Best IoU so far: {best_iou:.4f}")
    # -------------------------
    # Training Loop
    # -------------------------

    for epoch in range(start_epoch, args.epochs + 1):

        model.train()

        train_tr = MetricTracker()

        pbar = tqdm(train_loader, desc=f"train-{epoch:03d}", leave=False)

        for batch in pbar:

            pre = batch["pre"].to(device, non_blocking=True)
            post = batch["post"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if hasattr(torch, "amp"):
                autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

            with autocast_ctx:

                logits = model(pre, post)

                seg_loss = criterion(logits, mask)

                offset_reg = model.get_offset_reg_loss()

                loss = seg_loss + float(args.lambda_offset) * offset_reg

            # -------------------------
            # Backward
            # -------------------------

            if use_amp:

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip_norm
                )

                scaler.step(optimizer)

                scaler.update()

            else:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip_norm
                )

                optimizer.step()

            # -------------------------
            # Metrics
            # -------------------------

            with torch.no_grad():

                iou, f1, oa = batch_metrics_from_logits(logits, mask)

            bs = pre.size(0)

            train_tr.update(
                loss.item(),
                iou,
                f1,
                oa,
                bs
            )

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "iou": f"{iou:.4f}"
            })

        # -------------------------
        # Validation
        # -------------------------

        val_tr = evaluate(
            model,
            val_loader,
            criterion,
            device,
            desc="val"
        )

        scheduler.step()

        gc.collect()
        torch.cuda.empty_cache()

        # -------------------------
        # Metrics
        # -------------------------

        train_loss, train_iou, train_f1, train_oa = train_tr.compute()

        val_loss, val_iou, val_f1, val_oa = val_tr.compute()

        lr_now = float(optimizer.param_groups[0]["lr"])

        print(
            f"epoch={epoch:03d} "
            f"lr={lr_now:.6f} "
            f"train(loss={train_loss:.4f}, iou={train_iou:.4f}, "
            f"f1={train_f1:.4f}, oa={train_oa:.4f}) "
            f"val(loss={val_loss:.4f}, iou={val_iou:.4f}, "
            f"f1={val_f1:.4f}, oa={val_oa:.4f})"
        )

        # -------------------------
        # Save Last Checkpoint
        # -------------------------

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_iou": best_iou,
        }, last_path)

        # -------------------------
        # Save Best Checkpoint
        # -------------------------

        if val_iou > best_iou:

            best_iou = val_iou

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_iou": best_iou,
            }, best_path)

            print(f"saved_best_iou={best_iou:.4f}")


if __name__ == "__main__":
    main()