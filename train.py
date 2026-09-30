import argparse
import gc
import math
import os
import sys

import torch
from tqdm import tqdm

from config import Config
from models.model import AdaptiveScratchFormerCD
from utils.losses import BCEDiceLoss
from utils.metrics import MetricTracker, batch_metrics_from_logits, batch_counts_from_logits
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

        if hasattr(torch, "amp"):
            autocast_ctx = torch.amp.autocast("cuda")
        else:
            autocast_ctx = torch.cuda.amp.autocast()

        with autocast_ctx:
            logits = model(pre, post)
            loss = criterion(logits, mask)

            tp, fp, fn, tn = batch_counts_from_logits(logits, mask)

        bs = pre.size(0)
        tracker.update(loss.item(), tp, fp, fn, tn, bs)

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
    parser.add_argument("--exp_mode", type=str, default="E0", help="Ablation experiment mode: E0, E1, E2, E3, E4, E5")

    parser.add_argument("--seed", type=int, default=Config().seed)

    args = parser.parse_args()

    # Offset loss is now enabled by default as part of the Stop Bleeding fixes.
    # The default value from Config (0.01 or 0.001) will be used unless overridden.
    if args.output_dir == "outputs":
        exp_names = {
            "A0": "A0_FullModel",
            "A1": "A1_NoCTA",
            "A2": "A2_NoPixelShuffle",
            "A3": "A3_NoFocalLoss",
            "A4": "A4_NoCEFF",
            # Legacy modes below
            "E0": "E0_baseline",
            "E1": "E1_LayerNorm",
            "E2": "E2_DepthwiseDownsample",
            "E3": "E3_FFN",
            "E4": "E4_OffsetReg",
            "E5": "E5_PositionalEncoding",
        }
        subfolder = exp_names.get(args.exp_mode.upper(), args.exp_mode)
        args.output_dir = os.path.join("outputs", subfolder)

    # -------------------------
    # Setup Log File
    # -------------------------

    os.makedirs(args.output_dir, exist_ok=True)
    
    base_log_name = f"train{args.exp_mode.upper()}"
    log_idx = 1
    while True:
        log_file_path = os.path.join(args.output_dir, f"{base_log_name}_{log_idx}.log")
        if not os.path.exists(log_file_path):
            break
        log_idx += 1
        
    args.log_file_path = log_file_path
    print(f"Logging to: {args.log_file_path}")

    # -------------------------
    # Setup
    # -------------------------

    set_seed(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
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
        exp_mode=args.exp_mode,
    ).to(device)

    # Ablation A3: Disable Focal Loss
    f_weight = 0.0 if args.exp_mode.upper() == "A3" else 0.5
    criterion = BCEDiceLoss(focal_weight=f_weight)

    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": other_params, "lr": args.lr}
    ], weight_decay=args.weight_decay)

    # -------------------------
    # Scheduler
    # -------------------------

    warmup_epochs = max(int(args.warmup_epochs), 0)
    total_epochs = int(args.epochs)

    # Warmup scheduler (linear ramp from 0 to base LR)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_epochs, 1),
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    # Single smooth cosine decay from base LR to eta_min over the remaining epochs
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
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

                logits, logits_f2, logits_f3 = model(pre, post)

                loss_final = criterion(logits, mask)
                loss_f2 = criterion(logits_f2, mask)
                loss_f3 = criterion(logits_f3, mask)

                seg_loss = loss_final + 0.4 * loss_f3 + 0.2 * loss_f2

                offset_reg = model.get_offset_reg_loss()

                loss = seg_loss + float(args.lambda_offset) * offset_reg

            # NaN guard: skip this batch entirely if loss is NaN/Inf
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            # -------------------------
            # Backward
            # -------------------------

            if use_amp:

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                # Check for NaN/Inf in gradients BEFORE stepping
                grad_ok = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grad_ok = False
                        break

                if not grad_ok:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip_norm
                )

                scaler.step(optimizer)

                scaler.update()

            else:

                loss.backward()

                # Check for NaN/Inf in gradients BEFORE stepping
                grad_ok = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grad_ok = False
                        break

                if not grad_ok:
                    optimizer.zero_grad(set_to_none=True)
                    continue

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip_norm
                )

                optimizer.step()

            # -------------------------
            # Metrics
            # -------------------------

            with torch.no_grad():

                tp, fp, fn, tn = batch_counts_from_logits(logits, mask)
                iou, f1, oa = batch_metrics_from_logits(logits, mask)

            bs = pre.size(0)

            train_tr.update(
                loss.item(),
                tp, fp, fn, tn,
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

        # -------------------------
        # Metrics
        # -------------------------

        train_loss, train_iou, train_f1, train_oa = train_tr.compute()

        val_loss, val_iou, val_f1, val_oa = val_tr.compute()

        lr_now = float(optimizer.param_groups[0]["lr"])

        summary_str = (
            f"epoch={epoch:03d} "
            f"lr={lr_now:.6f} "
            f"train(loss={train_loss:.4f}, iou={train_iou:.4f}, "
            f"f1={train_f1:.4f}, oa={train_oa:.4f}) "
            f"val(loss={val_loss:.4f}, iou={val_iou:.4f}, "
            f"f1={val_f1:.4f}, oa={val_oa:.4f})"
        )
        print(summary_str)
        with open(args.log_file_path, "a", encoding="utf-8") as f:
            f.write(summary_str + "\n")

        # -------------------------
        # Crash Recovery
        # -------------------------
        # If val loss is NaN or val IoU collapsed (dropped >20% below best),
        # reload the best checkpoint and continue training from there.

        val_crashed = (
            not math.isfinite(val_loss)
            or (best_iou > 0.5 and val_iou < best_iou * 0.8)
        )

        if val_crashed and os.path.exists(best_path):
            crash_str = f"CRASH DETECTED at epoch {epoch} — reloading best checkpoint (iou={best_iou:.4f})"
            print(crash_str)
            with open(args.log_file_path, "a", encoding="utf-8") as f:
                f.write(crash_str + "\n")

            # Reload best model weights
            ckpt_recovery = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt_recovery["model_state_dict"])

            # Reset optimizer state to clear any corrupted momentum
            optimizer = torch.optim.AdamW([
                {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": lr_now * 0.1},
                {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": lr_now}
            ], weight_decay=args.weight_decay)

            # Reset AMP scaler
            if hasattr(torch, "amp"):
                scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            else:
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            continue

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

            best_str = f"saved_best_iou={best_iou:.4f}"
            print(best_str)
            with open(args.log_file_path, "a", encoding="utf-8") as f:
                f.write(best_str + "\n")


if __name__ == "__main__":
    main()