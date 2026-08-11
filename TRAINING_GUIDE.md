# Deformable SSA Change Detection - Training & Command Reference Guide

This document is a comprehensive reference guide for running ablation experiments (`E0` through `E5`), resuming training from checkpoints, and evaluating models.

---

## 1. Clean Epoch Logging (No `Tee-Object` Required!)

We have modernized `train.py` with an **automatic epoch-only file logger**.
- When you run `train.py`, live `tqdm` progress bars will still display in your terminal for real-time visual feedback.
- However, **progress bars are never written to the log file**. Instead, `train.py` automatically maintains a clean, concise log file at `outputs/<exp_folder>/train_log.txt` containing only exact epoch requirements:
  ```text
  epoch=077 lr=0.000070 train(loss=0.1528, iou=0.7746, f1=0.8521, oa=0.9916) val(loss=0.1678, iou=0.5685, f1=0.6365, oa=0.9915)
  saved_best_iou=0.5685
  ```
- **You do NOT need to use PowerShell pipes or `Tee-Object` anymore!** You can run Python commands directly in your terminal.

---

## 2. Resuming Training from the Last Checkpoint

If training is interrupted (e.g., due to system sleep, reboot, or stopping after epoch 83), you can resume seamlessly from where it left off using the `--resume` flag.

### Command to Resume Experiment 1 (`E1: + LayerNorm`):
```powershell
python train.py --exp_mode E1 --resume outputs/E1_LayerNorm/last.ckpt
```

### What Happens When You Resume:
1. **Weights Restored**: Loads the exact model parameters from `last.ckpt`.
2. **Optimizer & Scheduler Restored**: Restores AdamW momentum/variance states and the exact learning rate curve.
3. **Epoch & IoU Continuity**: Resumes from `last_epoch + 1` and preserves the previous `best_iou` score, appending new epoch summaries directly to `outputs/E1_LayerNorm/train_log.txt`.

---

## 3. Starting New Ablation Experiments (`E0` – `E5`)

Each experiment automatically creates its own isolated output directory under `outputs/` and routes its checkpoints (`best.ckpt`, `last.ckpt`) and logs (`train_log.txt`) there.

| Experiment | Description | Command | Output Directory |
| :--- | :--- | :--- | :--- |
| **E0** | Baseline (No LayerNorm, AvgPool, No FFN) | `python train.py --exp_mode E0` | `outputs/E0_baseline/` |
| **E1** | **+ LayerNorm** in Attention | `python train.py --exp_mode E1` | `outputs/E1_LayerNorm/` |
| **E2** | **+ Depthwise Conv** Downsampling | `python train.py --exp_mode E2` | `outputs/E2_DepthwiseDownsample/` |
| **E3** | **+ FFN / MLP** Blocks | `python train.py --exp_mode E3` | `outputs/E3_FFN/` |
| **E4** | **+ Offset Regularization** ($\lambda=0.001$) | `python train.py --exp_mode E4` | `outputs/E4_OffsetReg/` |
| **E5** | **+ Positional Encoding** (Depthwise Conv) | `python train.py --exp_mode E5` | `outputs/E5_PositionalEncoding/` |

---

## 4. Evaluating / Testing Trained Checkpoints

To evaluate a trained model on the test dataset and generate visual prediction overlays (saved in `outputs/vis/`):

### Test Experiment 1 Best Checkpoint:
```powershell
python test.py --exp_mode E1 --checkpoint outputs/E1_LayerNorm/best.ckpt
```

### Test Baseline (E0) Best Checkpoint:
```powershell
python test.py --exp_mode E0 --checkpoint outputs/E0_baseline/best.ckpt
```

---

## 5. Helpful Monitoring Commands (PowerShell)

### Check the last 10 epochs of an ongoing training run:
```powershell
Get-Content outputs/E1_LayerNorm/train_log.txt -Tail 10
```

### Watch log file updates in real-time (like `tail -f`):
```powershell
Get-Content outputs/E1_LayerNorm/train_log.txt -Wait -Tail 15
```

### Check current highest validation IoU achieved:
```powershell
Select-String -Path "outputs/E1_LayerNorm/train_log.txt" -Pattern "saved_best_iou" | Select-Object -Last 5
```
