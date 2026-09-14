import os
import subprocess
import re
import numpy as np

def main():
    seeds = [42, 100, 2026, 333, 777]
    best_ious = {}

    print(f"Starting multi-seed evaluation with seeds: {seeds}\n")

    for seed in seeds:
        print("="*60)
        print(f"▶ RUNNING SEED: {seed}")
        print("="*60)
        
        cmd = ["python", "train.py", "--exp_mode", "E0", "--seed", str(seed)]
        
        # We use Popen to stream the output to the console in real-time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        current_best_iou = 0.0
        
        for line in process.stdout:
            print(line, end="")  # Print to console so you can monitor progress
            
            # Keep track of the best IoU reported in this run
            if "saved_best_iou=" in line:
                match = re.search(r"saved_best_iou=([0-9.]+)", line)
                if match:
                    iou = float(match.group(1))
                    if iou > current_best_iou:
                        current_best_iou = iou
                        
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[WARNING] Run for seed {seed} exited with code {process.returncode}")
            
        best_ious[seed] = current_best_iou
        print(f"\n✔ Finished seed {seed}. Best IoU achieved: {current_best_iou * 100:.2f}%\n")

    # Final Statistical Summary
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY (TEST IoU)")
    print("="*60)
    
    iou_list = list(best_ious.values())
    
    for seed, iou in best_ious.items():
        print(f"Seed {seed:<6}: {iou * 100:.2f}%")
        
    print("-" * 60)
    print(f"Mean IoU:   {np.mean(iou_list) * 100:.2f}%")
    print(f"Std Dev:    {np.std(iou_list) * 100:.2f}%")
    print(f"Min IoU:    {np.min(iou_list) * 100:.2f}%")
    print(f"Max IoU:    {np.max(iou_list) * 100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
