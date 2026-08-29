import re
import csv

# Input and output file names
input_file = r"C:\Users\srush\OneDrive\Documents\Honors\Experimentation\Main_model\outputs\E0_baseline\trainE0_14.log"
output_file = "train_log.csv"

# Regular expression to match the specific format of your log lines
log_pattern = re.compile(
    r"epoch=(\d+)\s+lr=([0-9.]+)\s+"
    r"train\(loss=([0-9.]+),\s+iou=([0-9.]+),\s+f1=([0-9.]+),\s+oa=([0-9.]+)\)\s+"
    r"val\(loss=([0-9.]+),\s+iou=([0-9.]+),\s+f1=([0-9.]+),\s+oa=([0-9.]+)\)"
)

try:
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Write the CSV headers
        writer.writerow(['epoch', 'lr', 'train_loss', 'train_iou', 'train_f1', 'train_oa', 
                         'val_loss', 'val_iou', 'val_f1', 'val_oa'])
        
        count = 0
        for line in infile:
            match = log_pattern.search(line)
            if match:
                # Extract groups. Convert epoch to int to drop leading zeros (001 -> 1)
                row = [int(match.group(1))] + list(match.groups()[1:])
                writer.writerow(row)
                count += 1
                
    print(f"Success! Extracted {count} epochs and saved to '{output_file}'.")

except FileNotFoundError:
    print(f"Error: Could not find the file '{input_file}'. Please ensure it is in the same directory as this script.")