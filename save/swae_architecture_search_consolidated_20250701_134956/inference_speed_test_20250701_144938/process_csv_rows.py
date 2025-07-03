#!/usr/bin/env python3
import csv
import sys

def process_csv_rows(csv_file):
    """Process CSV rows and output them in a format that bash can parse safely"""
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, 1):
            # Output row data separated by ||| to avoid comma issues
            print(f"{row_idx}|||{row['latent_dim']}|||{row['arch_config']}|||{row['channels']}|||{row['lr']}|||{row['batch_size']}|||{row['lambda_reg']}|||{row['compression_ratio']}|||{row['final_psnr']}|||{row['best_psnr']}|||{row['training_time']}|||{row['status']}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_csv_rows.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    process_csv_rows(csv_file)
