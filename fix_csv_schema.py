import pandas as pd
import shutil
import os

csv_path = "data/sop_registry.csv"
backup_path = "data/sop_registry.csv.bak"

def fix_csv():
    if not os.path.exists(csv_path):
        print("File not found.")
        return

    # Backup
    shutil.copy(csv_path, backup_path)
    print(f"Backed up to {backup_path}")

    # Define the target full header
    full_headers = [
        "Timestamp", 
        "concern_summary", 
        "resolution_sop", 
        "key_topics", 
        "sentiment_transition", 
        "Input Tokens", 
        "Output Tokens", 
        "Total Tokens", 
        "Cost ($)", 
        "Cost (INR)"
    ]

    fixed_rows = []
    
    # We read manually to handle the mess
    import csv
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            # existing header
            old_header = next(reader)
            print(f"Old Header: {old_header}")
        except StopIteration:
            print("Empty file")
            return

        # We will ignore the old header and just rewrite the file with the new header
        # But we need to process the data rows
        
        for row in reader:
            # Pad row to 10 columns
            cleaned_row = row + [''] * (10 - len(row))
            # If row has > 10 cols, it's problematic (formatting issue), but let's keep first 10
            cleaned_row = cleaned_row[:10]
            fixed_rows.append(cleaned_row)

    # Write new file
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(full_headers)
        writer.writerows(fixed_rows)
        
    print(f"Fixed {len(fixed_rows)} rows. New header has 10 columns.")

if __name__ == "__main__":
    fix_csv()
