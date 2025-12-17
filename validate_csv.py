import pandas as pd
import sys

csv_path = "data/sop_registry.csv"

def check_csv():
    print(f"Checking {csv_path}...")
    
    # Method 1: Count total raw lines
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Total raw lines: {len(lines)}")
        
    # Method 2: Try standard pandas read
    try:
        df = pd.read_csv(csv_path)
        print(f"Standard read success. Rows: {len(df)}")
    except Exception as e:
        print(f"Standard read failed: {e}")
        
    # Method 3: Parsing line by line to find bad ones
    print("\n--- LINE ANALYSIS ---")
    good_count = 0
    bad_count = 0
    
    import csv
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            print(f"Header columns ({len(header)}): {header}")
            
            for i, row in enumerate(reader):
                if len(row) != len(header):
                    bad_count += 1
                    print(f"Line {i+2} BAD: Expected {len(header)} cols, got {len(row)}")
                    # print(f"Content: {row}")
                else:
                    good_count += 1
        except Exception as e:
            print(f"CSV Reader crashed at some point: {e}")
            
    print(f"\nGood Rows: {good_count}")
    print(f"Bad Rows: {bad_count}")

if __name__ == "__main__":
    check_csv()
