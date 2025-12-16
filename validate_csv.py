
import csv

CSV_PATH = 'data/sop_registry.csv'

with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    try:
        for i, row in enumerate(reader, 1):
            if i >= 40 and i <= 46:
                print(f"Line {i}: {len(row)} fields")
                if len(row) != 5:
                    print(f"  -> ERROR CONTENT: {row}")
    except Exception as e:
        print(f"Parser Error at line {reader.line_num}: {e}")
