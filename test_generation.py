
import time
import sys
import os

# Add current dir to path
sys.path.append(os.path.abspath(os.getcwd()))

from pipeline.sop_generator import generate_sop

TRANSCRIPT = """
Seller: I want to update my catalog.
Support: Sure, open the app, go to products, and click edit.
Seller: Okay done.
Support: Great, anything else?
Seller: No thanks.
"""

if __name__ == "__main__":
    print("--- Starting Generation Test ---")
    start_time = time.time()
    
    try:
        print("Calling generate_sop...")
        result = generate_sop(TRANSCRIPT, metadata={"Category": "Test"})
        end_time = time.time()
        
        print(f"--- Finished in {end_time - start_time:.2f} seconds ---")
        print("Result:", result)
        
        if result:
            from pipeline.sop_generator import update_sop_sheet
            print("Writing to CSV...")
            # result is a list of sops
            for sop in result:
                update_sop_sheet(sop)
            print("Done.")
    except Exception as e:
        print(f"ERROR: {e}")
