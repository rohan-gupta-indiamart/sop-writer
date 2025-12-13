
import openai
import json
import pandas as pd
import os
import datetime
import csv

# Configuration
API_KEY = "sk-XhmNFaPbVqeIwl3RqrAvfQ" # Using the key found in writer.py
BASE_URL = "https://imllm.intermesh.net/v1"
# Determine the project root (assuming this script is in pipeline/ or similar depth)
# pipeline/sop_generator.py -> parent is project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Priority list of filenames to look for
CSV_FILENAMES = [
    "Copy of Call Transcriptions - IM Hackathon - Transciption.csv",
    "Call Transcriptions - IM Hackathon - Transciption translated.csv"
]

# Try multiple locations
POSSIBLE_PATHS = []
for fname in CSV_FILENAMES:
    POSSIBLE_PATHS.append(os.path.join(PROJECT_ROOT, fname))
    POSSIBLE_PATHS.append(os.path.join(os.getcwd(), fname))
    POSSIBLE_PATHS.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", fname))
    POSSIBLE_PATHS.append(fname)

INPUT_CSV = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        INPUT_CSV = os.path.abspath(path) # Force absolute path
        break

if not INPUT_CSV:
    print(f"WARNING: Could not find '{CSV_FILENAME}' in any logical location.")
    # Fallback to the project root with first filename
    INPUT_CSV = os.path.join(PROJECT_ROOT, CSV_FILENAMES[0])

OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "knowledge_base.json")
CSV_REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "sop_registry.csv")

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_INSTRUCTION = """
### SYSTEM ROLE
You are an expert Process Engineer for IndiaMART. Your goal is to convert raw call transcripts into re-usable Standard Operating Procedures (SOPs) for a Voice AI Agent.

### OUTPUT FORMAT (JSON)
{
  "concern_summary": "String. A generic, high-level title. DO NOT use specific product names (e.g., use 'Product Switch' instead of 'Jeans to Charcoal').",
  "resolution_sop": "String. A numbered list of executable steps. Use short sentences suitable for Text-to-Speech.",
  "key_topics": ["String", "String"],
  "sentiment_transition": "String"
}

### CONTENT GUIDELINES
1. **concern_summary (Abstraction):**
   - BAD: "Customer wants to change Jeans to Charcoal."
   - GOOD: "Catalog Update: Product Category Switch"
   - Rule: Abstract specific entities (names, products) into categories.

2. **resolution_sop (Voice-First Formatting):**
   - Must use numbered steps (1. Step one. 2. Step two.) within the string.
   - Keep sentences punchy. Avoid long clauses.
   - **Mandatory Structure:**
     1. Verification/Login steps.
     2. Navigation steps (e.g., "Go to X > Click Y").
     3. The Core Action (e.g., "Click Deactivate").
     4. **Upsell/Value Add** (Must be the final step).

3. **key_topics:** Include technical terms (e.g., 'Seller Tools', 'GST') and the specific entities (e.g., 'Jeans', 'Charcoal') here so search still works.
"""

def transcribe_audio(file_pointer):
    """
    Transcribes audio file object using OpenAI Whisper model.
    """
    try:
        # file_pointer is a file-like object from streamlit
        # We might need to save it temporarily or pass it directly if supported.
        # client.audio.transcriptions.create expects a file-like object with a name or a path.
        
        # If it's a streamlit UploadedFile, it has a name.
        
        response = client.audio.transcriptions.create(
            model="openai/whisper-1",
            file=file_pointer
        )
        return response.text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def update_sop_sheet(sop_data, csv_path=CSV_REGISTRY_PATH):
    """
    Appends the generated SOP to a persistent CSV registry.
    """
    headers = ["Timestamp", "concern_summary", "resolution_sop", "key_topics", "sentiment_transition"]
    
    # Flatten data for CSV
    row_data = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "concern_summary": sop_data.get("concern_summary", ""),
        "resolution_sop": sop_data.get("resolution_sop", ""),
        "key_topics": ", ".join(sop_data.get("key_topics", [])),
        "sentiment_transition": sop_data.get("sentiment_transition", "")
    }
    
    file_exists = os.path.isfile(csv_path)
    file_empty = not file_exists or os.stat(csv_path).st_size == 0
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if file_empty:
            writer.writeheader()
        writer.writerow(row_data)

def update_knowledge_base_json(sop_data, json_path=OUTPUT_FILE):
    """
    Appends the generated SOP to the JSON Knowledge Base.
    """
    try:
        data = []
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        
        # Ensure ID
        if 'id' not in sop_data:
            sop_data['id'] = len(data)
            
        data.append(sop_data)
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        print(f"Error updating KB JSON: {e}")

def generate_sop(transcript_text, metadata=None):
    context_str = ""
    if metadata:
        context_str = f"\nCONTEXT:\n"
        for k, v in metadata.items():
            context_str += f"{k}: {v}\n"

    try:
        response = client.chat.completions.create(
            model="openai/gpt-5-mini", # Updated fallback model
            # Note: writer.py had issues with chat models vs completion models. 
            # The error in writer.py said "This is not a chat model". 
            # However, the code was using client.chat.completions.create.
            # I will try "openai/gpt-5" as per the successful attempt in writer.py or "gpt-4o".
            # Let's stick to what looked like it was intended: "openai/gpt-5" was used in the try block on line 122.
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": f"{context_str}\nTRANSCRIPT:\n{transcript_text}"}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error generating SOP: {e}")
        # Fallback for the demo if API fails
        return None

def run_pipeline():
    try:
        print(f"Reading from {INPUT_CSV}...")
        if not INPUT_CSV or not os.path.exists(INPUT_CSV):
            print(f"ERROR: CSV file not found at looked-up path: {INPUT_CSV}")
            return
    
        df = pd.read_csv(INPUT_CSV)
        
        knowledge_base = []
    
        # Process all transcripts
        print(f"Processing {len(df)} transcripts...")
        for index, row in df.iterrows():
            transcript = row['Translated Transcript']
            
            # Extract Metadata if columns exist (Mirroring app.py logic)
            meta = {}
            # Standard Fields
            if 'Call Category' in row: meta['Category'] = row['Call Category']
            if 'Call Disposition' in row: meta['Disposition'] = row['Call Disposition']
            
            # Augmented Fields (GLID, Summary, Timestamp, URL)
            if 'GLID' in row: meta['GLID'] = row['GLID']
            if 'Summary' in row and pd.notna(row['Summary']): meta['Call Summary'] = row['Summary']
            if 'Call Timestamp' in row: meta['Timestamp'] = row['Call Timestamp']
            if 'Call recording url' in row: meta['Audio URL'] = row['Call recording url']
            
            print(f"Processing row {index}...")
            sop_data = generate_sop(str(transcript), metadata=meta)
            if sop_data:
                sop_data['id'] = index
                sop_data.update(meta) # Include meta in JSON
                knowledge_base.append(sop_data)
                
                # Also to CSV
                update_sop_sheet(sop_data)
            
        print(f"Generated {len(knowledge_base)} SOPs.")
        
        update_knowledge_base_json_bulk(knowledge_base)
        print(f"Saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"CRITICAL ERROR in pipeline: {e}")
        import traceback
        traceback.print_exc()

def update_knowledge_base_json_bulk(kb_data):
    # Helper to save all at once
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(kb_data, f, indent=2)

if __name__ == "__main__":
    run_pipeline()
