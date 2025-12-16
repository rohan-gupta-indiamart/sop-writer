
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

SYSTEM_INSTRUCTION = """You are an Expert Process Engineer for IndiaMART.  
Your job is to read a raw call transcript between a Seller and IndiaMART Support, identify every distinct problem raised by the seller, and convert each problem into a reusable and voice-friendly Standard Operating Procedure (SOP) suitable for an IndiaMART Voice AI Agent.

You MUST produce SOPs that are abstract, reusable, and optimized for Text-to-Speech (TTS).  
You MUST identify multiple problems if present.

---

### OUTPUT FORMAT (STRICT JSON ARRAY)

[
  {
    "concern_summary": "String. High-level, abstracted problem title. DO NOT include specific product names, brand names, cities, seller IDs, or overly specific details.",
    "resolution_sop": "String. A numbered procedural script (1., 2., 3., ...) with short, crisp TTS-ready sentences. Must include upsell/value-add in the final step.",
    "key_topics": ["String", "String"],
    "sentiment_transition": "String"
  }
]

Important:
The output MUST be a valid JSON array.
Each array element corresponds to ONE seller concern.

---

### RULES & CONTEXT SETTINGS

#### 1. Concern Summary (Abstract)
Represent the problem at a category level.
ALWAYS remove specificity.
  - BAD: “Customer wants to switch Jeans to Charcoal.”
  - GOOD: “Catalog Update: Product Category Switch”
NO personal names, NO exact product names in this field.
Summaries must describe the nature of the problem, not the specific details.

#### 2. Resolution SOP (Voice-First, Actionable)
This MUST be a **single string containing numbered steps**, like:

"1. Verify login credentials.  
 2. Open the Seller Panel.  
 3. Go to Manage Products.  
 4. Choose the product and click Edit.  
 5. Save changes.  
 6. Suggest upgrading for better lead visibility."

**Mandatory Structure:**
1. Start with Verification/Login steps.  
2. Navigate steps (e.g., “Go to X > Click Y”).  
3. Perform the Core Action (Deactivate, Update Category, Upload GST, etc.).  
4. Add an **Upsell/Value-Add statement** as the final step (e.g., “For more visibility, consider upgrading your plan.”).

Constraints:
Keep sentences **short, clear, and TTS-friendly**.
No long clauses.
No ambiguity.

#### 3. Key Topics (for RAG & Search)
MUST include:
  - Technical phrases (e.g., “Seller Tools”, “GST Verification”, “Lead Quality”)
  - Specific entities that appear in the call, such as product names, category names, etc.
These DO NOT need abstraction (unlike concern_summary).

#### 4. Sentiment Transition
Infer how the seller’s emotion changed during the call.
Format:
"Angry → Neutral"
"Confused → Satisfied"
"Frustrated → Calm"

If no change detected:
"Neutral → Neutral"

#### 5. Multi-Problem Detection
If the call contains more than one issue:
SPLIT into separate SOP objects.
Do NOT merge different concerns into one SOP.

Examples of multiple concerns:
GST verification issue + pricing enquiry  
Product deletion + catalog visibility problem  
Lead issue + account login problem  

Each becomes ONE JSON object in the array.

---

### INSTRUCTIONS TO THE MODEL
1. Read the transcript carefully.  
2. Detect ALL distinct seller concerns.  
3. For EACH concern:
   - Generate an abstract concern_summary  
   - Create a procedural SOP (voice-ready)  
   - Capture key_topics  
   - Identify sentiment transition  
4. Output a STRICT JSON ARRAY.

-----
ULTRA STRICT ANTI-HALLUCINATION RULE:
You are FORBIDDEN from inventing or assuming any information not directly stated in the transcript. 
You must NOT guess product names, seller intentions, missing steps, data values, or context. 
If the transcript does not mention something, do NOT include it.

SELF-CHECK BEFORE ANSWERING:
Before producing the output, re-check every sentence and confirm:
"Is this statement explicitly supported by the transcript?"
If the answer is no, REMOVE IT.

Your final output must contain ONLY transcript-grounded information.


### SOP IMPROVEMENT CHECK
If the transcript describes a problem matching an EXISTING SOP, check the outcome:
1. Was the customer unsatisfied with the initial/standard resolution?
2. Did a different executive (or the same one) provide a "Better Solution" that resolved the issue?

If YES (Better Solution Found):
- DO NOT use `duplicate_of_id`.
- Generate the FULL SOP with the NEW, BETTER resolution.
- Modify the `concern_summary` to include "(Improved)".
- This will allow the system to capture the better method.

### DUPLICATE CHECK
(If NO better solution is found, proceed with duplicate check)
You have access to a list of "Existing SOPs" (titles).
Check if the current transcript is describing a problem that is ALREADY covered by one of these titles.
If the transcript describes the EXACT SAME problem as an existing SOP:
- You DO NOT need to generate a new `resolution_sop`.
- Instead, output a JSON object with:
  {
     "duplicate_of_id": <ID of the matching SOP>,
     "concern_summary": "Duplicate: <Title of the matching SOP>",
     "key_topics": [<new topics found in this call>],
     "sentiment_transition": ...
  }
If it is a NEW problem (or an IMPROVED solution), generate the full standard JSON object as described above.
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
    # IST Timezone (UTC+5:30)
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    timestamp = datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

    row_data = {
        "Timestamp": timestamp,
        "concern_summary": sop_data.get("concern_summary", ""),
        "resolution_sop": sop_data.get("resolution_sop", ""),
        "key_topics": sop_data.get("key_topics", []),
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

def generate_sop(transcript_text, metadata=None, existing_sops=None):
    context_str = ""
    if metadata:
        context_str += f"\nCONTEXT:\n"
        for k, v in metadata.items():
            context_str += f"{k}: {v}\n"
    
    if existing_sops:
        context_str += "\nEXISTING KNOWLEDGE BASE SOPS:\n"
        for item in existing_sops:
            context_str += f"- ID {item.get('id', '?')}: {item.get('concern_summary', 'Unknown')}\n"

    try:
        response = client.chat.completions.create(
            model="openai/gpt-5", # Updated to full model
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
        raw_json = json.loads(response.choices[0].message.content)
        
        # Robust handling for "wrapped" lists (e.g. {"sops": [...]})
        if isinstance(raw_json, dict):
            # Check for common wrapper keys
            for key in ["sops", "concerns", "results", "result", "output", "data", "SOPs", "response"]:
                if key in raw_json and isinstance(raw_json[key], list):
                    return raw_json[key]
            
            # If no wrapper found but it looks like a single SOP (has concern_summary), wrap it
            if "concern_summary" in raw_json:
                return [raw_json]
                
            # Fallback: maybe the dict itself is just junk or unknown wrapper
            # Return as list of 1 if it has content, else empty
            return [raw_json] if raw_json else []
            
        elif isinstance(raw_json, list):
            return raw_json
            
        return []
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
        
        # Load Existing KB for Deduplication
        knowledge_base = []
        if os.path.exists(OUTPUT_FILE):
             try:
                 with open(OUTPUT_FILE, 'r') as f:
                     knowledge_base = json.load(f)
             except:
                 knowledge_base = []
        
        # We need a fresh list for the new batch if we are re-processing everything?
        # Actually, if we are running the pipeline, maybe we want to extend?
        # For this logic, let's treat the loaded KB as "Existing" and we append strictly new ones.
        
        initial_kb_size = len(knowledge_base)
    
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
            # Pass existing KB to check for dupes
            sop_data_list = generate_sop(str(transcript), metadata=meta, existing_sops=knowledge_base)
            
            # The model might return a list (as per system prompt instructions saying "OUTPUT FORMAT (STRICT JSON ARRAY)")
            if not isinstance(sop_data_list, list):
                sop_data_list = [sop_data_list] # Handle single object return just in case
            
            for sop_data in sop_data_list:
                if not sop_data: continue

                # Check if duplicate
                if 'duplicate_of_id' in sop_data:
                    print(f"  -> Detected Duplicate of SOP ID {sop_data['duplicate_of_id']}")
                    # We can log this in registry but NOT add to KB
                    sop_data['resolution_sop'] = f"Refer to SOP ID {sop_data['duplicate_of_id']}"
                    update_sop_sheet(sop_data) # Log call
                else:
                    # New SOP
                    sop_data['id'] = len(knowledge_base)
                    sop_data.update(meta)
                    knowledge_base.append(sop_data)
                    update_sop_sheet(sop_data)
                    print(f"  -> Generated New SOP ID {sop_data['id']}")
            
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
