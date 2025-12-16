import openai
import json

def generate_sop_from_transcript(transcript_text):
    client = openai.OpenAI(api_key="sk-uUANxcPXCaQOwrCHUQ-FJg",base_url = "https://imllm.intermesh.net/v1")

    # The Prompt Definition
    system_instruction = """
    You are an expert QA Analyst. Convert the transcript into a JSON Knowledge Base entry.
    Schema:
    {
      "concern_summary": "Concise Title (Max 6 words)",
      "resolution_sop": "Step-by-step resolution paragraph.",
      "key_topics": ["List", "of", "keywords"],
      "sentiment_transition": "Start > End"
    }
    """

    system_instruction = """
### SYSTEM ROLE
You are an expert Quality Assurance Analyst and Process Engineer for IndiaMART's Customer Service department. Your goal is to convert raw, unstructured customer call transcripts into structured, actionable Knowledge Base entries.

### TASK
Analyze the provided customer service call transcript and extract structured data into a JSON object. You must ignore pleasantries, small talk, and repetitive confirmations to focus strictly on the business logic and resolution steps.

### OUTPUT FORMAT (JSON)
You must return a single JSON object with the following schema:

{
  "concern_summary": "String. A concise, title-style header (max 5-8 words). Describes the core issue.",
  "resolution_sop": "String. A dense, instructional paragraph describing the steps taken to resolve the issue. Use imperative or descriptive language (e.g., 'Guide to...', 'Instruct to...'). Merge related steps.",
  "key_topics": ["String", "String", "String"],
  "sentiment_transition": "String. Format: 'Start Emotion > End Emotion' (e.g., 'Angry > Satisfied')"
}

### CONTENT GUIDELINES
1. concern_summary: Must be a Title. NO sentences.
   - BAD: "The customer wanted to change products."
   - GOOD: "Product Category Switch & Catalog Update"
   
2. resolution_sop: Must be actionable. Imagine this text will be read by a Voice AI to help another user with the same problem.
   - Include specific navigation steps (e.g., "Go to Seller Tools > Manage Product").
   - Mention specific advice given (e.g., "Pitch subscription benefits").
   
3. key_topics: Extract 4-6 specific business entities (e.g., 'GST', 'Buyleads', 'Catalog', 'OTP').

4. sentiment_transition: Infer the sentiment based on tone, word choice, and resolution confirmation.

### ONE-SHOT EXAMPLE (Follow this style exactly)

INPUT TRANSCRIPT:
Executive: Welcome to IndiaMART.
Customer: I want to sell charcoal, not jeans anymore. I have a new company name.
Executive: Okay, do you have the app?
Customer: No.
Executive: Please download it. Log in with your number. Go to the three lines at the top, click Seller Tools, then Manage Product.
Customer: Okay, I see jeans.
Executive: Click the three dots next to jeans and click Deactivate. Then click Add Product to add charcoal.
Customer: Done.
Executive: Since you have GST, you can also buy a subscription for better leads.

OUTPUT JSON:
{
  "concern_summary": "Product Category Switch & Catalog Update",
  "resolution_sop": "Guide customer to download app and login. Navigate to Seller Tools > Manage Product. Instruct to click three dots next to old products and select 'Deactivate'. Click 'Add Product' to enter new details. Pitch subscription benefits based on GST availability.",
  "key_topics": ["product update", "manage product", "deactivate", "add product", "gst", "subscription"],
  "sentiment_transition": "Neutral > Satisfied"
}

### YOUR INPUT
TRANSCRIPT:
{{INSERT_TRANSCRIPT_HERE}}
    """




    system_instruction = """
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

### ONE-SHOT EXAMPLE
TRANSCRIPT: "I want to stop selling rice and start selling wheat. How do I do that on the app?" ... "Click delete on rice, add wheat." ... "Also buy a plan."

OUTPUT JSON:
{
  "concern_summary": "Catalog Update: Replace Existing Product",
  "resolution_sop": "1. Ask customer to open App and login. 2. Navigate to 'Seller Tools' > 'Manage Product'. 3. Locate old product and select 'Delete' or 'Deactivate'. 4. Select 'Add Product' to list the new item. 5. Pitch premium plan for better visibility on the new category.",
  "key_topics": ["Rice", "Wheat", "Manage Product", "Premium Plan"],
  "sentiment_transition": "Neutral > Satisfied"
}
"""

    try:
        response = client.chat.completions.create( #-> itellm.BadRequestError: OpenAIException - This is not a chat model and thus not supported in the v1/chat/completions endpoint. Did you mean to use v1/completions?No fallback model group found for original model_group=openai/gpt-4o-transcribe. Fallbacks=[]
      # response = client.completions.create(
            model= "openai/gpt-5", #"openai/gpt-4o-transcribe"#gave not a chat model error, #"gpt-4o", # Recommended for complex extraction
            response_format={ "type": "json_object" }, # ENSURES valid JSON
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"TRANSCRIPT:\n{transcript_text}"}
            ]
        )

        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error generating SOP: {e}")
        return "{'concern_summary': '', 'resolution_sop': '', 'key_topics': [], 'sentiment_transition': '', 'error': 'some error occurred'}"


import pandas as pd

try:
    # 1. Read the CSV file into a DataFrame
    # Replace 'your_file.csv' with your actual file name
    df = pd.read_csv('Call Transcriptions - IM Hackathon - Transciption translated.csv')

    # 2. Read the 1st row value from the "Translated Transcript" column
    # .iloc[0] gets the first row by position
    my_raw_transcript_string = df['Translated Transcript'].iloc[0]

    # 3. Verify the output
    print(my_raw_transcript_string)
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Example Usage
data = generate_sop_from_transcript(my_raw_transcript_string)
print(data['concern_summary'])
print(data)