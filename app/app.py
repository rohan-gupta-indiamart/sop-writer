
import streamlit as st
import sys
import os
import pandas as pd

# Add parent dir to path to import agent/pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.brain import VoiceAgent
from pipeline.sop_generator import run_pipeline, generate_sop, transcribe_audio, update_sop_sheet

# Define absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

CSV_PATH = os.path.join(DATA_DIR, 'sop_registry.csv')

# Ensure data dir exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

st.set_page_config(page_title="IndiaMART AI SOP Writer", layout="wide", page_icon="📞")

# --- Custom CSS for Responsiveness & Styling ---
st.markdown("""
<style>
    /* Main Container */
    .main {
        background-color: #f8f9fa;
        padding-top: 1rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2c3e50;
    }
    h1 {
        text-align: center;
        margin-bottom: 2rem;
        color: #00875a; /* IndiaMART Greenish tone */
    }
    
    /* Card-like containers for Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #00875a;
        color: #00875a;
        font-weight: bold;
    }
    
    /* Responsive Improvements for Mobile */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }
        h1 {
            font-size: 1.8rem;
        }
        .stButton>button {
            width: 100%;
        }
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #00875a;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #006c48;
        color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.title("📞 AI Policy Writer & Voice Agent")

tab1, tab2, tab3 = st.tabs(["SOP Generator (Bulk)", "Real-time SOP Creator", "Voice Agent Prototype"])

with tab1:
    st.header("Automated SOP Generation (Bulk)")
    st.write("Upload a CSV file containing transcripts in column 'Translated Transcript' to generate SOPs in bulk.")
    
    uploaded_file = st.file_uploader("Upload Transcripts CSV", type=['csv'])
    
    if uploaded_file is not None:
        if st.button("Generate SOPs from CSV"):
            with st.spinner("Processing uploaded CSV..."):
                try:
                    # Save uploaded file temporarily or read directly
                    # For simplicity, let's read into DF and process
                    df = pd.read_csv(uploaded_file)
                    st.info(f"Loaded {len(df)} rows. Processing all...")
                    
                    # We need to adapt run_pipeline to accept a dataframe or filepath
                    # Or just inline the logic here since run_pipeline was a CLI wrapper
                    knowledge_base = []
                    
                    # Find the transcript column
                    col_name = 'Translated Transcript'
                    if col_name not in df.columns:
                        # try to find first string col
                        possible_cols = [c for c in df.columns if df[c].dtype == 'object']
                        if not possible_cols:
                            st.error("No text column found in CSV.")
                            st.stop()
                        col_name = possible_cols[0]
                        st.warning(f"Column '{col_name}' not found. Using '{possible_cols[0]}' instead.")
                        col_name = possible_cols[0]

                    for index, row in df.iterrows():
                        transcript = row[col_name]
                        
                        # Extract Metadata if available
                        meta = {}
                        if 'Call Category' in row: meta['Category'] = row['Call Category']
                        if 'Call Disposition' in row: meta['Disposition'] = row['Call Disposition']
                        if 'C2C Record ID' in row: meta['Record ID'] = row['C2C Record ID']
                        
                        # New Fields requested
                        if 'GLID' in row: meta['GLID'] = row['GLID']
                        if 'Summary' in row and pd.notna(row['Summary']): meta['Call Summary'] = row['Summary']
                        if 'Call Timestamp' in row: meta['Timestamp'] = row['Call Timestamp']
                        if 'Call recording url' in row: meta['Audio URL'] = row['Call recording url']
                            
                        # st.write(f"Processing row {index}...") # debug
                        sop_data_list = generate_sop(str(transcript), metadata=meta, existing_sops=knowledge_base)
                        
                        if sop_data_list:
                            if not isinstance(sop_data_list, list):
                                sop_data_list = [sop_data_list]
                                
                            for sop_data in sop_data_list:
                                if not sop_data: continue
                                
                                # Check for duplicates (if prompt follows instruction)
                                if 'duplicate_of_id' in sop_data:
                                    # Log duplicate but don't add to KB
                                    sop_data['resolution_sop'] = f"Duplicate of ID {sop_data['duplicate_of_id']}"
                                    update_sop_sheet(sop_data)
                                    continue

                                sop_data['id'] = len(knowledge_base) # Incremental ID
                                # Merge metadata into SOP for storage visibility
                                sop_data.update(meta)
                                
                                knowledge_base.append(sop_data)
                                
                                # Also update persistent storage
                                update_sop_sheet(sop_data)
                    
                    # Save to JSON
                    st.success(f"Generated {len(knowledge_base)} SOPs! Check 'SOP Registry' below.")
                    
                except Exception as e:
                    st.error(f"Processing failed: {e}")
    else:
        st.info("Please upload a CSV file to start.")
                
    st.subheader("SOP Registry (Knowledge Base)")
    if os.path.exists(CSV_PATH):
        try:
            df_sheet = pd.read_csv(CSV_PATH, on_bad_lines='skip', engine='python')
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df_sheet = pd.DataFrame()
        st.dataframe(df_sheet, width="stretch")

    else:
        st.info("Knowledge Base is empty.")

with tab2:
    st.header("Real-time SOP Creator")
    st.write("Generate SOPs from live input (Text or Audio) and save to Registry.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Data")
        input_type = st.radio("Input Type", ["Text (Paste)", "Text File (Upload)", "Audio File"])
        
        transcript_text = ""
        meta = {}
        
        if input_type == "Audio File":
            audio_file = st.file_uploader("Upload Audio (mp3, wav)", type=['mp3', 'wav'])
            if audio_file:
                st.audio(audio_file)
                if st.button("Transcribe Audio"):
                    with st.spinner("Transcribing..."):
                        transcript_text = transcribe_audio(audio_file)
                        if transcript_text:
                            st.success("Transcription Complete!")
                            # Store in session state to persist
                            st.session_state['current_transcript'] = transcript_text
                        else:
                            st.error("Transcription failed.")
        
        elif input_type == "Text File (Upload)":
            txt_file = st.file_uploader("Upload Transcript (txt, csv)", type=['txt', 'csv'])
            if txt_file:
                if txt_file.name.endswith('.csv'):
                    df = pd.read_csv(txt_file)
                    col = 'Translated Transcript' if 'Translated Transcript' in df.columns else df.columns[0]
                    transcript_text = str(df[col].iloc[0])
                    st.info(f"Loaded first row from CSV column: {col}")
                    
                    # Metadata Extraction
                    row0 = df.iloc[0]
                    if 'Call Category' in row0: meta['Category'] = row0['Call Category']
                    if 'Call Disposition' in row0: meta['Disposition'] = row0['Call Disposition']
                    if 'GLID' in row0: meta['GLID'] = row0['GLID']
                else:
                    transcript_text = txt_file.read().decode("utf-8")
        else:
            transcript_text = st.text_area("Paste Transcript Here", height=300)

        # Handle persistent transcript from audio or file
        if 'current_transcript' in st.session_state and input_type == "Audio File":
            transcript_text = st.session_state['current_transcript']
        
        if transcript_text and input_type != "Text (Paste)":
             st.text_area("Transcript Preview", transcript_text, height=150)

        generate_btn = st.button("Generate SOP", type="primary", use_container_width=True)

    with col2:
        st.subheader("Generated SOP")
        if generate_btn:
            if not transcript_text:
                st.warning("Please provide transcript text first.")
            else:
                with st.spinner("Generating SOP..."):
                    # Load existing KB for deduplication check
                    existing_kb = []
                    if os.path.exists(CSV_PATH):
                        try: 
                             _df = pd.read_csv(CSV_PATH)
                             for _idx, _row in _df.iterrows():
                                 existing_kb.append({'id': _idx, 'concern_summary': _row.get('concern_summary', '')})
                        except: existing_kb = []

                    sop_data_list = generate_sop(transcript_text, metadata=meta, existing_sops=existing_kb)
                    
                    if sop_data_list:
                        if not isinstance(sop_data_list, list):
                            sop_data_list = [sop_data_list]
                        
                        st.success("Analysis Complete!")
                        for sop in sop_data_list:
                            if 'duplicate_of_id' in sop:
                                st.warning(f"Duplicate content detected! (Matches SOP ID {sop['duplicate_of_id']})")
                                update_sop_sheet(sop)
                            else:
                                st.json(sop)
                                # Update Storage
                                update_sop_sheet(sop)

                                st.toast("SOP saved to Registry & Agent Brain!")
                    else:
                        st.error("SOP Generation failed.")
        else:
            st.info("SOP output will appear here.")
                    
    st.subheader("SOP Registry (Knowledge Base)")
    if os.path.exists(CSV_PATH):
        try:
            df_sheet = pd.read_csv(CSV_PATH, on_bad_lines='skip', engine='python')
        except Exception as e:
            st.error(f"Error reading Registry: {e}")
            df_sheet = pd.DataFrame()
        st.dataframe(df_sheet, width="stretch")


    else:
        st.info("Registry is empty.")

with tab3:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.header("Voice Agent Simulation")
    with c2:
        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    st.write("Interact with the AI Agent powered by the generated SOPs.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Say something..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get Agent Response
        agent = VoiceAgent()
        response = agent.get_response(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
