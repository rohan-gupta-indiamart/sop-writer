# SOP Writer & Voice Agent Workflow

## High Level Architecture

```mermaid
graph TD
    A[Call Transcripts (CSV)] -->|Ingest| B[SOP Generator]
    B -->|LLM Extraction| C{OpenAI / LLM}
    C -->|Structured SOP| D[Knowledge Base (JSON)]
    
    E[User Query] -->|Input| F[Voice Agent]
    F -->|Search| D
    F -->|Context + SOP| C
    C -->|Natural Response| E
```

## Workflow Description
1.  **Ingestion Phase**:
    *   The `pipeline/sop_generator.py` reads the CSV containing call transcriptions.
    *   It sends each transcript to the LLM with a specific System Prompt designed to extract:
        *   Concern Summary (Title)
        *   Resolution SOP (Step-by-step instructions)
        *   Key Topics (Metadata)
        *   Sentiment Transition
    *   The output is validated and saved to `data/knowledge_base.json`.

2.  **Agent Interaction Phase**:
    *   The User interacts with the Agent via the Streamlit Interface (`app/app.py`).
    *   The `agent/brain.py` receives the query.
    *   It first searches the `knowledge_base.json` to find the most relevant SOP based on the query.
    *   Examples: "I want to change product" -> Matches "Catalog Update" SOP.
    *   The Agent then constructs a prompt including the user's query and the specific *Resolution SOP* steps.
    *   The LLM generates a conversational response that strictly follows the SOP guidelines.

## Technologies Used
*   **Python**: Core logic.
*   **OpenAI API**: Intelligent extraction and response generation.
*   **Pandas**: Data handling.
*   **Streamlit**: User Interface for demo.
