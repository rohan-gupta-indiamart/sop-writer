#!/bin/bash

# Navigate to the script's directory (assuming script is in the project root)
cd "$(dirname "$0")"

echo "Setup for WSL Environment"

# 1. Create a fresh virtual environment for Linux/WSL if it doesn't exist
if [ ! -d ".venv_wsl" ]; then
    echo "Creating Python virtual environment (.venv_wsl)..."
    python3 -m venv .venv_wsl
fi

# 2. Activate
source .venv_wsl/bin/activate

# 3. Install dependencies
echo "Installing dependencies..."
pip install pandas openai streamlit

# 4. Instructions
echo "========================================"
echo "Environment Setup Complete!"
echo "To generate SOPs:"
echo "  python3 pipeline/sop_generator.py"
echo ""
echo "To run the App:"
echo "  streamlit run app/app.py"
echo "========================================"
