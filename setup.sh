#!/bin/bash

# Check if Python is installed
if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Please install it before proceeding."
    exit 1
fi

# # Create a virtual environment (optional but recommended)
# python3 -m venv venv
# source venv/bin/activate

# Install dependencies and the package
pip install -r requirements.txt
pip install -e .

echo "Setup complete."
# echo "Activate the virtual environment with 'source venv/bin/activate'."
