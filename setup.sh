#!/bin/bash

# Setup script for Amharic LLM Data Collection

echo "=========================================="
echo "Amharic LLM Data Collection Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"

# Install requirements
echo ""
echo "Installing requirements (this may take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. (Optional) Add API keys to .env file for synthetic data"
echo "3. Run quick test: python scripts/quickstart.py"
echo "4. Run full collection: python src/data_collector.py"
echo ""
echo "For reduced testing, run:"
echo "  python scripts/quickstart.py"
echo ""
echo "For full data collection, run:"
echo "  python src/data_collector.py"
