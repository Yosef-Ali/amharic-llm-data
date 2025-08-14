#!/bin/bash

# Main run script for Amharic LLM Data Collection

echo "=========================================="
echo "🚀 Amharic LLM Data Collection Pipeline"
echo "=========================================="

# Function to display menu
show_menu() {
    echo ""
    echo "Select an option:"
    echo "1) Quick Test (100 examples per dataset)"
    echo "2) Full Data Collection"
    echo "3) Analyze Existing Dataset"
    echo "4) Train Model (QLoRA)"
    echo "5) Setup Environment"
    echo "6) Exit"
    echo ""
}

# Check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        echo "⚠️  Virtual environment not found!"
        echo "Run option 5 to setup environment first."
        return 1
    fi
    return 0
}

# Activate virtual environment
activate_venv() {
    if check_venv; then
        source venv/bin/activate
        echo "✓ Virtual environment activated"
        return 0
    fi
    return 1
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [1-6]: " choice
    
    case $choice in
        1)
            echo ""
            echo "Running Quick Test..."
            if activate_venv; then
                python scripts/quickstart.py
            fi
            ;;
            
        2)
            echo ""
            echo "Running Full Data Collection..."
            echo "This may take several hours depending on your internet connection."
            read -p "Continue? (y/n): " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                if activate_venv; then
                    python src/data_collector.py
                fi
            fi
            ;;
            
        3)
            echo ""
            echo "Analyzing Dataset..."
            if activate_venv; then
                if [ -f "data/final_amharic_dataset.jsonl" ]; then
                    python scripts/analyze_data.py
                else
                    echo "❌ No dataset found. Run data collection first."
                fi
            fi
            ;;
            
        4)
            echo ""
            echo "Training Model with QLoRA..."
            echo "⚠️  This requires a GPU with at least 6GB VRAM"
            read -p "Continue? (y/n): " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                if activate_venv; then
                    if [ -f "data/processed/train.jsonl" ]; then
                        python scripts/train_example.py --train
                    else
                        echo "❌ No training data found. Run data collection first."
                    fi
                fi
            fi
            ;;
            
        5)
            echo ""
            echo "Setting up environment..."
            chmod +x setup.sh
            ./setup.sh
            ;;
            
        6)
            echo ""
            echo "👋 Goodbye!"
            exit 0
            ;;
            
        *)
            echo "❌ Invalid option. Please select 1-6."
            ;;
    esac
done
