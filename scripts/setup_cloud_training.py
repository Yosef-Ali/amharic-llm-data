#!/usr/bin/env python3
"""
Cloud Training Setup Helper
Automatically prepares your environment for cloud-based Amharic LLM training.
"""

import os
import json
import argparse
import subprocess
import sys
from pathlib import Path

def check_environment():
    """Check if we're running in a cloud environment"""
    environments = {
        'colab': '/content' in os.getcwd() or 'COLAB_GPU' in os.environ,
        'kaggle': '/kaggle' in os.getcwd() or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ,
        'local': True  # fallback
    }
    
    for env, condition in environments.items():
        if condition and env != 'local':
            return env
    return 'local'

def install_requirements():
    """Install required packages for cloud training"""
    packages = [
        'transformers>=4.30.0',
        'datasets>=2.12.0',
        'peft>=0.4.0',
        'accelerate>=0.20.0',
        'bitsandbytes>=0.39.0',
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0'
    ]
    
    print("📦 Installing required packages...")
    for package in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
            print(f"✅ {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def check_gpu():
    """Check GPU availability and specs"""
    try:
        import torch
        
        print("\n🔍 GPU Check:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name}")
            print(f"Memory: {gpu_memory:.1f} GB")
            print(f"CUDA version: {torch.version.cuda}")
            
            # Recommend models based on GPU memory
            if gpu_memory >= 15:
                print("\n💡 Recommended: Phi-3.5-mini or Bloom-1B1 (high quality)")
            elif gpu_memory >= 8:
                print("\n💡 Recommended: Bloom-560M (balanced)")
            else:
                print("\n💡 Recommended: DistilGPT2 (fast)")
                
            return True
        else:
            print("❌ No GPU available. Training will be very slow.")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def setup_data_paths(env):
    """Setup data paths for different environments"""
    paths = {
        'colab': {
            'data': '/content/drive/MyDrive/amharic-llm-data/data',
            'models': '/content/drive/MyDrive/amharic-models',
            'working': '/content/amharic-llm-data'
        },
        'kaggle': {
            'data': '/kaggle/working/amharic-llm-data/data',
            'models': '/kaggle/working/models',
            'working': '/kaggle/working/amharic-llm-data'
        },
        'local': {
            'data': './data',
            'models': './models',
            'working': '.'
        }
    }
    
    return paths.get(env, paths['local'])

def verify_dataset(data_path):
    """Verify dataset is available and show statistics"""
    processed_path = os.path.join(data_path, 'processed')
    stats_file = os.path.join(data_path, 'dataset_statistics.json')
    
    print("\n📊 Dataset Verification:")
    
    if not os.path.exists(processed_path):
        print(f"❌ Dataset not found at: {processed_path}")
        print("\n📋 Setup Instructions:")
        print("1. Upload your data to the cloud platform")
        print("2. Ensure the 'data/processed' folder exists")
        print("3. Run this script again")
        return False
    
    # Check for required files
    required_files = ['train.jsonl', 'validation.jsonl', 'test.jsonl']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(processed_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"✅ {file} ({size:.1f} KB)")
        else:
            missing_files.append(file)
            print(f"❌ {file} (missing)")
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    # Show statistics if available
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            print(f"\n📈 Dataset Statistics:")
            print(f"Total examples: {stats.get('total_examples', 'N/A')}")
            print(f"Train: {stats.get('train_size', 'N/A')}")
            print(f"Validation: {stats.get('validation_size', 'N/A')}")
            print(f"Test: {stats.get('test_size', 'N/A')}")
        except Exception as e:
            print(f"⚠️  Could not read statistics: {e}")
    
    return True

def generate_training_commands(env, paths, gpu_available):
    """Generate training commands for the environment"""
    models_dir = paths['models']
    
    print("\n🚀 Training Commands:")
    print("=" * 50)
    
    if gpu_available:
        commands = [
            {
                'name': 'Quick Test (10-15 min)',
                'cmd': f'python scripts/fast_training.py --train --model distilgpt2 --steps 100 --output {models_dir}/amharic-distilgpt2-test',
                'desc': 'Fast pipeline validation'
            },
            {
                'name': 'Balanced Training (30-45 min)',
                'cmd': f'python scripts/fast_training.py --train --model bloom-560m --steps 300 --output {models_dir}/amharic-bloom560m-finetuned',
                'desc': 'Good quality for most uses'
            },
            {
                'name': 'High Quality (1-2 hours)',
                'cmd': f'python scripts/fast_training.py --train --model bloom-1b1 --steps 500 --output {models_dir}/amharic-bloom1b1',
                'desc': 'Production-ready quality'
            },
            {
                'name': 'Premium Quality (2-4 hours)',
                'cmd': f'python scripts/train_example.py --train --model microsoft/Phi-3.5-mini-instruct --output {models_dir}/amharic-phi35',
                'desc': 'Highest quality available'
            }
        ]
    else:
        commands = [
            {
                'name': 'CPU Training (Very Slow)',
                'cmd': f'python scripts/fast_training.py --train --model distilgpt2 --steps 50 --output {models_dir}/amharic-cpu-test',
                'desc': 'Minimal training for testing'
            }
        ]
    
    for i, cmd_info in enumerate(commands, 1):
        print(f"\n{i}. {cmd_info['name']}")
        print(f"   Description: {cmd_info['desc']}")
        print(f"   Command: {cmd_info['cmd']}")
    
    # Testing command
    print(f"\n🧪 Testing Command:")
    print(f"python scripts/fast_training.py --test --output {models_dir}/[MODEL_NAME]")
    
    return commands

def create_quick_start_script(env, paths, commands):
    """Create a quick start script for the environment"""
    script_content = f'''#!/bin/bash
# Quick Start Script for {env.title()} Training
# Generated automatically by setup_cloud_training.py

echo "🚀 Starting Amharic LLM Training on {env.title()}"
echo "Environment: {env}"
echo "Models will be saved to: {paths['models']}"
echo ""

# Create models directory
mkdir -p {paths['models']}

# Quick validation training
echo "1️⃣  Running quick validation..."
{commands[0]['cmd']}

if [ $? -eq 0 ]; then
    echo "✅ Quick training successful!"
    echo "🧪 Testing the model..."
    python scripts/fast_training.py --test --output {paths['models']}/amharic-distilgpt2-test
    
    echo ""
    echo "🎉 Setup complete! Your training pipeline is working."
    echo "📋 Next steps:"
    echo "   - Run balanced training: {commands[1]['cmd'] if len(commands) > 1 else 'N/A'}"
    echo "   - Or high quality training for production use"
    echo "   - Check the models in: {paths['models']}"
else
    echo "❌ Training failed. Check the error messages above."
    exit 1
fi
'''
    
    script_path = f'quick_start_{env}.sh'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"\n📝 Quick start script created: {script_path}")
    print(f"Run with: bash {script_path}")
    
    return script_path

def main():
    parser = argparse.ArgumentParser(description='Setup cloud training environment')
    parser.add_argument('--install', action='store_true', help='Install required packages')
    parser.add_argument('--check-only', action='store_true', help='Only check environment, don\'t install')
    parser.add_argument('--create-script', action='store_true', help='Create quick start script')
    
    args = parser.parse_args()
    
    print("🔧 Amharic LLM Cloud Training Setup")
    print("=" * 40)
    
    # Detect environment
    env = check_environment()
    print(f"Environment: {env.title()}")
    
    # Setup paths
    paths = setup_data_paths(env)
    print(f"Working directory: {paths['working']}")
    print(f"Models directory: {paths['models']}")
    
    # Install packages if requested
    if args.install and not args.check_only:
        if not install_requirements():
            print("❌ Package installation failed")
            return 1
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Verify dataset
    dataset_ok = verify_dataset(paths['data'])
    
    if not dataset_ok:
        print("\n⚠️  Dataset verification failed. Please upload your data first.")
        return 1
    
    # Generate training commands
    commands = generate_training_commands(env, paths, gpu_available)
    
    # Create quick start script if requested
    if args.create_script:
        create_quick_start_script(env, paths, commands)
    
    # Final recommendations
    print("\n🎯 Recommendations:")
    if gpu_available:
        print("✅ GPU detected - you're ready for fast training!")
        print("💡 Start with the 'Quick Test' to validate your setup")
        print("🚀 Then run 'Balanced Training' for good quality results")
    else:
        print("⚠️  No GPU detected - training will be slow")
        print("💡 Consider using Google Colab or Kaggle for GPU access")
    
    print("\n📚 For detailed instructions, see: CLOUD_TRAINING_SETUP.md")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())