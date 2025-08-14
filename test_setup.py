#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
"""

print("Testing Amharic LLM Data Collection Setup...")
print("=" * 50)

# Test imports
try:
    print("✓ Testing core imports...")
    import sys
    from pathlib import Path
    import json
    import logging
    print("  Core imports successful")
except ImportError as e:
    print(f"  ✗ Core import failed: {e}")
    sys.exit(1)

# Test data science imports
try:
    print("✓ Testing data science imports...")
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    print("  Data science imports successful")
except ImportError as e:
    print(f"  ⚠️  Data science import failed: {e}")
    print("  Run: pip install pandas numpy tqdm")

# Test ML imports
try:
    print("✓ Testing ML imports...")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    print("  ML imports successful")
except ImportError as e:
    print(f"  ⚠️  ML import failed: {e}")
    print("  Run: pip install datasets transformers sentence-transformers")

# Test web scraping imports
try:
    print("✓ Testing web scraping imports...")
    import requests
    from bs4 import BeautifulSoup
    print("  Web scraping imports successful")
except ImportError as e:
    print(f"  ⚠️  Web scraping import failed: {e}")
    print("  Run: pip install requests beautifulsoup4")

# Test project structure
print("\n✓ Testing project structure...")
project_root = Path("/Users/mekdesyared/amharic-llm-data")

required_dirs = [
    "src",
    "configs", 
    "scripts",
    "data",
    "data/raw",
    "data/processed",
    "data/synthetic"
]

for dir_name in required_dirs:
    dir_path = project_root / dir_name
    if dir_path.exists():
        print(f"  ✓ {dir_name}/ exists")
    else:
        print(f"  ✗ {dir_name}/ missing")

# Test config import
try:
    print("\n✓ Testing project imports...")
    sys.path.append(str(project_root))
    from configs.config import DATA_SOURCES, INSTRUCTION_TEMPLATES
    print("  Config import successful")
    print(f"  Found {len(DATA_SOURCES['huggingface'])} HuggingFace sources")
    print(f"  Found {len(INSTRUCTION_TEMPLATES)} task templates")
except ImportError as e:
    print(f"  ✗ Config import failed: {e}")

# Test Amharic text
print("\n✓ Testing Amharic text handling...")
test_text = "ሰላም ዓለም! Hello World!"
print(f"  Test text: {test_text}")
import re
amharic_chars = re.findall(r'[\u1200-\u137F]', test_text)
print(f"  Found {len(amharic_chars)} Amharic characters")

# Summary
print("\n" + "=" * 50)
print("✅ Setup test complete!")
print("\nNext steps:")
print("1. Run quick test: python scripts/quickstart.py")
print("2. Or use menu: ./run.sh")
print("=" * 50)
