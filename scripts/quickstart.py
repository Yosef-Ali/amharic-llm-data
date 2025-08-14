#!/usr/bin/env python3
"""
Quick start script to test the data collection pipeline
This version works without API keys using only public datasets
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collector import AmharicDataCollector
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def quick_test():
    """Quick test with a small subset of data"""
    
    # Override config for quick testing
    from configs import config
    
    # Reduce sample sizes for testing
    for source in config.DATA_SOURCES["huggingface"].values():
        source["max_samples"] = 100  # Only get 100 samples per dataset
    
    logger.info("Starting quick test with reduced dataset sizes...")
    
    # Initialize collector
    collector = AmharicDataCollector(output_dir="data_test")
    
    # Test individual components
    logger.info("\n1. Testing HuggingFace collection...")
    hf_data = collector.collect_huggingface_datasets()
    logger.info(f"   Collected {len(hf_data)} examples")
    
    if hf_data:
        logger.info("\n2. Testing quality filters...")
        filtered = collector.apply_quality_filters(hf_data[:50])
        logger.info(f"   {len(filtered)} examples passed filters")
        
        logger.info("\n3. Testing instruction conversion...")
        instructions = collector.convert_to_instructions(filtered)
        logger.info(f"   Created {len(instructions)} instruction examples")
        
        # Show sample
        if instructions:
            logger.info("\n4. Sample instruction:")
            sample = instructions[0]
            logger.info(f"   Instruction: {sample.get('instruction', '')[:100]}...")
            logger.info(f"   Response: {sample.get('response', '')[:100]}...")
    
    logger.info("\n✓ Quick test complete!")
    logger.info("Run 'python src/data_collector.py' for full collection")

if __name__ == "__main__":
    quick_test()
