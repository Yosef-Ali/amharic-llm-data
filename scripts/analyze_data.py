#!/usr/bin/env python3
"""
Utility script to analyze collected Amharic dataset
"""

import json
import jsonlines
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

def analyze_dataset(data_path="data/final_amharic_dataset.jsonl"):
    """Analyze the collected dataset"""
    
    print("=" * 50)
    print("Amharic Dataset Analysis")
    print("=" * 50)
    
    # Load data
    data = []
    with jsonlines.open(data_path) as reader:
        for item in reader:
            data.append(item)
    
    print(f"\n📊 Total examples: {len(data)}")
    
    # Task distribution
    tasks = Counter([item.get('task', 'unknown') for item in data])
    print("\n📝 Task Distribution:")
    for task, count in tasks.most_common():
        print(f"  - {task}: {count:,} ({count/len(data)*100:.1f}%)")
    
    # Source distribution
    sources = Counter([item.get('source', 'unknown') for item in data])
    print("\n📚 Source Distribution:")
    for source, count in sources.most_common():
        print(f"  - {source}: {count:,} ({count/len(data)*100:.1f}%)")
    
    # Length statistics
    instruction_lengths = []
    response_lengths = []
    amharic_ratios = []
    
    for item in data:
        if 'instruction' in item:
            instruction_lengths.append(len(item['instruction']))
            # Calculate Amharic ratio
            text = item['instruction']
            amharic_chars = len(re.findall(r'[\u1200-\u137F]', text))
            total_chars = len(text.replace(" ", ""))
            if total_chars > 0:
                amharic_ratios.append(amharic_chars / total_chars)
        
        if 'response' in item:
            response_lengths.append(len(item['response']))
    
    print("\n📏 Length Statistics:")
    if instruction_lengths:
        print(f"  Instructions:")
        print(f"    - Average: {sum(instruction_lengths)/len(instruction_lengths):.0f} chars")
        print(f"    - Min: {min(instruction_lengths)} chars")
        print(f"    - Max: {max(instruction_lengths)} chars")
    
    if response_lengths:
        print(f"  Responses:")
        print(f"    - Average: {sum(response_lengths)/len(response_lengths):.0f} chars")
        print(f"    - Min: {min(response_lengths)} chars")
        print(f"    - Max: {max(response_lengths)} chars")
    
    if amharic_ratios:
        print(f"\n🔤 Amharic Content:")
        print(f"  - Average Amharic ratio: {sum(amharic_ratios)/len(amharic_ratios)*100:.1f}%")
        print(f"  - Min ratio: {min(amharic_ratios)*100:.1f}%")
        print(f"  - Max ratio: {max(amharic_ratios)*100:.1f}%")
    
    # Sample examples
    print("\n📝 Sample Examples:")
    print("-" * 50)
    
    for i, item in enumerate(data[:3]):
        print(f"\nExample {i+1}:")
        print(f"Task: {item.get('task', 'unknown')}")
        print(f"Source: {item.get('source', 'unknown')}")
        
        instruction = item.get('instruction', '')[:150]
        if len(item.get('instruction', '')) > 150:
            instruction += "..."
        print(f"Instruction: {instruction}")
        
        response = item.get('response', '')[:150]
        if len(item.get('response', '')) > 150:
            response += "..."
        print(f"Response: {response}")
    
    # Quality checks
    print("\n✅ Quality Checks:")
    empty_instructions = sum(1 for item in data if not item.get('instruction', '').strip())
    empty_responses = sum(1 for item in data if not item.get('response', '').strip())
    duplicates = len(data) - len(set(item.get('instruction', '') for item in data))
    
    print(f"  - Empty instructions: {empty_instructions}")
    print(f"  - Empty responses: {empty_responses}")
    print(f"  - Duplicate instructions: {duplicates}")
    
    # Save detailed statistics
    stats = {
        "total_examples": len(data),
        "tasks": dict(tasks),
        "sources": dict(sources),
        "avg_instruction_length": sum(instruction_lengths)/len(instruction_lengths) if instruction_lengths else 0,
        "avg_response_length": sum(response_lengths)/len(response_lengths) if response_lengths else 0,
        "avg_amharic_ratio": sum(amharic_ratios)/len(amharic_ratios) if amharic_ratios else 0,
        "quality": {
            "empty_instructions": empty_instructions,
            "empty_responses": empty_responses,
            "duplicates": duplicates
        }
    }
    
    stats_path = Path(data_path).parent / "detailed_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Detailed statistics saved to: {stats_path}")
    print("=" * 50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Amharic dataset")
    parser.add_argument(
        "--data",
        default="data/final_amharic_dataset.jsonl",
        help="Path to dataset file"
    )
    
    args = parser.parse_args()
    analyze_dataset(args.data)
