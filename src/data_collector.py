#!/usr/bin/env python3
"""
Amharic LLM Data Collection Pipeline
Modern approach combining existing datasets with synthetic generation
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import warnings

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import configurations
from configs.config import (
    DATA_SOURCES, 
    INSTRUCTION_TEMPLATES,
    QUALITY_FILTERS,
    OUTPUT_PATHS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class AmharicDataCollector:
    """Main data collection orchestrator"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.synthetic_dir = self.output_dir / "synthetic"
        
        for dir_path in [self.raw_dir, self.processed_dir, self.synthetic_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.all_data = []
        logger.info(f"Initialized AmharicDataCollector with output_dir: {output_dir}")
    
    def collect_all_data(self):
        """Main pipeline to collect all data sources"""
        logger.info("Starting comprehensive data collection...")
        
        # 1. Collect HuggingFace datasets
        logger.info("Step 1: Collecting HuggingFace datasets...")
        hf_data = self.collect_huggingface_datasets()
        logger.info(f"Collected {len(hf_data)} examples from HuggingFace")
        
        # 2. Scrape web data
        logger.info("Step 2: Scraping web data...")
        web_data = self.scrape_web_data()
        logger.info(f"Collected {len(web_data)} examples from web scraping")
        
        # 3. Generate synthetic data
        logger.info("Step 3: Generating synthetic data...")
        synthetic_data = self.generate_synthetic_data()
        logger.info(f"Generated {len(synthetic_data)} synthetic examples")
        
        # 4. Combine all data
        self.all_data = hf_data + web_data + synthetic_data
        logger.info(f"Total collected: {len(self.all_data)} examples")
        
        # 5. Apply quality filters
        logger.info("Step 4: Applying quality filters...")
        filtered_data = self.apply_quality_filters(self.all_data)
        logger.info(f"After filtering: {len(filtered_data)} examples")
        
        # 6. Convert to instruction format
        logger.info("Step 5: Converting to instruction format...")
        instruction_data = self.convert_to_instructions(filtered_data)
        
        # 7. Save final dataset
        self.save_dataset(instruction_data)
        
        return instruction_data
    
    def collect_huggingface_datasets(self) -> List[Dict]:
        """Collect all configured HuggingFace datasets"""
        from datasets import load_dataset
        import pandas as pd
        
        collected_data = []
        
        for dataset_name, config in DATA_SOURCES["huggingface"].items():
            try:
                logger.info(f"Loading {dataset_name}...")
                
                # Load dataset
                if config.get("subset"):
                    dataset = load_dataset(
                        config["dataset"], 
                        config["subset"]
                    )
                else:
                    dataset = load_dataset(
                        config["dataset"]
                    )
                
                # Get train split
                if "train" in dataset:
                    data = dataset["train"]
                else:
                    data = dataset
                
                # Limit samples if specified
                max_samples = config.get("max_samples", 10000)
                if len(data) > max_samples:
                    data = data.select(range(max_samples))
                
                # Process based on task type
                task = config["task"]
                
                for item in data:
                    processed_item = self.process_dataset_item(item, task, dataset_name)
                    if processed_item:
                        collected_data.append(processed_item)
                
                logger.info(f"✓ Collected {len(data)} examples from {dataset_name}")
                
            except Exception as e:
                logger.error(f"✗ Error loading {dataset_name}: {e}")
                continue
        
        # Save raw data
        self.save_intermediate(collected_data, "huggingface_raw.jsonl")
        return collected_data
    
    def process_dataset_item(self, item: Dict, task: str, source: str) -> Optional[Dict]:
        """Process a single dataset item based on task type"""
        try:
            processed = {
                "source": source,
                "task": task,
                "timestamp": datetime.now().isoformat()
            }
            
            if task == "sentiment":
                processed["text"] = item.get("text", "")
                processed["label"] = item.get("label", "")
                
            elif task == "classification":
                processed["text"] = item.get("text", "") or item.get("headline", "")
                processed["label"] = item.get("category", "") or item.get("label", "")
                
            elif task == "ner":
                processed["text"] = " ".join(item.get("tokens", []))
                processed["entities"] = item.get("ner_tags", [])
                
            elif task == "summarization":
                processed["text"] = item.get("text", "") or item.get("article", "")
                processed["summary"] = item.get("summary", "")
                
            elif task == "translation":
                if "translation" in item:
                    processed["source_text"] = item["translation"].get("en", "")
                    processed["target_text"] = item["translation"].get("am", "")
                else:
                    processed["source_text"] = item.get("source", "")
                    processed["target_text"] = item.get("target", "")
                    
            elif task == "pretrain":
                processed["text"] = item.get("text", "")
                
            else:
                processed["text"] = str(item)
            
            return processed if processed.get("text") else None
            
        except Exception as e:
            logger.debug(f"Error processing item: {e}")
            return None
    
    def scrape_web_data(self) -> List[Dict]:
        """Scrape data from configured web sources"""
        import requests
        from bs4 import BeautifulSoup
        import time
        
        collected_data = []
        
        for source_name, config in DATA_SOURCES["web_scraping"].items():
            try:
                logger.info(f"Scraping {source_name}...")
                
                if source_name == "bbc_amharic":
                    data = self.scrape_bbc_amharic(config)
                elif source_name == "voa_amharic":
                    data = self.scrape_voa_amharic(config)
                elif source_name == "wikimezmur":
                    data = self.scrape_wikimezmur(config)
                else:
                    logger.warning(f"Scraper not implemented for {source_name}")
                    continue
                
                collected_data.extend(data)
                logger.info(f"✓ Scraped {len(data)} items from {source_name}")
                
            except Exception as e:
                logger.error(f"✗ Error scraping {source_name}: {e}")
                continue
        
        # Save scraped data
        self.save_intermediate(collected_data, "scraped_raw.jsonl")
        return collected_data
    
    def scrape_bbc_amharic(self, config: Dict) -> List[Dict]:
        """Scrape BBC Amharic news articles"""
        import requests
        from bs4 import BeautifulSoup
        import time
        
        articles = []
        base_url = config["url"]
        
        try:
            response = requests.get(base_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links (BBC specific selectors)
            article_links = soup.find_all('a', class_='focusIndicatorDisplayBlock')[:20]
            
            for link in article_links[:config.get("max_pages", 10)]:
                try:
                    article_url = f"https://www.bbc.com{link.get('href')}"
                    article_response = requests.get(article_url, timeout=10)
                    article_soup = BeautifulSoup(article_response.content, 'html.parser')
                    
                    # Extract title and content
                    title = article_soup.find('h1')
                    content = article_soup.find_all('p')
                    
                    if title and content:
                        articles.append({
                            "source": "bbc_amharic",
                            "task": "news",
                            "title": title.text.strip(),
                            "text": " ".join([p.text.strip() for p in content[:10]]),
                            "url": article_url
                        })
                    
                    time.sleep(1)  # Be respectful to the server
                    
                except Exception as e:
                    logger.debug(f"Error scraping article: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error accessing BBC Amharic: {e}")
        
        return articles
    
    def scrape_voa_amharic(self, config: Dict) -> List[Dict]:
        """Scrape VOA Amharic - placeholder for now"""
        # Similar implementation to BBC
        return []
    
    def scrape_wikimezmur(self, config: Dict) -> List[Dict]:
        """Scrape WikiMezmur lyrics - placeholder for now"""
        # Implementation for lyrics scraping
        return []
    
    def generate_synthetic_data(self, num_examples: int = 1000) -> List[Dict]:
        """Generate synthetic Amharic data using LLMs"""
        synthetic_data = []
        
        # Check for API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not openai_key and not anthropic_key:
            logger.warning("No API keys found for synthetic generation. Skipping...")
            return []
        
        if openai_key:
            logger.info("Generating synthetic data with OpenAI...")
            openai_data = self.generate_with_openai(num_examples // 2)
            synthetic_data.extend(openai_data)
        
        if anthropic_key:
            logger.info("Generating synthetic data with Anthropic...")
            anthropic_data = self.generate_with_anthropic(num_examples // 2)
            synthetic_data.extend(anthropic_data)
        
        # Save synthetic data
        self.save_intermediate(synthetic_data, "synthetic_raw.jsonl")
        return synthetic_data
    
    def generate_with_openai(self, num_examples: int) -> List[Dict]:
        """Generate synthetic data using OpenAI GPT-4"""
        try:
            from openai import OpenAI
            client = OpenAI()
            
            synthetic_data = []
            batch_size = 10
            
            prompt = """Generate 10 diverse Amharic instruction-following examples.
            Each example should be culturally relevant to Ethiopia and natural.
            Format each as JSON with 'instruction' and 'response' fields.
            Topics: history, culture, education, daily life, technology.
            
            Return as JSON array."""
            
            for i in range(0, num_examples, batch_size):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.8,
                        max_tokens=2000
                    )
                    
                    # Parse response
                    content = response.choices[0].message.content
                    examples = json.loads(content)
                    
                    for ex in examples:
                        synthetic_data.append({
                            "source": "gpt4_synthetic",
                            "task": "instruction",
                            "instruction": ex.get("instruction", ""),
                            "response": ex.get("response", "")
                        })
                    
                    logger.info(f"Generated batch {i//batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"Error in batch generation: {e}")
                    continue
            
            return synthetic_data
            
        except ImportError:
            logger.warning("OpenAI library not installed")
            return []
        except Exception as e:
            logger.error(f"Error with OpenAI generation: {e}")
            return []
    
    def apply_quality_filters(self, data: List[Dict]) -> List[Dict]:
        """Apply quality filters to the collected data"""
        import re
        from collections import Counter
        
        filtered_data = []
        stats = {
            "total": len(data),
            "too_short": 0,
            "too_long": 0,
            "not_amharic": 0,
            "repetitive": 0,
            "passed": 0
        }
        
        for item in data:
            text = item.get("text", "") or item.get("instruction", "")
            
            if not text:
                continue
            
            # Check length
            if len(text) < QUALITY_FILTERS["min_length"]:
                stats["too_short"] += 1
                continue
            
            if len(text) > QUALITY_FILTERS["max_length"]:
                stats["too_long"] += 1
                continue
            
            # Check if text contains enough Amharic characters
            amharic_chars = len(re.findall(r'[\u1200-\u137F]', text))
            total_chars = len(text.replace(" ", ""))
            
            if total_chars > 0:
                amharic_ratio = amharic_chars / total_chars
                if amharic_ratio < QUALITY_FILTERS["min_amharic_ratio"]:
                    stats["not_amharic"] += 1
                    continue
            
            # Check for repetition
            words = text.split()
            if len(words) > 10:
                word_counts = Counter(words)
                most_common_count = word_counts.most_common(1)[0][1]
                repetition_ratio = most_common_count / len(words)
                
                if repetition_ratio > QUALITY_FILTERS["max_repetition_ratio"]:
                    stats["repetitive"] += 1
                    continue
            
            # Passed all filters
            filtered_data.append(item)
            stats["passed"] += 1
        
        # Log statistics
        logger.info("Quality filter statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return filtered_data
    
    def convert_to_instructions(self, data: List[Dict]) -> List[Dict]:
        """Convert data to instruction-following format"""
        import random
        
        instruction_data = []
        
        for item in data:
            task = item.get("task", "general")
            
            # Skip if no templates for this task
            if task not in INSTRUCTION_TEMPLATES:
                task = "generation"  # Default to generation
            
            templates = INSTRUCTION_TEMPLATES.get(task, [])
            if not templates:
                continue
            
            # Generate multiple instruction variations
            num_variations = min(3, len(templates))  # Create up to 3 variations
            selected_templates = random.sample(templates, num_variations)
            
            for template in selected_templates:
                instruction_item = self.create_instruction_item(item, template, task)
                if instruction_item:
                    instruction_data.append(instruction_item)
        
        logger.info(f"Created {len(instruction_data)} instruction examples from {len(data)} items")
        return instruction_data
    
    def create_instruction_item(self, item: Dict, template: str, task: str) -> Optional[Dict]:
        """Create a single instruction item from data and template"""
        try:
            instruction_item = {
                "task": task,
                "source": item.get("source", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
            
            if task == "sentiment":
                instruction = template.format(text=item.get("text", ""))
                response = item.get("label", "")
                
            elif task == "classification":
                instruction = template.format(text=item.get("text", ""))
                response = item.get("label", "")
                
            elif task == "summarization":
                instruction = template.format(text=item.get("text", ""))
                response = item.get("summary", "")
                
            elif task == "translation":
                source_text = item.get("source_text", "")
                if "አማርኛ" in template or "amharic" in template.lower():
                    instruction = template.format(text=source_text)
                    response = item.get("target_text", "")
                else:
                    instruction = template.format(text=item.get("target_text", ""))
                    response = source_text
                
            elif task == "qa":
                question = item.get("question", "")
                context = item.get("context", "")
                instruction = template.format(question=question, context=context)
                response = item.get("answer", "")
                
            elif task == "ner":
                instruction = template.format(text=item.get("text", ""))
                entities = item.get("entities", [])
                response = ", ".join([str(e) for e in entities])
                
            else:  # generation or other tasks
                if "instruction" in item:
                    instruction = item["instruction"]
                    response = item.get("response", "") or item.get("output", "")
                else:
                    instruction = template.format(
                        topic=item.get("topic", "ኢትዮጵያ"),
                        text=item.get("text", "")[:100]
                    )
                    response = item.get("text", "")
            
            instruction_item["instruction"] = instruction.strip()
            instruction_item["response"] = response.strip()
            
            # Only return if both instruction and response are non-empty
            if instruction_item["instruction"] and instruction_item["response"]:
                return instruction_item
                
        except Exception as e:
            logger.debug(f"Error creating instruction item: {e}")
        
        return None
    
    def save_intermediate(self, data: List[Dict], filename: str):
        """Save intermediate data for debugging"""
        import jsonlines
        
        filepath = self.raw_dir / filename
        try:
            with jsonlines.open(filepath, mode='w') as writer:
                writer.write_all(data)
            logger.debug(f"Saved intermediate data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving intermediate data: {e}")
    
    def save_dataset(self, data: List[Dict]):
        """Save the final dataset in multiple formats"""
        import jsonlines
        import pandas as pd
        import random
        
        # Shuffle data
        random.shuffle(data)
        
        # Save as JSONL
        jsonl_path = self.output_dir / "final_amharic_dataset.jsonl"
        with jsonlines.open(jsonl_path, mode='w') as writer:
            writer.write_all(data)
        logger.info(f"✓ Saved JSONL dataset to {jsonl_path}")
        
        # Save as JSON
        json_path = self.output_dir / "final_amharic_dataset.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved JSON dataset to {json_path}")
        
        # Create train/val/test splits
        n = len(data)
        train_size = int(n * 0.9)
        val_size = int(n * 0.05)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        # Save splits
        splits = {
            "train": train_data,
            "validation": val_data,
            "test": test_data
        }
        
        for split_name, split_data in splits.items():
            split_path = self.processed_dir / f"{split_name}.jsonl"
            with jsonlines.open(split_path, mode='w') as writer:
                writer.write_all(split_data)
            logger.info(f"✓ Saved {split_name} split: {len(split_data)} examples")
        
        # Save statistics
        self.save_statistics(data, splits)
        
        return data
    
    def save_statistics(self, data: List[Dict], splits: Dict):
        """Save dataset statistics"""
        stats = {
            "total_examples": len(data),
            "splits": {
                name: len(split_data) 
                for name, split_data in splits.items()
            },
            "tasks": {},
            "sources": {},
            "avg_instruction_length": 0,
            "avg_response_length": 0
        }
        
        # Calculate statistics
        instruction_lengths = []
        response_lengths = []
        
        for item in data:
            task = item.get("task", "unknown")
            source = item.get("source", "unknown")
            
            stats["tasks"][task] = stats["tasks"].get(task, 0) + 1
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
            
            if "instruction" in item:
                instruction_lengths.append(len(item["instruction"]))
            if "response" in item:
                response_lengths.append(len(item["response"]))
        
        if instruction_lengths:
            stats["avg_instruction_length"] = sum(instruction_lengths) / len(instruction_lengths)
        if response_lengths:
            stats["avg_response_length"] = sum(response_lengths) / len(response_lengths)
        
        # Save statistics
        stats_path = self.output_dir / "dataset_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info("Dataset Statistics:")
        logger.info(f"  Total examples: {stats['total_examples']}")
        logger.info(f"  Tasks: {stats['tasks']}")
        logger.info(f"  Sources: {stats['sources']}")
        
        return stats


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Amharic LLM Data Collection Pipeline")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for collected data"
    )
    parser.add_argument(
        "--skip-huggingface",
        action="store_true",
        help="Skip HuggingFace dataset collection"
    )
    parser.add_argument(
        "--skip-scraping",
        action="store_true",
        help="Skip web scraping"
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip synthetic data generation"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize collector
    collector = AmharicDataCollector(output_dir=args.output_dir)
    
    # Collect data based on flags
    logger.info("="*50)
    logger.info("Amharic LLM Data Collection Pipeline")
    logger.info("="*50)
    
    # Run collection
    final_dataset = collector.collect_all_data()
    
    logger.info("="*50)
    logger.info("✓ Data collection complete!")
    logger.info(f"✓ Final dataset: {len(final_dataset)} examples")
    logger.info("="*50)

if __name__ == "__main__":
    main()
