#!/usr/bin/env python3
"""
Updated data collector that handles Parquet format and avoids deprecated scripts
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

# Suppress the deprecated dataset script warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*script.*deprecated.*')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModernAmharicDataCollector:
    """
    Modern data collector that prioritizes Parquet datasets
    and handles deprecated script warnings
    """
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_parquet_datasets(self) -> Dict:
        """
        Modern Amharic datasets that use Parquet format (no scripts)
        These load faster and don't show deprecation warnings
        """
        return {
            # Datasets already in Parquet format or with direct data access
            "amharic_news": {
                "source": "israel/Amharic-News-Text-classification-Dataset",
                "format": "parquet",
                "task": "classification",
                "loading_method": "direct"
            },
            "amharic_corpus": {
                "source": "oscar-corpus/OSCAR-2301",
                "config": "am",  
                "format": "parquet",
                "task": "text",
                "loading_method": "streaming"
            },
            "xlsum_amharic": {
                "source": "csebuetnlp/xlsum",
                "config": "amharic",
                "format": "parquet",
                "task": "summarization",
                "loading_method": "direct"
            },
            "mafand_amharic": {
                "source": "masakhane/mafand",
                "config": "am",
                "format": "json",  # Some use JSON which also works
                "task": "translation",
                "loading_method": "direct"
            }
        }
    
    def load_dataset_safely(self, dataset_info: Dict) -> Optional[Any]:
        """
        Load dataset with proper error handling for deprecated scripts
        """
        from datasets import load_dataset
        import warnings
        
        try:
            source = dataset_info["source"]
            config = dataset_info.get("config")
            
            # Method 1: Try loading with trust_remote_code=False (Parquet priority)
            try:
                logger.info(f"Loading {source} (Parquet mode)...")
                if config:
                    dataset = load_dataset(
                        source, 
                        config,
                        trust_remote_code=False,  # Avoid scripts
                        streaming=dataset_info.get("loading_method") == "streaming"
                    )
                else:
                    dataset = load_dataset(
                        source,
                        trust_remote_code=False,
                        streaming=dataset_info.get("loading_method") == "streaming"  
                    )
                logger.info(f"✓ Loaded via Parquet/direct method")
                return dataset
                
            except Exception as e:
                logger.debug(f"Parquet loading failed: {e}")
                
            # Method 2: Try with revision parameter (gets latest Parquet version)
            try:
                logger.info(f"Trying latest revision...")
                if config:
                    dataset = load_dataset(
                        source,
                        config, 
                        revision="main",  # Force latest version
                        trust_remote_code=False
                    )
                else:
                    dataset = load_dataset(
                        source,
                        revision="main",
                        trust_remote_code=False
                    )
                logger.info(f"✓ Loaded via latest revision")
                return dataset
                
            except Exception as e:
                logger.debug(f"Latest revision failed: {e}")
            
            # Method 3: Load from Hub API directly (bypasses scripts)
            try:
                logger.info(f"Loading via Hub API...")
                from huggingface_hub import hf_hub_download, HfApi
                
                api = HfApi()
                # Get dataset info
                dataset_info = api.dataset_info(source)
                
                # Look for Parquet files
                parquet_files = [
                    f for f in api.list_files_info(source, repo_type="dataset")
                    if f.rfilename.endswith('.parquet')
                ]
                
                if parquet_files:
                    # Download and load Parquet files directly
                    data_files = []
                    for file_info in parquet_files[:5]:  # Limit for testing
                        file_path = hf_hub_download(
                            repo_id=source,
                            filename=file_info.rfilename,
                            repo_type="dataset"
                        )
                        data_files.append(file_path)
                    
                    dataset = load_dataset("parquet", data_files=data_files)
                    logger.info(f"✓ Loaded {len(data_files)} Parquet files directly")
                    return dataset
                    
            except Exception as e:
                logger.debug(f"Hub API method failed: {e}")
            
            # Method 4: Last resort - with script but suppressed warnings
            try:
                logger.warning(f"Using legacy script loading for {source}...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if config:
                        dataset = load_dataset(
                            source,
                            config,
                            trust_remote_code=True  # Allow scripts as last resort
                        )
                    else:
                        dataset = load_dataset(
                            source,
                            trust_remote_code=True
                        )
                logger.info(f"⚠️ Loaded via legacy script")
                return dataset
                
            except Exception as e:
                logger.error(f"All loading methods failed: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load {dataset_info.get('source')}: {e}")
            return None
    
    def find_amharic_datasets_without_scripts(self):
        """
        Search for Amharic datasets that don't use deprecated scripts
        """
        from huggingface_hub import HfApi, list_datasets
        
        logger.info("Searching for modern Amharic datasets (Parquet/JSON format)...")
        
        api = HfApi()
        amharic_datasets = []
        
        # Search for datasets with Amharic in name or tags
        search_terms = ["amharic", "amh", "ethiopia", "geez", "tigrinya"]
        
        for term in search_terms:
            try:
                datasets = list_datasets(
                    search=term,
                    limit=20,
                    sort="downloads"  # Get popular ones
                )
                
                for dataset in datasets:
                    dataset_id = dataset.id
                    
                    try:
                        # Check if dataset has Parquet files
                        files = api.list_files_info(
                            dataset_id, 
                            repo_type="dataset"
                        )
                        
                        has_parquet = any(f.rfilename.endswith('.parquet') for f in files)
                        has_json = any(f.rfilename.endswith('.json') or f.rfilename.endswith('.jsonl') for f in files)
                        has_script = any(f.rfilename.endswith('.py') for f in files)
                        
                        if (has_parquet or has_json) and not has_script:
                            amharic_datasets.append({
                                "id": dataset_id,
                                "downloads": dataset.downloads,
                                "format": "parquet" if has_parquet else "json",
                                "tags": dataset.tags
                            })
                            logger.info(f"  ✓ Found: {dataset_id} (format: {'parquet' if has_parquet else 'json'})")
                            
                    except Exception as e:
                        logger.debug(f"Error checking {dataset_id}: {e}")
                        
            except Exception as e:
                logger.debug(f"Error searching for {term}: {e}")
        
        return amharic_datasets
    
    def convert_to_parquet(self, dataset, output_path: str):
        """
        Convert any dataset to Parquet format for faster loading
        """
        logger.info(f"Converting dataset to Parquet format...")
        
        try:
            # Save as Parquet
            dataset.to_parquet(output_path)
            logger.info(f"✓ Saved to {output_path}")
            
            # Now you can load it without scripts
            reloaded = load_dataset("parquet", data_files=output_path)
            return reloaded
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return None
    
    def collect_modern_amharic_data(self):
        """
        Collect data using modern methods that avoid deprecated scripts
        """
        collected_data = []
        
        # 1. Load Parquet-ready datasets
        parquet_datasets = self.get_parquet_datasets()
        
        for name, info in parquet_datasets.items():
            logger.info(f"\nCollecting {name}...")
            dataset = self.load_dataset_safely(info)
            
            if dataset:
                # Process the dataset
                if "train" in dataset:
                    data = dataset["train"]
                else:
                    data = dataset
                
                # Convert to list of dicts
                for item in data:
                    collected_data.append({
                        "source": name,
                        "task": info["task"],
                        "data": item
                    })
                    
                    if len(collected_data) >= 1000:  # Limit for testing
                        break
                
                logger.info(f"  Collected {len(collected_data)} examples so far")
        
        # 2. Find and use other modern datasets
        modern_datasets = self.find_amharic_datasets_without_scripts()
        logger.info(f"\nFound {len(modern_datasets)} modern Amharic datasets")
        
        return collected_data


def main():
    """Test modern collection methods"""
    collector = ModernAmharicDataCollector()
    
    # Find modern datasets
    logger.info("=" * 50)
    logger.info("Finding Amharic datasets without deprecated scripts...")
    logger.info("=" * 50)
    
    modern_datasets = collector.find_amharic_datasets_without_scripts()
    
    if modern_datasets:
        logger.info(f"\n✓ Found {len(modern_datasets)} modern datasets:")
        for ds in modern_datasets[:10]:
            logger.info(f"  - {ds['id']} ({ds['format']}, {ds['downloads']} downloads)")
    
    # Collect data
    logger.info("\n" + "=" * 50)
    logger.info("Collecting data from modern sources...")
    logger.info("=" * 50)
    
    data = collector.collect_modern_amharic_data()
    logger.info(f"\n✓ Total collected: {len(data)} examples")
    
    # Save
    output_path = Path("data/modern_amharic_data.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data[:100], f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Saved sample to {output_path}")

if __name__ == "__main__":
    main()
