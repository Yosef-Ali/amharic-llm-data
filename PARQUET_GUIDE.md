# 📊 Understanding Parquet Format & Deprecated Scripts Issue

## The Problem You're Facing

When you try to load datasets like:
```python
dataset = load_dataset("shmuhammad/AfriSenti-twitter-sentiment", "amh")
```

You get warnings:
```
FutureWarning: This dataset uses a deprecated dataset script which will be removed in the next major version of datasets.
```

## Why This Happens

### Old Way (Deprecated)
- Datasets used Python scripts (`dataset_script.py`)
- Security risk: arbitrary code execution
- Slower loading
- Maintenance burden

### New Way (Parquet Format)
- Data stored as `.parquet` files
- No code execution needed
- 10x faster loading
- Better compression

## Solutions

### Solution 1: Find Parquet-Ready Datasets

```python
# These work without warnings:
from datasets import load_dataset

# 1. OSCAR (already in Parquet)
dataset = load_dataset("oscar-corpus/OSCAR-2301", "am", streaming=True)

# 2. MC4 (Parquet format)
dataset = load_dataset("mc4", "am", streaming=True)

# 3. Wikipedia
dataset = load_dataset("wikimedia/wikipedia", "20231101.am")
```

### Solution 2: Load Datasets Despite Warnings

```python
import warnings
from datasets import load_dataset

# Suppress the warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    dataset = load_dataset(
        "shmuhammad/AfriSenti-twitter-sentiment", 
        "amh",
        trust_remote_code=True  # Required for script-based datasets
    )
```

### Solution 3: Convert to Parquet Locally

```python
# Load once with script
dataset = load_dataset("old_dataset", trust_remote_code=True)

# Save as Parquet
dataset.save_to_disk("./data/dataset_parquet")
# OR
dataset.to_parquet("./data/dataset.parquet")

# Next time, load from Parquet
dataset = load_dataset("parquet", data_files="./data/dataset.parquet")
```

### Solution 4: Use Direct Download

```python
from huggingface_hub import hf_hub_download
import pandas as pd

# Download the data file directly
file = hf_hub_download(
    repo_id="shmuhammad/AfriSenti-twitter-sentiment",
    filename="data/amh/train.tsv",
    repo_type="dataset"
)

# Load with pandas
df = pd.read_csv(file, sep='\t')
```

## Modern Amharic Datasets (Parquet/JSON)

Here are Amharic datasets that work without deprecation warnings:

### Already in Parquet Format:
1. **OSCAR-2301**: `oscar-corpus/OSCAR-2301`
2. **MC4**: `mc4` (config: "am")
3. **mC4**: `allenai/c4` (multilingual)
4. **Wikipedia**: `wikimedia/wikipedia`
5. **Common Crawl**: `allenai/c4`

### JSON Format (No Scripts):
1. **FLORES-200**: `facebook/flores` 
2. **MAFAND**: `masakhane/mafand`
3. **XLSum**: Some versions

### How to Check Format:

```python
from huggingface_hub import HfApi

api = HfApi()
files = api.list_files_info("dataset_name", repo_type="dataset")

for f in files:
    print(f.rfilename)
    # Look for .parquet, .json, or .arrow files (good)
    # .py files mean it uses scripts (deprecated)
```

## Best Practices for 2025

### 1. Prioritize Parquet Datasets
```python
# Good - direct Parquet loading
dataset = load_dataset("parquet", data_files="*.parquet")
```

### 2. Use Streaming for Large Datasets
```python
# Don't load everything into memory
dataset = load_dataset("oscar-corpus/OSCAR-2301", "am", streaming=True)

for example in dataset['train'].take(1000):
    process(example)
```

### 3. Cache Converted Datasets
```python
import os

cache_path = "./cache/dataset.parquet"

if os.path.exists(cache_path):
    # Load from cache
    dataset = load_dataset("parquet", data_files=cache_path)
else:
    # Load with script (once)
    dataset = load_dataset("old_dataset", trust_remote_code=True)
    # Save as Parquet
    dataset.to_parquet(cache_path)
```

### 4. Use the Modern Collector

Run the updated script:
```bash
python src/modern_collector.py
```

This will:
- Find Parquet-ready datasets automatically
- Handle deprecated scripts gracefully
- Convert to Parquet for future use

## Summary

**The Parquet shift is good!** It means:
- ✅ Faster loading (10x improvement)
- ✅ Better security (no arbitrary code)
- ✅ Smaller file sizes (better compression)
- ✅ Streaming support (don't need all RAM)

**Your options:**
1. Use modern datasets (OSCAR, MC4, etc.)
2. Suppress warnings and continue
3. Convert to Parquet once, use forever
4. Use the `modern_collector.py` script

The deprecation won't break your code immediately - you have time to migrate!
