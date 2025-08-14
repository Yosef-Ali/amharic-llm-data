# Troubleshooting Guide for Amharic LLM Training

## Common Issues and Solutions

### 1. Model Loading Errors

#### Error: `RepositoryNotFoundError: 401 Client Error`
```
RepositoryNotFoundError: 401 Client Error. (Request ID: ...)
Repository Not Found for url: https://huggingface.co/models/amharic-bloom560m/resolve/main/tokenizer_config.json
```

**Cause**: Trying to load a model that doesn't exist on Hugging Face Hub.

**Solution**: 
- Use correct model names from the available models list:
  - `distilgpt2` (82M parameters)
  - `gpt2` (124M parameters) 
  - `bigscience/bloom-560m` (560M parameters)
  - `bigscience/bloom-1b1` (1.1B parameters)
  - `EleutherAI/pythia-160m` (160M parameters)
  - `EleutherAI/pythia-410m` (410M parameters)

#### Error: Model path confusion
**Problem**: Confusing base model names with output paths.

**Explanation**:
- **Base Model**: `bigscience/bloom-560m` (from Hugging Face)
- **Output Path**: `models/amharic-bloom560m-finetuned` (your trained model)

**Correct Usage**:
```python
# Training (uses base model)
!python scripts/fast_training.py --train --model bloom-560m --output models/my-amharic-model

# Loading trained model (uses output path)
model_path = "models/my-amharic-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
```

### 2. Authentication Issues

#### Error: `Invalid username or password`
**Cause**: Some models require Hugging Face authentication.

**Solution**:
```python
# In Colab/Kaggle
from huggingface_hub import login
login()  # Enter your HF token when prompted

# Or set environment variable
import os
os.environ['HF_TOKEN'] = 'your_token_here'
```

### 3. Memory Issues

#### Error: `CUDA out of memory`
**Solutions**:
1. Use smaller models:
   ```python
   # Instead of bloom-1b1, use:
   --model distilgpt2  # Smallest option
   --model bloom-560m  # Medium option
   ```

2. Reduce batch size:
   ```python
   --batch-size 1  # Reduce from default
   ```

3. Use gradient checkpointing:
   ```python
   --gradient-checkpointing
   ```

### 4. Dataset Issues

#### Error: `FileNotFoundError: data/processed/train.jsonl`
**Solution**: Ensure dataset is properly set up:
```bash
# Check if files exist
ls data/processed/

# If missing, check main dataset
ls data/final_amharic_dataset.jsonl

# Run data preparation if needed
python scripts/analyze_data.py
```

### 5. Training Speed Issues

#### Problem: Training too slow
**Solutions**:
1. Use faster models:
   ```bash
   # Ultra fast (2-5 minutes)
   python scripts/fast_training.py --train --model distilgpt2 --steps 50
   
   # Fast (5-15 minutes)
   python scripts/fast_training.py --train --model gpt2 --steps 100
   ```

2. Reduce training steps:
   ```bash
   --steps 50    # Instead of 300+
   ```

3. Use LoRA (Low-Rank Adaptation):
   ```bash
   --use-lora    # Faster fine-tuning
   ```

### 6. Cloud Platform Issues

#### Google Colab
- **Session timeout**: Save models to Google Drive regularly
- **GPU not available**: Check runtime type (GPU enabled)
- **Disk space**: Clean up unnecessary files

#### Kaggle
- **Internet disabled**: Enable in settings for model downloads
- **Time limit**: Use faster training configurations
- **GPU quota**: Monitor usage in account settings

### 7. Model Quality Issues

#### Problem: Poor model performance
**Solutions**:
1. Increase training steps:
   ```bash
   --steps 500   # Instead of 50-100
   ```

2. Use larger models:
   ```bash
   --model bloom-1b1  # Instead of distilgpt2
   ```

3. Adjust learning rate:
   ```bash
   --learning-rate 5e-5  # Default is usually good
   ```

### 8. File Path Issues

#### Problem: Model not found after training
**Check**:
```python
import os
print("Available models:")
for root, dirs, files in os.walk("models"):
    for dir_name in dirs:
        print(f"  {os.path.join(root, dir_name)}")
```

**Solution**: Use correct path from training output.

## Quick Fixes Checklist

- [ ] Use correct model names (not custom names)
- [ ] Check file paths exist
- [ ] Verify dataset is in `data/processed/`
- [ ] Ensure sufficient GPU memory
- [ ] Use appropriate training steps for time constraints
- [ ] Save models to persistent storage (Drive/Kaggle Datasets)

## Getting Help

1. **Check logs**: Look for specific error messages
2. **Verify setup**: Run `python scripts/setup_cloud_training.py`
3. **Test with minimal config**: Use ultra-fast settings first
4. **Check GitHub issues**: Visit the repository for known issues

## Model Name Reference

| Display Name | Actual Model ID | Parameters | Speed |
|--------------|-----------------|------------|-------|
| distilgpt2 | `distilgpt2` | 82M | Fastest |
| gpt2 | `gpt2` | 124M | Fast |
| bloom-560m | `bigscience/bloom-560m` | 560M | Medium |
| bloom-1b1 | `bigscience/bloom-1b1` | 1.1B | Slow |
| pythia-160m | `EleutherAI/pythia-160m` | 160M | Fast |
| pythia-410m | `EleutherAI/pythia-410m` | 410M | Medium |

Remember: Always use the "Actual Model ID" for training, and your custom output path for loading trained models!