# 🚀 Cloud Training Setup Guide

This guide helps you train your Amharic LLM using free cloud resources. We've removed all local testing to focus on fast, GPU-powered cloud training.

## 🎯 Quick Start Options

### Option 1: Google Colab (Recommended)
- **GPU**: Tesla T4 (16GB VRAM)
- **Time Limit**: 12 hours
- **Storage**: 100GB temporary
- **Setup Time**: 2-3 minutes

### Option 2: Kaggle Notebooks
- **GPU**: Tesla P100 (16GB VRAM)
- **Time Limit**: 30 hours/week
- **Storage**: 20GB persistent
- **Setup Time**: 1-2 minutes

## 📊 Training Time Estimates

| Model | Parameters | Colab Time | Kaggle Time | Quality |
|-------|------------|------------|-------------|----------|
| DistilGPT2 | 82M | 10-15 min | 8-12 min | ⭐⭐ |
| Bloom-560M | 560M | 30-45 min | 25-35 min | ⭐⭐⭐ |
| Bloom-1B1 | 1.1B | 1-2 hours | 45-90 min | ⭐⭐⭐⭐ |
| Phi-3.5-mini | 3.8B | 2-4 hours | 1.5-3 hours | ⭐⭐⭐⭐⭐ |

## 🔥 Google Colab Setup

### Step 1: Open Colab
1. Go to [Google Colab](https://colab.research.google.com)
2. Sign in with your Google account
3. Upload the notebook: `notebooks/Amharic_LLM_Training_Colab.ipynb`

### Step 2: Enable GPU
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. GPU type → T4 (free tier)
4. Save

### Step 3: Connect Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Upload Your Data
```bash
# Option A: Upload to Google Drive (RECOMMENDED)
# 1. Upload your entire 'amharic-llm-data' folder to Google Drive
# 2. Run: !cp -r '/content/drive/MyDrive/amharic-llm-data' /content/
# 3. Run: %cd /content/amharic-llm-data

# Option B: Clone from GitHub (if you've pushed your data)
!git clone https://github.com/Yosef-Ali/amharic-llm-data.git
%cd amharic-llm-data
```

### Step 5: Start Training
```python
# Quick test (10 minutes)
!python scripts/fast_training.py --train --model distilgpt2 --steps 100 --output /content/drive/MyDrive/models/amharic-distilgpt2

# Production quality (1-2 hours)
!python scripts/fast_training.py --train --model bloom-1b1 --steps 500 --output /content/drive/MyDrive/models/amharic-bloom1b1
```

## ⚡ Kaggle Setup

### Step 1: Create Account
1. Go to [Kaggle](https://www.kaggle.com)
2. Sign up/Sign in
3. Verify your phone number (required for GPU)

### Step 2: Create New Notebook
1. Click "Create" → "New Notebook"
2. Settings → Accelerator → GPU
3. Settings → Internet → On
4. Upload the notebook: `notebooks/Amharic_LLM_Training_Kaggle.ipynb`

### Step 3: Upload Dataset
```bash
# Option A: Create Kaggle Dataset (RECOMMENDED)
# 1. Create new Kaggle dataset
# 2. Upload your entire 'amharic-llm-data' folder
# 3. Add dataset to notebook
# 4. Run: !cp -r /kaggle/input/amharic-llm-data/* .

# Option B: Clone from GitHub (if you've pushed your data)
!git clone https://github.com/Yosef-Ali/amharic-llm-data.git
%cd amharic-llm-data
```

### Step 4: Start Training
```python
# Quick test (8 minutes)
!python scripts/fast_training.py --train --model distilgpt2 --steps 100 --output models/amharic-distilgpt2

# Best quality (1.5 hours)
!python scripts/fast_training.py --train --model bloom-1b1 --steps 500 --output models/amharic-bloom1b1
```

## 🎯 Recommended Training Strategy

### Phase 1: Quick Validation (15 minutes)
```bash
# Test the pipeline works
python scripts/fast_training.py --train --model distilgpt2 --steps 50 --output models/test
python scripts/fast_training.py --test --output models/test
```

### Phase 2: Balanced Training (45 minutes)
```bash
# Good quality for most use cases
python scripts/fast_training.py --train --model bloom-560m --steps 300 --output models/amharic-balanced
```

### Phase 3: Production Training (2 hours)
```bash
# Best quality for deployment
python scripts/fast_training.py --train --model bloom-1b1 --steps 500 --output models/amharic-production
```

## 📁 File Management

### Google Colab
```python
# Save to Google Drive
output_dir = "/content/drive/MyDrive/amharic-models"

# Download files
from google.colab import files
files.download('models/amharic-production.zip')
```

### Kaggle
```python
# Save to Kaggle output
output_dir = "/kaggle/working/models"

# Files automatically saved to output
# Download from notebook output section
```

## 🔧 Troubleshooting

### Common Issues

**GPU Not Available**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
```

**Out of Memory**
```python
# Reduce batch size
--per_device_train_batch_size 1
--gradient_accumulation_steps 8

# Use smaller model
--model distilgpt2  # instead of bloom-1b1
```

**Slow Training**
```python
# Reduce steps
--steps 100  # instead of 500

# Use gradient checkpointing
--gradient_checkpointing True
```

**Connection Timeout**
```python
# Save checkpoints frequently
--save_steps 50
--logging_steps 10
```

### Performance Tips

1. **Use Mixed Precision**: Automatically enabled on GPU
2. **Gradient Checkpointing**: Saves memory
3. **Batch Size Optimization**: Start small, increase gradually
4. **Regular Saving**: Save every 50-100 steps
5. **Monitor Resources**: Check GPU memory usage

## 📊 Expected Results

After training, you should have:

- ✅ Trained Amharic language model
- ✅ Model files (pytorch_model.bin, config.json, etc.)
- ✅ Training logs and metrics
- ✅ Test results with Amharic examples
- ✅ Ready-to-deploy model

## 🚀 Next Steps

1. **Download Models**: Save trained models locally
2. **Create Demo**: Build Gradio/Streamlit interface
3. **Deploy**: Use Hugging Face Spaces
4. **Evaluate**: Test with more Amharic data
5. **Improve**: Collect more training data

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section
2. Review notebook outputs for errors
3. Verify GPU availability
4. Ensure data is properly uploaded
5. Try with smaller models first

---

**Ready to start? Choose your platform and begin training! 🚀**

- [Open Google Colab Notebook](notebooks/Amharic_LLM_Training_Colab.ipynb)
- [Open Kaggle Notebook](notebooks/Amharic_LLM_Training_Kaggle.ipynb)