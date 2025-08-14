# ☁️ Cloud Training Guide for Amharic LLM

Since local training is slow, here are **free** cloud alternatives for faster training:

## 🚀 Quick Solutions (Recommended)

### 1. Google Colab (FREE GPU)
**Best for: Quick experimentation and small models**

```python
# In Colab notebook:
!git clone https://github.com/your-username/amharic-llm-data.git
%cd amharic-llm-data
!pip install -r requirements.txt

# Fast training (5-10 minutes on T4 GPU)
!python scripts/fast_training.py --train --model distilgpt2 --steps 200
```

**Advantages:**
- Free T4 GPU (16GB VRAM)
- 12+ hours daily usage
- Pre-installed ML libraries
- Easy sharing and collaboration

**Setup Steps:**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create new notebook
3. Runtime → Change runtime type → GPU
4. Copy your code and run

### 2. Kaggle Notebooks (FREE GPU)
**Best for: Longer training sessions**

```bash
# In Kaggle notebook:
!git clone https://github.com/your-username/amharic-llm-data.git
%cd amharic-llm-data
!pip install transformers datasets peft accelerate

# Train with better model
!python scripts/fast_training.py --train --model bloom-1b1 --steps 500
```

**Advantages:**
- Free P100 or T4 GPU
- 30 hours/week GPU quota
- 20GB RAM
- Persistent datasets

**Setup Steps:**
1. Go to [kaggle.com/notebooks](https://kaggle.com/notebooks)
2. Create new notebook
3. Settings → Accelerator → GPU
4. Upload your data or clone repo

### 3. Hugging Face Spaces (FREE)
**Best for: Model deployment and sharing**

```python
# Create Space with Gradio interface
import gradio as gr
from transformers import pipeline

# Load your trained model
model = pipeline("text-generation", model="your-username/amharic-model")

def generate_text(prompt):
    return model(prompt, max_length=100)[0]['generated_text']

iface = gr.Interface(fn=generate_text, inputs="text", outputs="text")
iface.launch()
```

## 🔧 Optimization Strategies

### Local Training Optimizations

1. **Use Smaller Models:**
```bash
# Ultra fast (2-5 minutes)
python scripts/fast_training.py --train --model distilgpt2 --steps 50

# Fast (5-10 minutes)
python scripts/fast_training.py --train --model gpt2 --steps 100

# Balanced (15-30 minutes)
python scripts/fast_training.py --train --model bloom-560m --steps 200
```

2. **Reduce Dataset Size:**
```python
# In your training script, use subset:
train_dataset = dataset['train'].select(range(500))  # Use only 500 examples
val_dataset = dataset['validation'].select(range(50))  # Use only 50 examples
```

3. **Optimize Training Parameters:**
```python
training_args = TrainingArguments(
    max_steps=50,  # Very short training
    per_device_train_batch_size=1,  # Smallest batch
    gradient_accumulation_steps=4,  # Simulate larger batch
    fp16=True,  # Half precision (if supported)
    gradient_checkpointing=True,  # Save memory
    dataloader_num_workers=0,  # Reduce overhead
)
```

### Memory Management

```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()  # If using CUDA

# Use smaller sequence length
max_length = 128  # Instead of 512

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

## 📊 Training Time Comparison

| Environment | Model | Steps | Time | Quality |
|-------------|-------|-------|------|----------|
| MacBook M1 | Phi-3.5-mini | 100 | 2-3 hours | High |
| MacBook M1 | DistilGPT2 | 100 | 5-10 min | Medium |
| Colab T4 | Phi-3.5-mini | 100 | 15-20 min | High |
| Colab T4 | Bloom-1B1 | 500 | 30-45 min | High |
| Kaggle P100 | Llama-2-7B | 1000 | 1-2 hours | Very High |

## 🎯 Recommended Workflow

### Phase 1: Quick Validation (5-10 minutes)
```bash
# Test your pipeline works
python scripts/fast_training.py --train --model distilgpt2 --steps 50
python scripts/fast_training.py --test
```

### Phase 2: Better Model (30-60 minutes on cloud)
```bash
# Use Colab/Kaggle for better results
python scripts/fast_training.py --train --model bloom-1b1 --steps 500
```

### Phase 3: Production Model (2-4 hours on cloud)
```bash
# Use original script with optimizations
python scripts/train_example.py --train --model microsoft/Phi-3.5-mini-instruct
```

## 🔗 Cloud Setup Templates

### Colab Notebook Template
```python
# Mount Google Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

# Clone your repo
!git clone https://github.com/your-username/amharic-llm-data.git
%cd amharic-llm-data

# Install requirements
!pip install -r requirements.txt

# Check GPU
!nvidia-smi

# Fast training
!python scripts/fast_training.py --train --model bloom-560m --steps 200

# Save to Drive
!cp -r models/ /content/drive/MyDrive/amharic-models/
```

### Kaggle Notebook Template
```python
# Upload your dataset as Kaggle dataset first
import os
os.chdir('/kaggle/working')

# Clone repo
!git clone https://github.com/your-username/amharic-llm-data.git
%cd amharic-llm-data

# Copy dataset from Kaggle input
!cp -r /kaggle/input/amharic-dataset/* data/

# Install and train
!pip install transformers datasets peft accelerate
!python scripts/fast_training.py --train --model bloom-1b1 --steps 500
```

## 💡 Pro Tips

1. **Start Small:** Always test with `distilgpt2` first
2. **Use Cloud for Production:** Local for development, cloud for training
3. **Save Frequently:** Use checkpoints every 50-100 steps
4. **Monitor Resources:** Watch GPU memory and adjust batch size
5. **Version Control:** Push models to Hugging Face Hub

## 🆘 Troubleshooting

**Out of Memory:**
```bash
# Reduce batch size
python scripts/fast_training.py --train --model gpt2 --steps 100
# Or use gradient accumulation
```

**Too Slow:**
```bash
# Use smaller model
python scripts/fast_training.py --train --model distilgpt2 --steps 50
```

**Poor Quality:**
```bash
# Increase steps or use better model on cloud
python scripts/fast_training.py --train --model bloom-1b1 --steps 500
```

---

**Next Steps:**
1. Try the fast training script locally first
2. Set up Colab for GPU training
3. Upload your best model to Hugging Face
4. Create a demo with Gradio