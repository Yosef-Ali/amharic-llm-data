#!/usr/bin/env python3
"""
Fast Training Script for Amharic LLM
Optimized for local training with limited resources
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import logging
from pathlib import Path
import argparse
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Recommended small models for fast training
SMALL_MODELS = {
    "gpt2": "gpt2",  # 124M parameters
    "distilgpt2": "distilgpt2",  # 82M parameters
    "bloom-560m": "bigscience/bloom-560m",  # 560M parameters
    "bloom-1b1": "bigscience/bloom-1b1",  # 1.1B parameters
    "pythia-160m": "EleutherAI/pythia-160m",  # 160M parameters
    "pythia-410m": "EleutherAI/pythia-410m",  # 410M parameters
}

def load_amharic_dataset(dataset_path="data/processed"):
    """Load processed Amharic dataset"""
    return load_dataset("json", data_files={
        "train": f"{dataset_path}/train.jsonl",
        "validation": f"{dataset_path}/validation.jsonl",
        "test": f"{dataset_path}/test.jsonl"
    })

def format_instruction(example):
    """Format instruction for training"""
    return {
        "text": f"Instruction: {example['instruction']}\nResponse: {example['response']}"
    }

def get_device_info():
    """Get optimal device configuration"""
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Apple Silicon) device")
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        logger.info("Using CPU device")
    return device

def fast_train_model(model_name="distilgpt2", output_dir="models/amharic-fast", dataset_path="data/processed", max_steps=100):
    """Fast training with optimized settings"""
    
    start_time = time.time()
    device = get_device_info()
    
    # Load model and tokenizer with optimizations
    logger.info(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        use_cache=False  # Disable cache to save memory
    )
    
    if device != "cuda":
        model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA configuration for efficient fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Reduced rank for faster training
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"] if "bloom" in model_name.lower() else ["c_attn"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    logger.info("Loading Amharic dataset...")
    dataset = load_amharic_dataset(dataset_path)
    
    # Take a smaller subset for fast training
    train_size = min(1000, len(dataset["train"]))
    val_size = min(100, len(dataset["validation"]))
    
    small_dataset = {
        "train": dataset["train"].select(range(train_size)),
        "validation": dataset["validation"].select(range(val_size))
    }
    
    # Format dataset
    formatted_dataset = {}
    for split in small_dataset:
        formatted_dataset[split] = small_dataset[split].map(format_instruction)
    
    # Tokenize with shorter sequences
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256  # Reduced from 512 for speed
        )
    
    tokenized_dataset = {}
    for split in formatted_dataset:
        tokenized_dataset[split] = formatted_dataset[split].map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset[split].column_names
        )
    
    # Optimized training arguments for speed
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,  # Use steps instead of epochs for faster training
        per_device_train_batch_size=2,  # Small batch size
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,  # Reduced for speed
        warmup_steps=10,  # Reduced warmup
        logging_steps=5,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=25,
        save_total_limit=2,
        load_best_model_at_end=False,  # Disable for speed
        fp16=device == "cuda",
        gradient_checkpointing=True,  # Save memory
        dataloader_num_workers=0,  # Reduce for stability
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator
    )
    
    # Train
    logger.info(f"Starting fast training with {train_size} examples for {max_steps} steps...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    end_time = time.time()
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
    
    return output_dir

def test_model(model_path="models/amharic-fast"):
    """Test the trained model"""
    logger.info(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Test with sample Amharic instruction
    test_instruction = "Instruction: የአማርኛ ቋንቋ ምንድን ነው?\nResponse:"
    
    inputs = tokenizer(test_instruction, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Model response: {response}")

def print_training_options():
    """Print available training options"""
    print("\n🚀 Fast Training Options for Amharic LLM:\n")
    
    print("1. ULTRA FAST (Recommended for testing):")
    print("   python scripts/fast_training.py --model distilgpt2 --steps 50")
    print("   ⏱️  ~2-5 minutes on Apple Silicon")
    
    print("\n2. FAST:")
    print("   python scripts/fast_training.py --model gpt2 --steps 100")
    print("   ⏱️  ~5-10 minutes on Apple Silicon")
    
    print("\n3. BALANCED:")
    print("   python scripts/fast_training.py --model bloom-560m --steps 200")
    print("   ⏱️  ~15-30 minutes on Apple Silicon")
    
    print("\n4. QUALITY (if you have time):")
    print("   python scripts/fast_training.py --model bloom-1b1 --steps 500")
    print("   ⏱️  ~1-2 hours on Apple Silicon")
    
    print("\n📊 Available Models:")
    for key, value in SMALL_MODELS.items():
        print(f"   {key}: {value}")
    
    print("\n☁️  For Faster Training, Consider:")
    print("   • Google Colab (Free GPU): https://colab.research.google.com")
    print("   • Kaggle Notebooks (Free GPU): https://kaggle.com/notebooks")
    print("   • Hugging Face Spaces (Free): https://huggingface.co/spaces")
    
    print("\n🔧 Memory Optimization Tips:")
    print("   • Close other applications")
    print("   • Use smaller batch sizes (--batch-size 1)")
    print("   • Reduce max sequence length")
    print("   • Use gradient checkpointing (enabled by default)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Amharic LLM Training")
    parser.add_argument("--model", default="distilgpt2", choices=list(SMALL_MODELS.keys()) + list(SMALL_MODELS.values()),
                       help="Model to use for training")
    parser.add_argument("--output", default="models/amharic-fast", help="Output directory")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--train", action="store_true", help="Start training")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    parser.add_argument("--options", action="store_true", help="Show training options")
    
    args = parser.parse_args()
    
    if args.options:
        print_training_options()
    elif args.train:
        # Resolve model name
        model_name = SMALL_MODELS.get(args.model, args.model)
        fast_train_model(model_name=model_name, output_dir=args.output, max_steps=args.steps)
    elif args.test:
        test_model(model_path=args.output)
    else:
        print("Use --options to see training options, --train to start training, or --test to test model")