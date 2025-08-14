#!/usr/bin/env python3
"""
Simple training example for Amharic dataset
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_amharic_dataset(dataset_path="data/processed"):
    """Load the processed Amharic dataset"""
    return load_dataset("json", data_files={
        "train": f"{dataset_path}/train.jsonl",
        "validation": f"{dataset_path}/validation.jsonl",
        "test": f"{dataset_path}/test.jsonl"
    })

def format_instruction(example):
    """Format instruction for training"""
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

def train_model(model_name="bigscience/bloom-560m", output_dir="models/amharic-lora", dataset_path="data/processed"):
    """Train Amharic model with LoRA"""
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    
    # Check if MPS is available (Apple Silicon)
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Apple Silicon) device")
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info("Using CUDA device")
    else:
        device = "cpu"
        logger.info("Using CPU device")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    if device != "cuda":
        model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    logger.info("Loading Amharic dataset...")
    dataset = load_amharic_dataset(dataset_path)
    
    # Format dataset
    formatted_dataset = dataset.map(format_instruction)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=formatted_dataset["train"].column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=device == "cuda",  # Only use fp16 on CUDA
        push_to_hub=False,
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

def test_model(model_path="models/amharic-lora"):
    """Test the trained model"""
    from peft import PeftModel
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-1b1",
        load_in_4bit=True,
        device_map="auto"
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test prompts
    test_prompts = [
        "ስለ ኢትዮጵያ ታሪክ አጭር ማጠቃለያ ስጥ",
        "የሚከተለውን ወደ እንግሊዝኛ ተርጉም: ሰላም ዓለም",
        "አዲስ አበባ ምን ያህል ሰው ይኖራል?"
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--test", action="store_true", help="Test model")
    parser.add_argument("--model", default="bigscience/bloom-1b1", help="Base model")
    parser.add_argument("--output", default="models/amharic-lora", help="Output directory")
    
    args = parser.parse_args()
    
    if args.train:
        train_model(model_name=args.model, output_dir=args.output)
    
    if args.test:
        test_model(model_path=args.output)
    
    if not args.train and not args.test:
        print("Use --train to train or --test to test the model")
