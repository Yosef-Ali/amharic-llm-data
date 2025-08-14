#!/usr/bin/env python3
"""
Optimized training configuration for the 5,883 example Amharic dataset
Designed for small-scale but high-quality training
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

class AmharicModelTrainer:
    """Optimized trainer for small Amharic dataset"""
    
    def __init__(self, dataset_path="data/processed", model_name="bigscience/bloom-560m"):
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        self.output_dir = f"models/{model_name.split('/')[-1]}-amharic"
        
    def get_optimized_config(self, dataset_size=5883):
        """Get training config optimized for dataset size"""
        
        if dataset_size < 10000:
            # Small dataset configuration
            return {
                "training_args": TrainingArguments(
                    output_dir=self.output_dir,
                    num_train_epochs=5,  # More epochs for small data
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=2,
                    gradient_accumulation_steps=8,  # Effective batch size = 16
                    warmup_steps=100,
                    learning_rate=2e-4,  # Higher LR for small data
                    logging_steps=20,
                    save_steps=200,
                    eval_steps=200,
                    evaluation_strategy="steps",
                    save_total_limit=3,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    fp16=True,
                    gradient_checkpointing=True,
                    report_to="none",
                    seed=42
                ),
                "lora_config": LoraConfig(
                    r=16,  # Higher rank for small dataset
                    lora_alpha=32,
                    target_modules=["query_key_value"],  # BLOOM specific
                    lora_dropout=0.05,  # Lower dropout for small data
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                ),
                "optimizer_config": {
                    "weight_decay": 0.01,
                    "adam_epsilon": 1e-8,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.95
                }
            }
        else:
            # Larger dataset configuration
            return self.get_standard_config()
    
    def prepare_dataset(self):
        """Load and prepare the Amharic dataset"""
        
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        # Load the collected data
        dataset = load_dataset(
            'json',
            data_files={
                'train': str(self.dataset_path / 'train.jsonl'),
                'validation': str(self.dataset_path / 'validation.jsonl'),
                'test': str(self.dataset_path / 'test.jsonl')
            }
        )
        
        logger.info(f"Dataset loaded:")
        logger.info(f"  Train: {len(dataset['train'])} examples")
        logger.info(f"  Validation: {len(dataset['validation'])} examples")
        logger.info(f"  Test: {len(dataset['test'])} examples")
        
        return dataset
    
    def train(self):
        """Run optimized training for small Amharic dataset"""
        
        # Load dataset
        dataset = self.prepare_dataset()
        
        # Get optimized config
        config = self.get_optimized_config(len(dataset['train']))
        
        logger.info("Loading model and tokenizer...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Apply LoRA
        logger.info("Applying LoRA configuration...")
        model = get_peft_model(model, config['lora_config'])
        model.print_trainable_parameters()
        
        # Format dataset
        def format_instruction(example):
            text = f"""### የሚከተለውን መመሪያ መልስ:
### መመሪያ:
{example['instruction']}

### መልስ:
{example['response']}"""
            return {"text": text}
        
        formatted_dataset = dataset.map(format_instruction)
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=256  # Shorter for news classification
            )
        
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset["train"].column_names
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=config['training_args'],
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Train
        logger.info("Starting training...")
        logger.info(f"Total training steps: {len(dataset['train']) // config['training_args'].per_device_train_batch_size * config['training_args'].num_train_epochs}")
        
        trainer.train()
        
        # Save model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(self.output_dir)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
        
        logger.info(f"Test results:")
        for key, value in test_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return trainer, test_results
    
    def quick_inference_test(self, model_path=None):
        """Test the trained model with sample inputs"""
        
        if model_path is None:
            model_path = self.output_dir
        
        logger.info(f"Loading model from {model_path}")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test prompts
        test_prompts = [
            "የሚከተለውን ዜና መድብ: የኢትዮጵያ መንግስት አዲስ የኢኮኖሚ ፖሊሲ አወጣ",
            "ወደ ትክክለኛው ምድብ አስገባ: የአፍሪካ ህብረት ስብሰባ በአዲስ አበባ ተካሄደ",
            "ይህ ዜና የትኛው ምድብ ነው? የቴክኖሎጂ ኩባንያዎች በኢትዮጵያ ኢንቨስት ለማድረግ ፍላጎት አሳዩ"
        ]
        
        logger.info("\nInference Test Results:")
        logger.info("-" * 50)
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
    
    def calculate_training_time(self):
        """Estimate training time based on dataset size"""
        
        dataset = self.prepare_dataset()
        train_size = len(dataset['train'])
        
        # Rough estimates based on hardware
        estimates = {
            "CPU": train_size * 0.5 / 60,  # 0.5 sec per example
            "GTX_1060": train_size * 0.1 / 60,  # 0.1 sec per example
            "RTX_3060": train_size * 0.05 / 60,  # 0.05 sec per example
            "V100": train_size * 0.02 / 60  # 0.02 sec per example
        }
        
        logger.info("\nEstimated Training Time (5 epochs):")
        logger.info("-" * 40)
        for hardware, time_min in estimates.items():
            time_total = time_min * 5  # 5 epochs
            hours = int(time_total // 60)
            minutes = int(time_total % 60)
            logger.info(f"  {hardware:10} ~{hours}h {minutes}m")
        
        return estimates

def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Amharic model on collected dataset")
    parser.add_argument("--model", default="bigscience/bloom-560m", help="Base model to use")
    parser.add_argument("--dataset", default="data/processed", help="Path to dataset")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run inference test")
    parser.add_argument("--estimate", action="store_true", help="Estimate training time")
    
    args = parser.parse_args()
    
    trainer = AmharicModelTrainer(
        dataset_path=args.dataset,
        model_name=args.model
    )
    
    if args.estimate:
        trainer.calculate_training_time()
    
    if args.train:
        trainer.train()
    
    if args.test:
        trainer.quick_inference_test()
    
    if not any([args.train, args.test, args.estimate]):
        logger.info("Use --train to train, --test to test, or --estimate for time estimates")

if __name__ == "__main__":
    main()
