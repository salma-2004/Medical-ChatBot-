import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForSeq2Seq
)
from accelerate import Accelerator
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedMedicalDataset(Dataset):
    """Memory-efficient dataset with pre-tokenization"""
    def __init__(self, data_path: str, tokenizer: T5Tokenizer, max_length: int = 256):
        try:
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.df = pd.read_csv(data_path)
            
            # Pre-tokenize all data during initialization
            self.examples = [
                self.tokenizer(
                    text=row["input"],
                    text_target=row["output"],
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt"
                )
                for _, row in self.df.iterrows()
            ]
            logger.info(f"Loaded and preprocessed {len(self.examples)} examples")

        except Exception as e:
            logger.error(f"Dataset initialization failed: {str(e)}")
            raise

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx]["input_ids"].squeeze(),
            "attention_mask": self.examples[idx]["attention_mask"].squeeze(),
            "labels": self.examples[idx]["labels"].squeeze()
        }

def train_model(data_path: str, output_dir: str):
    try:
        # Configuration for RTX 4060 (8GB VRAM)
        MODEL_NAME = "google/flan-t5-base"
        MAX_LENGTH = 256
        BATCH_SIZE = 4  # Can increase to 8 if using gradient accumulation
        GRADIENT_ACCUM_STEPS = 2

        # Initialize accelerator with BF16 mixed precision
        accelerator = Accelerator(mixed_precision="bf16")
        
        # Load model and tokenizer
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)  # Add legacy=False
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        
        # Prepare dataset
        dataset = OptimizedMedicalDataset(data_path, tokenizer, MAX_LENGTH)
        train_size = int(0.9 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )

        # Optimized data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,  # Better for tensor cores
            padding="longest",
            return_tensors="pt"
        )
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            logging_strategy="steps",
            logging_steps=100,
            learning_rate=3e-5,
            per_device_train_batch_size=8,  # Increased from 4
            per_device_eval_batch_size=16,
            num_train_epochs=1,  # Reduced from 3
            max_steps=5000,  # New: Stop early
            weight_decay=0.01,
            gradient_accumulation_steps=2,
            bf16=True,
            optim="adamw_torch_fused",
            warmup_ratio=0.1,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            report_to="none",
        )





        # Initialize and prepare trainer
        trainer = accelerator.prepare(Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        ))



        

        # Start training
        logger.info("Starting optimized training...")
        trainer.train()
        
        # Save final model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    DATA_PATH = "C:/Users/hzezo/OneDrive/Desktop/AI_CHATBOT_BIOT5/data/processed/processed_dialogues.csv"
    MODEL_DIR = "C:/Users/hzezo/OneDrive/Desktop/AI_CHATBOT_BIOT5/models/biot5_v1"
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_model(DATA_PATH, MODEL_DIR)