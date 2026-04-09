import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import f1_score
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationDataset(Dataset):
    """Dataset for evaluation with pre-tokenized inputs."""
    def __init__(self, df: pd.DataFrame, tokenizer: T5Tokenizer, max_length: int = 256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = [
            tokenizer(
                text=row["input"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ) for _, row in df.iterrows()
        ]
        self.outputs = df["output"].tolist()
        logger.info(f"Preprocessed {len(self.df)} evaluation samples")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[idx]["input_ids"].squeeze(),
            "attention_mask": self.inputs[idx]["attention_mask"].squeeze(),
            "output": self.outputs[idx]
        }

def load_model_and_tokenizer(model_path: str) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """Load the fine-tuned T5 model and tokenizer."""
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        logger.info(f"Model loaded on {device}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def compute_metrics(data_path: str, model_path: str, batch_size: int = 16, num_samples: int = 500) -> Dict[str, float]:
    """Compute BLEU and F1-score for model predictions with batched inference."""
    try:
        # Load data
        df = pd.read_csv(data_path)
        df = df.sample(n=min(num_samples, len(df)), random_state=42)  # Reduced sample size
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # Create dataset and dataloader
        dataset = EvaluationDataset(df, tokenizer, max_length=256)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        bleu_scores = []
        predicted_labels = []
        true_labels = []
        smoothie = SmoothingFunction().method1  # Smoothing for BLEU
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            
            # Generate predictions in batch
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=50,  # Reduced for speed
                    num_beams=3  # Reduced for speed
                )
            predicted_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Process each prediction
            for pred, true in zip(predicted_responses, batch["output"]):
                # Clean responses
                if pred.startswith("Doctor:"):
                    pred = pred[len("Doctor:"):].strip()
                if true.startswith("Doctor:"):
                    true = true[len("Doctor:"):].strip()
                
                # Compute BLEU
                reference = [true.split()]
                candidate = pred.split()
                bleu = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
                bleu_scores.append(bleu)
                
                # Compute F1-score
                true_terms = set(true.lower().split())
                pred_terms = set(pred.lower().split())
                is_correct = len(true_terms.intersection(pred_terms)) / len(true_terms) > 0.5
                predicted_labels.append(1 if is_correct else 0)
                true_labels.append(1)  # Ground truth is correct
        
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        f1 = f1_score(true_labels, predicted_labels, average="binary")
        
        logger.info(f"Average BLEU: {avg_bleu:.4f}, F1-score: {f1:.4f}")
        return {"bleu": avg_bleu, "f1_score": f1}
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        raise

def main() -> None:
    """Main function to evaluate the model."""
    data_path = "C:/Users/hzezo/OneDrive/Desktop/AI_CHATBOT_BIOT5/data/processed/processed_dialogues.csv"
    model_path = "C:/Users/hzezo/OneDrive/Desktop/AI_CHATBOT_BIOT5/models/biot5_v1"
    metrics = compute_metrics(data_path, model_path, batch_size=16, num_samples=500)
    print(f"Evaluation Metrics: {metrics}")

if __name__ == "__main__":
    main()