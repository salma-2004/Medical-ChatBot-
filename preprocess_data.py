import pandas as pd
import os
from typing import Tuple, List
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the Kaggle dataset from a CSV file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        df = pd.read_csv(file_path)
        required_columns = ["Patient", "Doctor", "Description"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}")
        logger.info(f"Loaded dataset with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def preprocess_dialogues(df: pd.DataFrame, use_description: bool = True) -> List[Tuple[str, str]]:
    """Preprocess dialogues into input-output pairs for BioT5."""
    try:
        dialogue_pairs = []
        for _, row in df.iterrows():
            patient_query = str(row["Patient"]).strip()
            doctor_response = str(row["Doctor"]).strip()
            if not patient_query or not doctor_response:
                continue  # Skip empty or invalid rows
            
            # Combine patient query with description (if used) for context
            if use_description and str(row["Description"]).strip():
                input_text = f"Patient: {patient_query} [Context: {row['Description']}]"
            else:
                input_text = f"Patient: {patient_query}"
            
            # Format output as doctor response
            output_text = f"Doctor: {doctor_response}"
            dialogue_pairs.append((input_text, output_text))
        
        logger.info(f"Preprocessed {len(dialogue_pairs)} dialogue pairs")
        return dialogue_pairs
    except Exception as e:
        logger.error(f"Error preprocessing dialogues: {str(e)}")
        raise

def save_processed_data(pairs: List[Tuple[str, str]], output_path: str) -> None:
    """Save preprocessed dialogue pairs to a CSV file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_df = pd.DataFrame(pairs, columns=["input", "output"])
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved preprocessed data to {output_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise

def main() -> None:
    """Main function to preprocess the dataset."""
    raw_data_path = "C:/Users/hzezo/OneDrive/Desktop/AI_CHATBOT_BIOT5/data/ai-medical-chatbot.csv"
    processed_data_path = "C:/Users/hzezo/OneDrive/Desktop/AI_CHATBOT_BIOT5/data/processed/processed_dialogues.csv"
    
    # Load and preprocess
    df = load_dataset(raw_data_path)
    dialogue_pairs = preprocess_dialogues(df, use_description=True)
    
    # Save processed data
    save_processed_data(dialogue_pairs, processed_data_path)

if __name__ == "__main__":
    main()