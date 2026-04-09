
# AI Medical Chatbot using BioT5

# Overview

This project focuses on building an AI-powered medical chatbot that answers patient queries using a fine-tuned BioT5 model. The system uses Natural Language Processing (NLP) techniques with PyTorch and Hugging Face Transformers.

The system supports:

- Conversational medical assistance
- Explanation of medical queries in natural language
- Evaluation using BLEU and F1-score metrics

# Objectives

- Provide instant responses to patient queries
- Build a fine-tuned NLP model using T5 architecture
- Preprocess patient-doctor dialogue datasets for training
- Evaluate chatbot responses for accuracy and relevance

# Dataset

The dataset consists of patient-doctor conversations in CSV format.

# Columns:

- Patient
- Doctor
- Description (context for the query)

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers (T5/BioT5)
- Flask
- Pandas / NumPy
- NLTK (BLEU score)
- Scikit-learn (F1-score)

# Data Preprocessing

- Load raw CSV dataset
- Clean dialogues and remove empty rows
- Combine patient query and description for context
- Format input-output pairs:
  - Input: `Patient: <query> [Context: <description>]`
  - Output: `Doctor: <response>`
- Save processed dialogues for training

# Model Architecture

- Base Model: Pretrained `T5` / `BioT5`
- Fine-tuned on medical dialogues
- Input: tokenized patient queries
- Output: generated doctor responses
- Uses beam search for improved response generation

# Training Strategy

1. Preprocess dataset and tokenize dialogues
2. Use memory-efficient dataset with pre-tokenization
3. Train model with mixed precision (BF16) for performance
4. Apply gradient checkpointing and gradient accumulation
5. Evaluate using BLEU and F1-score on held-out samples
6. Save final trained model and tokenizer

# Results

- Chatbot can generate meaningful doctor-like responses
- BLEU Score: ~0.65 (depends on dataset)
- F1-score: ~0.72 (approximate for relevant response detection)
- Supports batch evaluation for multiple dialogues

# Project Structure

AI_CHATBOT_BIOT5/
│── data/
│   ├── ai-medical-chatbot.csv
│   └── processed/
│       └── processed_dialogues.csv
│── models/
│   └── biot5_v1
│── templates/
│   └── index.html
│── app.py
│── train.py
│── evaluate.py
│── README.md

# How to Run:

# Install dependencies

```bash
pip install -r requirements.txt
````

# Run the Chatbot Web App

```bash
python app.py
```

 Open browser at `http://localhost:5000`

# Train the Model

```bash
python train.py
```

# Evaluate the Model

```bash
python evaluate.py
```

# Challenges

 Limited dataset size
 Complex medical terminology
 Generating accurate and context-aware responses

# Future Improvements

 Add multi-turn conversation support
 Integrate with real medical APIs for up-to-date info
 Deploy using Docker or cloud services
 Add explainability for responses (why the model answered that way)

# Author

Salma Khairy & Abdelaziz ElHelaly & Habiba Arafa
AI & Data Science Students – Zewail City

