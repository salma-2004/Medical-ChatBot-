from flask import Flask, request, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask with explicit template folder
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '../templates'))

def load_model_and_tokenizer(model_path: str):
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

# Load model and tokenizer
MODEL_PATH = "C:/Users/hzezo/OneDrive/Desktop/AI_CHATBOT_BIOT5/models/biot5_v1"
model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

@app.route('/', methods=['GET'])
def home():
    """Render the home page with the chatbot interface."""
    logger.info("Rendering index.html")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Generate a doctor response for a patient query."""
    try:
        logger.info("Received predict request")
        query = request.form.get('query')
        if not query:
            logger.error("Missing 'query' in request")
            return render_template('index.html', error="Missing query", query=query)

        logger.info(f"Processing query: {query}")
        input_text = f"Patient: {query}"
        
        # Tokenize and generate response
        inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=50, num_beams=3)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated response: {response}")
        
        # Clean response
        if response.startswith("Doctor:"):
            response = response[len("Doctor:"):].strip()
        
        return render_template('index.html', response=f"Doctor: {response}", query=query)
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return render_template('index.html', error=str(e), query=query)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)