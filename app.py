import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the opus100 dataset (Modify as per your requirement)
dataset = load_dataset("opus100", "en-it")

def preprocess_data(dataset):
    """Extracts English-Italian translation pairs."""
    en_data = [example['translation']['en'] for example in dataset['train']]
    it_data = [example['translation']['it'] for example in dataset['train']]
    return en_data, it_data

en_data, it_data = preprocess_data(dataset)

# Load pre-trained translation model
model_name = "Helsinki-NLP/opus-mt-en-it"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text):
    """Translates English text to Italian using the MarianMT model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

@app.route("/translate", methods=["POST"])
def translate():
    """API endpoint to translate text."""
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    translation = translate_text(text)
    return jsonify({"translated_text": translation})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
