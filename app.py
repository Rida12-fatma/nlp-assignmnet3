try:
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    from datasets import load_dataset
    from flask import Flask, request, jsonify
except ModuleNotFoundError as e:
    import sys
    sys.stderr.write(f"Error: {e}. Please install missing dependencies using:\n\n    pip install torch transformers datasets flask\n")
    sys.exit(1)

app = Flask(__name__)

# Load the opus100 dataset with error handling
try:
    dataset = load_dataset("opus100", "en-it")
except Exception as e:
    sys.stderr.write(f"Error loading dataset: {e}\n")
    sys.exit(1)

def preprocess_data(dataset):
    """Extracts English-Italian translation pairs."""
    try:
        en_data = [example['translation']['en'] for example in dataset['train']]
        it_data = [example['translation']['it'] for example in dataset['train']]
        return en_data, it_data
    except KeyError as e:
        sys.stderr.write(f"Dataset format error: {e}\n")
        sys.exit(1)

en_data, it_data = preprocess_data(dataset)

# Load pre-trained translation model
model_name = "Helsinki-NLP/opus-mt-en-it"
try:
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
except Exception as e:
    sys.stderr.write(f"Error loading model: {e}\n")
    sys.exit(1)

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
