import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

# Load your model here using TensorFlow/Keras
translator_model = tf.keras.models.load_model('english_to_urdu_translator_final.h5')

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def load_tokenizer(filename):
    with open(filename, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    text = [request.form['text']]
    translated_text = translate_text(text)
    print(translated_text)
    return render_template('index.html', translation=translated_text)

def translate_text(text):
    # Preprocess the input text (if required)
    # For example, if your model expects tokenized input, tokenize the text

    # Load the Urdu tokenizer
    english_tokenizer = load_tokenizer('english_tokenizer.pkl')
    urdu_tokenizer = load_tokenizer('urdu_tokenizer.pkl')
    
    text = english_tokenizer.texts_to_sequences(text)
    text = pad_sequences(text, maxlen=10,padding = 'post')  # Pad the sequence to match the expected input shape
    text = np.array(text)

    # Translate the text using your loaded model
    prediction = translator_model.predict(text,)
    translated_text = logits_to_text(prediction[0], urdu_tokenizer)

    return translated_text.replace('<PAD>','')

if __name__ == '__main__':
    app.run(debug=True)
