# app.py

import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
import re
import nltk
from unidecode import unidecode
import stanza
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn

# Data Loading
dataset = load_dataset("opus100", "en-it")
dataset = dataset.filter(lambda example: example['translation'].keys() >= {'en', 'it'})

# Print some examples to see the structure
print(dataset['train'][0])

# Further process the dataset for translation tasks
en_data = [example['translation']['en'] for example in dataset['train']]
it_data = [example['translation']['it'] for example in dataset['train']]

print(dataset)

# Load the pre-trained MarianMT model for EN-IT
model_name = 'Helsinki-NLP/opus-mt-en-it'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Text Normalization
def normalize_english(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def normalize_italian(text):
    text = unidecode(text.lower())
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Tokenization with Stanza
stanza.download('it')
nlp_stanza = stanza.Pipeline('it')
doc = nlp_stanza("Ciao, come stai?")
tokens = [word.text for word in doc.words]
print(tokens)

# Tokenization with spaCy
nltk.download('punkt')
nlp_spacy = spacy.load('it_core_news_sm')

def tokenize_english(text):
    return nltk.word_tokenize(text)

def tokenize_italian(text):
    doc = nlp_spacy(text)
    return [token.text for token in doc]

# Example usage
english_tokens = tokenize_english("Hello, how are you?")
italian_tokens = tokenize_italian("Ciao, come stai?")
print(english_tokens, italian_tokens)

# Word Segmentation
def tokenize_data(example):
    example['translation']['en'] = tokenize_english(example['translation']['en'])
    example['translation']['it'] = tokenize_italian(example['translation']['it'])
    return example

dataset = dataset.map(tokenize_data)
train_dataset = dataset['train'].map(lambda example: {'input': example['translation']['en'], 'target': example['translation']['it']})

# Attention Mechanisms
class GeneralAttention(nn.Module):
    def __init__(self, input_dim):
        super(GeneralAttention, self).__init__()
        self.attn = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, query, values):
        scores = torch.bmm(query, self.attn(values).transpose(1, 2))
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights, values)
        return context, attention_weights

class MultiplicativeAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MultiplicativeAttention, self).__init__()
        self.attn = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, query, values):
        scores = torch.bmm(query, self.attn(values).transpose(1, 2))
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights, values)
        return context, attention_weights

class AdditiveAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, query, values):
        query_proj = self.W2(query).unsqueeze(1)
        keys_proj = self.W1(values)
        energy = torch.tanh(query_proj + keys_proj)
        scores = torch.matmul(energy, self.v)
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.transpose(1, 2), values)
        return context, attention_weights

# Plotting Loss Curves
def plot_loss_curves(train_losses, val_losses, attention_type):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{attention_type} Attention - Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Example usage after training
train_losses_general = [1.0, 0.8, 0.6, 0.4, 0.2]
val_losses_general = [1.2, 0.9, 0.7, 0.5, 0.3]
train_losses_multiplicative = [1.1, 0.7, 0.5, 0.3, 0.1]
val_losses_multiplicative = [1.3, 0.8, 0.6, 0.4, 0.2]
train_losses_additive = [1.2, 0.9, 0.7, 0.5, 0.3]
val_losses_additive = [1.4, 1.0, 0.8, 0.6, 0.4]

plot_loss_curves(train_losses_general, val_losses_general, 'General')
plot_loss_curves(train_losses_multiplicative, val_losses_multiplicative, 'Multiplicative')
plot_loss_curves(train_losses_additive, val_losses_additive, 'Additive')

# Plotting Attention Maps
def plot_attention_map(attention_weights, input_sentence, output_sentence, attention_type):
    attention_weights = attention_weights.squeeze().cpu().detach().numpy()

    if attention_weights.ndim == 1:
        attention_weights = attention_weights.reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='viridis', xticklabels=input_sentence, yticklabels=output_sentence, ax=ax)
    ax.set_title(f'{attention_type} Attention Map')
    plt.show()

input_sentence = ["Hello", "world", "!"]
output_sentence = ["Ciao", "mondo", "!"]
attention_weights_general = torch.randn(3, 3)

plot_attention_map(attention_weights_general, input_sentence, output_sentence, 'General')
```

Make sure to replace any placeholder paths or example data with the actual data and paths as needed.
