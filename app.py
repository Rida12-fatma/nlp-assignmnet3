import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Step 1: Define Attention Mechanisms

class GeneralAttention(nn.Module):
    def __init__(self, input_dim):
        super(GeneralAttention, self).__init__()
        self.attn = nn.Linear(input_dim, input_dim, bias=False)
    
    def forward(self, query, values):
        scores = torch.bmm(query, self.attn(values).transpose(1, 2))  # (batch_size, 1, seq_len)
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

# Step 2: Define Translation Model with Attention Mechanisms

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_type="general"):
        super(Seq2SeqWithAttention, self).__init__()
        self.attention_type = attention_type
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Initialize attention
        if attention_type == "general":
            self.attention = GeneralAttention(hidden_dim)
        elif attention_type == "multiplicative":
            self.attention = MultiplicativeAttention(hidden_dim, hidden_dim)
        elif attention_type == "additive":
            self.attention = AdditiveAttention(hidden_dim, hidden_dim)
    
    def forward(self, src, tgt):
        # Encode the input sequence
        output, (hidden, cell) = self.rnn(src)
        
        # Apply attention mechanism
        context, attention_weights = self.attention(hidden[-1], output)
        
        return context, attention_weights

# Step 3: Set up MarianMT for Translation

model_name = "Helsinki-NLP/opus-mt-en-it"  # English to Italian
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translation

# Step 4: Setup Evaluation and Visualization Functions

def plot_attention_map(attention_weights, input_sentence, output_sentence, attention_type):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attention_weights.squeeze().cpu().detach().numpy(), cmap='viridis', xticklabels=input_sentence, yticklabels=output_sentence, ax=ax)
    ax.set_title(f'{attention_type} Attention Map')
    plt.close(fig)
    return fig

# Step 5: Streamlit Interface

st.title("English to Italian Translation with Attention Mechanisms")

# User input for text
text = st.text_area("Enter English Text:")

# Dropdown for attention mechanism choice
attention_type = st.selectbox("Choose Attention Mechanism:", ["general", "multiplicative", "additive"])

# Translate and display results
if text:
    # Translate the text
    translated_text = translate_text(text, model, tokenizer)
    
    # Display the translation
    st.subheader("Translation")
    st.write(translated_text)

    # Visualize attention map (using dummy attention weights for demo)
    input_sentence = text.split()
    output_sentence = translated_text.split()
    attention_weights = torch.rand(len(output_sentence), len(input_sentence))  # Random weights for demo
    fig = plot_attention_map(attention_weights, input_sentence, output_sentence, attention_type)
    
    # Display the attention map
    st.subheader("Attention Map")
    st.pyplot(fig)
