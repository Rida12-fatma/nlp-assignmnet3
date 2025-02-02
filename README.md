# nlp-assignmnet3

English to Urdu Translator Web App
This repository contains a Python Flask web application that translates English sentences to Urdu using a pre-trained deep learning model. The model is built using TensorFlow/Keras and is trained on a dataset of aligned English and Urdu sentences.

Project Structure
The project is organized as follows:

app.py: The Flask web application code, including model loading, text preprocessing, and translation functions. Translator.ipynb: Jupyter Notebook containing the entire time series analysis pipeline. templates/index.html: HTML template for the web app's user interface.

Data Analysis and Preprocessing
The Translator.ipynb notebook starts with data analysis and preprocessing. It reads English and Urdu sentences from the text files, removes newline characters, and calculates word counts.

Word Cloud Visualization
The notebook uses Word Cloud to visualize the most common words in the English sentences.

Tokenization and Padding
The English and Urdu sentences are tokenized and padded to create input sequences for the machine learning model.

Model Training
The notebook uses a Bidirectional GRU model to train on the tokenized English and Urdu sentences. It compiles the model, trains it on the data, and evaluates its performance.

Model Evaluation
The trained model is evaluated on the test set, and accuracy and loss graphs are plotted.

Saving the Model
The trained model is saved as english_to_urdu_translator_final.h5 for later use.

Contributions
Contributions to the project are welcome. If you find any bugs or have ideas for improvements, feel free to open an issue or submit a pull request.
