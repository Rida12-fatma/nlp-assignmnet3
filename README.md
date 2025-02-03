

# nlp-assignmnet3

Dataset: Opus100 (English-Italian)
The dataset used in this project is Opus100, a multilingual translation dataset containing 100 different language pairs. In this notebook, the English-Italian ("en-it") subset is used.

1. Dataset Source
The dataset is loaded from the datasets library.
It consists of sentence pairs where one sentence is in English and its corresponding translation is in Italian.
2. Structure of the Dataset
The dataset is divided into three subsets:

Training Set: Used for model training.
Validation Set: Used for tuning model parameters.
Test Set: Used for final evaluation.
Each data sample contains:

json
Copy
Edit
{
  "translation": {
    "en": "Hello, how are you?",
    "it": "Ciao, come stai?"
  }
}
Where:

"en" represents the English text.
"it" represents the corresponding Italian translation.
3. Data Loading and Preprocessing
The dataset is loaded using:

python
Copy
Edit
from datasets import load_dataset

dataset = load_dataset("opus100", "en-it")
To ensure that only valid translations are included, a filtering step is applied:

python
Copy
Edit
dataset = dataset.filter(lambda example: 'en' in example['translation'] and 'it' in example['translation'])
4. Dataset Statistics
The total number of sentence pairs in the dataset is varied based on the language pair.
The training set contains the majority of the data, with smaller validation and test subsets.

Translation API

This is a simple Flask-based API that translates English text into Italian using the Helsinki-NLP MarianMT model.

## Requirements

Ensure you have the following dependencies installed:
```sh
pip install torch transformers datasets flask
```

## How to Run

1. Clone this repository or download the script.
2. Install dependencies using the command above.
3. Run the application:
```sh
python app.py
```
4. The server will start at [http://0.0.0.0:5000/](http://0.0.0.0:5000/).

## API Endpoint

### Translate Text

- **Endpoint:** POST `/translate`
- **Request Body (JSON):**
  ```json
  {
      "text": "Hello, how are you?"
  }
  ```

- **Response JSON:**
  ```json
  {
      "translated_text": "Ciao, come stai?"
  }
  ```

## Notes

- The translation model used is Helsinki-NLP/opus-mt-en-it.
- You can modify the model to support other language pairs.

## License

This project is released under the MIT License.
