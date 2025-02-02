# nlp-assignmnet3

Translation API

This is a simple Flask-based API that translates English text into Italian using the Helsinki-NLP MarianMT model.

Requirements

Ensure you have the following dependencies installed:
pip install torch transformers datasets flask
How to Run

Clone this repository or download the script.

Install dependencies using the command above.

Run the application:
python app.py
The server will start at http://0.0.0.0:5000/.
API Endpoint

Translate Text

Endpoint:
POST /translate
Request Body (JSON):
{
    "text": "Hello, how are you?"
}
Response Json
{
    "translated_text": "Ciao, come stai?"
}
Notes

The translation model used is Helsinki-NLP/opus-mt-en-it.

You can modify the model to support other language pairs.

License

This project is released under the MIT License.

