import numpy as np # linear algebra
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from flask import Flask, request


model = load_model("model.h5")
with open('word_tokenizer.pkl', 'rb') as f:
    word_tokenizer = pickle.load(f)

with open('tag_tokenizer.pkl', 'rb') as f:
    tag_tokenizer = pickle.load(f)


tag_dict = tag_tokenizer.word_index

# Swap the keys and values in the dict
tag_dict = {v: k for k, v in tag_dict.items()}

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    x = request.form.get("sentence")

    # Split the sentence to words
    x = re.findall(r"\w+|[^\w\s]", x)
    input_size = len(x)

    # make the list 2d as the model is expecting a batch
    x = [x]

    # Tokenize and pad the sequence
    x = word_tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=50, padding='post')

    predictions = model.predict(x)

    predictions = np.argmax(predictions, axis=2)
    predictions = predictions.flatten()
    predictions = predictions[:input_size]

    predictions = [tag_dict[key] for key in predictions]

    return predictions

if __name__ == "__main__":
    app.run(debug=True)
