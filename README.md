![Alt Text](https://i.imgur.com/eS0tl9M.png)

# Part-of-Speech Tagging using Keras
## Overview
This repository contains an implementation of a Part-of-Speech (POS) tagging model using recurrent neural networks (Bidirectional LSTMs) and pre-trained word embeddings. The model uses the Brown Corpus for training data and GloVe embeddings fo word representations. Additionally, a Flask API has been developed to serve the trained model, allowing users to send sentences via HTTP requests and receive POS-tagged outputs
 

The project involves the following steps:
1. **Exploratory Data Analysis (EDA)**: Analyzing the Brown Corpus to understand the distribution of sentence lengths and POS tags.
2. **Data Preprocessing**: Preparing the data by tokenizing words and tags, and padding sequences to a fixed length.
3. **Model Building**: Constructing a deep learning model using Bidirectional LSTM layers with GloVe embeddings.
4. **Model Training**: Training the model on the preprocessed data and evaluating its performance.
5. **Visualization**: Visualizing the training and validation accuracy and loss over epochs.
6. **Deployment**: The trained model is integrated with a Flask-based API for easy inference.


## Dataset

The dataset used in this project is the **Brown Corpus**, which is a comprehensive collection of English text samples. It contains 57,340 tagged sentences, making it a suitable dataset for POS tagging tasks.

## Model Architecture

The model architecture consists of the following layers:

- **Embedding Layer**: Uses pre-trained GloVe embeddings to convert words into dense vectors.
- **Bidirectional LSTM Layers**: Two layers of Bidirectional LSTMs to capture contextual information from both past and future words.
- **Batch Normalization**: Applied after each LSTM layer to stabilize and speed up training.
- **Time Distributed Dense Layer**: A dense layer applied to each time step to predict the POS tag for each word.

## Usage
### To use the model:

You can view the model and run it directly on Kaggle [here](https://www.kaggle.com/code/beasttitan/pos-tagging). 

### To use the API:
- Intsall the dependecies in the requirments file:
   ```bash
   pip install -r requirements.txt
   ```
- Run the Flask server:
   ```bash
   python main.py
   ```
- Send a request:
   ```bash
   Invoke-WebRequest -Uri "http://127.0.0.1:5000/predict" -Method Post -Body @{sentence="This is a test."}
   ```
   

