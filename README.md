![Alt Text](https://i.imgur.com/eS0tl9M.png)

# Part-of-Speech Tagging using Keras
## Overview
This repository contains an implementation of a Part-of-Speech (POS) tagging model using recurrent neural networks (Bidirectional LSTMs) and pre-trained word embeddings. The notebook uses the Brown Corpus for text data and GloVe embeddings fo word representations.
The notebook in this repository  was created on Kaggle, you can view the notebook and run it directly on Kaggle [here](https://www.kaggle.com/code/beasttitan/pos-tagging).  

The project involves the following steps:
1. **Exploratory Data Analysis (EDA)**: Analyzing the Brown Corpus to understand the distribution of sentence lengths and POS tags.
2. **Data Preprocessing**: Preparing the data by tokenizing words and tags, and padding sequences to a fixed length.
3. **Model Building**: Constructing a deep learning model using Bidirectional LSTM layers with GloVe embeddings.
4. **Model Training**: Training the model on the preprocessed data and evaluating its performance.
5. **Visualization**: Visualizing the training and validation accuracy and loss over epochs.

## Dataset

The dataset used in this project is the **Brown Corpus**, which is a comprehensive collection of English text samples. It contains 57,340 tagged sentences, making it a suitable dataset for POS tagging tasks.

## Model Architecture

The model architecture consists of the following layers:

- **Embedding Layer**: Uses pre-trained GloVe embeddings to convert words into dense vectors.
- **Bidirectional LSTM Layers**: Two layers of Bidirectional LSTMs to capture contextual information from both past and future words.
- **Batch Normalization**: Applied after each LSTM layer to stabilize and speed up training.
- **Time Distributed Dense Layer**: A dense layer applied to each time step to predict the POS tag for each word.

## Requirements

To run this project, you need the following Python libraries:

- `numpy`
- `pandas`
- `nltk`
- `tensorflow`
- `keras`
- `scikit-learn`
- `plotly`

You can install the required libraries using `pip`:

```bash  
pip install numpy pandas nltk tensorflow scikit-learn plotly
``` 
   

