# *Pun*derstand.py

A program written to detect puns in a sentence using k-Nearest Neighbour and Naive Bayes Classifier.

This is made for the Fundamentals of AI final project on Pun detection, location, and analysis with different models.

## Prerequisites

This program requires four dependencies, as listed below:

- `datasets`: Used to load and preprocess datasets.
- `scikit-learn`: Used for feature-extraction and machine learning models (kNN/NBC).
- `nltk`: NLP tools such as tokenization, text preprocessing.
- `joblib`: To persist trained models for reusability

Install the dependencies with

```shell
pip install -r requirements.txt
```

## How to Run

### Training the models

Running the main script will automatically train the two models and save them

```shell
python3 main.py
```

### Making a prediction with the models

Running the script below with your model/input sentences to predict whether the sentence is pun-intended.
When using kNN as model, you can also include `n_neighbors` flag to display a scattered plot of the input and its neighbors


```shell
python3 predict.py --model <model name> --input <text> --n_neighbors <optional number>
```
