import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import argparse
import joblib
import os
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define a preprocessing function (consistent with training pipeline)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(
        word) for word in words if word not in stop_words and word not in string.punctuation]
    return ' '.join(words)


def load_model(model_path):
    """
    Loads a trained model and the TF-IDF vectorizer.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict(model, vectorizer, text):
    """
    Preprocesses input text, transforms it using the TF-IDF vectorizer, 
    and uses the model to make predictions.
    """
    # Preprocess the input text
    clean_text = preprocess_text(text)

    # Transform the input text
    X = vectorizer.transform([clean_text])

    # Make prediction
    prediction = model.predict(X)
    probability = model.predict_proba(X) if hasattr(
        model, 'predict_proba') else None

    return prediction[0], probability


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Predict if a text contains a pun using a trained model.")
    parser.add_argument('--model', type=str, required=True,
                        help="Path to the trained model (e.g., model.pkl)")
    parser.add_argument('--input', type=str, required=True,
                        help="Input text to classify")

    args = parser.parse_args()

    # Load the trained model
    print("Loading the model...")
    pipeline = load_model(args.model)

    # Extract model and vectorizer from the pipeline
    vectorizer = pipeline['tfidf']
    model = pipeline['classifier']

    # Predict the label
    print("Making a prediction...")
    prediction, probability = predict(model, vectorizer, args.input)

    # Display the result
    if probability is not None:
        print(f"Prediction: {'Pun' if prediction == 1 else 'No Pun'} (Probability: {
              max(probability[0]) * 100:.2f}%)")
    else:
        print(f"Prediction: {'Pun' if prediction == 1 else 'No Pun'}")


if __name__ == "__main__":
    main()
