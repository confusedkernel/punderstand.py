import string
from nltk.stem import WordNetLemmatizer
from visualization import visualize_prediction, plot_knn_neighbors
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
    """
    Preprocesses input text by converting to lowercase, removing punctuation and stopwords,
    and tokenizing.
    """
    # Tokenize and convert to lowercase
    tokens = nltk.word_tokenize(text.lower())
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Rejoin tokens into a single string
    return " ".join(tokens)


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict(model, text):
    # Preprocess and predict using the pipeline
    prediction = model.predict([text])[0]
    probability = model.predict_proba([text])[0] if hasattr(
        model, 'predict_proba') else None

    return prediction, probability


def load_and_preprocess_data():
    """
    Load and preprocess the dataset dynamically.
    """

    print("Loading preprocessed training data...")
    # Load the saved training data
    X_train, y_train = joblib.load('train_data.pkl')

    return X_train, y_train


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Predict if a text contains a pun using a trained model.")
    parser.add_argument('--model', type=str, required=True,
                        help="Path to the trained model (e.g., model.pkl)")
    parser.add_argument('--input', type=str, required=True,
                        help="Input text to classify")
    parser.add_argument('--n_neighbors', type=int, default=5,
                        help="Number of neighbors to display in the scatter plot (only for kNN).")

    args = parser.parse_args()

    # Load the trained model
    print("Loading the model...")
    pipeline = load_model(args.model)

    # Load preprocessed test data
    X_test, y_test = load_and_preprocess_data()

    # Extract the vectorizer and model from the pipeline
    vectorizer = pipeline.named_steps['tfidf']
    knn_model = pipeline.named_steps['classifier']

    # Preprocess the input text
    processed_text = preprocess_text(args.input)

    # Predict the label
    print("Making a prediction...")
    prediction, probability = predict(pipeline, processed_text)

    # Display the result
    if probability is not None:
        print(f"Prediction: {'Pun' if prediction == 1 else 'No Pun'} (Probability: {
              max(probability) * 100:.2f}%)")
    else:
        print(f"Prediction: {'Pun' if prediction == 1 else 'No Pun'}")

    # # Visualize the prediction (optional)
    # visualize_prediction(args.input, prediction, probability)

    # Visualize kNN neighbors if the model supports it
    if hasattr(knn_model, 'kneighbors'):
        print("Visualizing kNN neighbors...")
        sample_vector = vectorizer.transform(
            [processed_text])  # Vectorize the input text
        plot_knn_neighbors(knn_model, sample_vector, X_test,
                           y_test, n_neighbors=args.n_neighbors)


if __name__ == "__main__":
    main()
