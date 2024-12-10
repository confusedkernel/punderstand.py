import string
from nltk.stem import WordNetLemmatizer
from visualization import visualize_prediction, plot_knn_neighbors
from nltk.corpus import stopwords
from datasets import load_dataset
import argparse
import joblib
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


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
    print("Loading dataset...")
    dataset = load_dataset('frostymelonade/SemEval2017-task7-pun-detection')

    print("Preprocessing dataset...")
    texts = [preprocess_text(text) for text in dataset['test']['text']]
    labels = dataset['test']['label']
    return texts, labels


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Predict if a text contains a pun using a trained model.")
    parser.add_argument('--model', type=str, required=True,
                        help="Path to the trained model (e.g., model.pkl)")
    parser.add_argument('--input', type=str, required=True,
                        help="Input text to classify")
    parser.add_argument('--n_neighbors', type=int, default=10,
                        help="Number of neighbors to display in the scatter plot (only for kNN).")

    args = parser.parse_args()

    # Load the trained model
    print("Loading the model...")
    pipeline = load_model(args.model)

    # Load and preprocess dataset dynamically
    texts, labels = load_and_preprocess_data()

    # Split data into training and testing sets
    print("Splitting dataset into training and testing sets...")
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.4, random_state=42
    )

    # Vectorize the dataset
    print("Vectorizing the dataset...")
    vectorizer = pipeline.named_steps['tfidf']
    X_train = vectorizer.transform(X_train_texts)  # Train set
    X_test = vectorizer.transform(X_test_texts)    # Test set

    # Predict the label
    print("Making a prediction...")
    processed_text = preprocess_text(args.input)
    prediction, probability = predict(pipeline, processed_text)

    # Display the result
    if probability is not None:
        print(f"Prediction: {'Pun' if prediction == 1 else 'No Pun'} (Probability: {
              max(probability) * 100:.2f}%)")
    else:
        print(f"Prediction: {'Pun' if prediction == 1 else 'No Pun'}")

    # # Visualize the prediction
    # visualize_prediction(args.input, prediction, probability)

    # Visualize kNN neighbors if the model supports it
    if hasattr(pipeline.named_steps['classifier'], 'kneighbors'):
        print("Visualizing kNN neighbors...")
        knn_model = pipeline.named_steps['classifier']
        sample_vector = vectorizer.transform(
            [processed_text])  # Vectorize the input text

        # Use X_train and y_train for neighbor visualization
        plot_knn_neighbors(knn_model, sample_vector, X_train,
                           y_train, n_neighbors=args.n_neighbors)


if __name__ == "__main__":
    main()
