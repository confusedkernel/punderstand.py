import numpy as np
import nltk
import logging
from datasets import load_dataset
from preprocessing import preprocess_text
from models import train_knn, train_naive_bayes, save_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.model_selection import train_test_split, cross_val_score

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Define scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}


# Function to load and preprocess dataset
def load_and_preprocess_data():
    logging.info("Loading dataset...")
    dataset = load_dataset('frostymelonade/SemEval2017-task7-pun-detection')

    logging.info("Preprocessing dataset...")
    texts = [preprocess_text(text) for text in dataset['test']['text']]
    labels = dataset['test']['label']
    return texts, labels


# Function to vectorize text
def vectorize_data(texts):
    logging.info("Vectorizing text data with TF-IDF...")
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


# Function to evaluate a model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


# Function to perform 10-fold cross-validation
def perform_cross_validation(model, X_train, y_train):
    logging.info("Performing 10-fold cross-validation...")
    scores = cross_val_score(model, X_train, y_train,
                             cv=10, scoring='accuracy')
    return np.mean(scores)


def main():
    # Load and preprocess data
    texts, labels = load_and_preprocess_data()

    # Split data into training and testing sets
    logging.info("Splitting dataset into train and test sets...")
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.4, random_state=42)

    # Vectorize text
    X_train, vectorizer = vectorize_data(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    # Train and evaluate kNN
    logging.info("Training kNN model...")
    knn_model = train_knn(X_train, y_train)
    accuracy_knn, report_knn = evaluate_model(knn_model, X_test, y_test)
    logging.info(f"kNN Model Accuracy: {accuracy_knn:.4f}")
    logging.info(f"kNN Classification Report:\n{report_knn}")

    # Train and evaluate Naive Bayes
    logging.info("Training Naive Bayes model...")
    nbc_model = train_naive_bayes(X_train, y_train)
    accuracy_nbc, report_nbc = evaluate_model(nbc_model, X_test, y_test)
    logging.info(f"Naive Bayes Model Accuracy: {accuracy_nbc:.4f}")
    logging.info(f"Naive Bayes Classification Report:\n{report_nbc}")

    # Perform cross-validation for kNN
    cv_accuracy_knn = perform_cross_validation(knn_model, X_train, y_train)
    logging.info(
        f"kNN 10-Fold Cross-Validation Accuracy: {cv_accuracy_knn:.4f}")

    # Perform cross-validation for Naive Bayes
    cv_accuracy_nbc = perform_cross_validation(nbc_model, X_train, y_train)
    logging.info(
        f"NBC 10-Fold Cross-Validation Accuracy: {cv_accuracy_nbc:.4f}")

    # Save models
    save_model(knn_model, vectorizer, 'knn_model.pkl')
    save_model(nbc_model, vectorizer, 'nbc_model.pkl')
    logging.info("Models saved successfully.")


if __name__ == "__main__":
    main()
