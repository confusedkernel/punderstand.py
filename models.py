from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib


def train_knn(X, y, n_neighbors=9):
    """
    Trains a k-Nearest Neighbors (kNN) classifier.

    Args:
        X: Training feature matrix.
        y: Training labels.
        n_neighbors: Number of neighbors to consider in kNN.

    Returns:
        Trained kNN model.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    return knn


def train_naive_bayes(X, y):
    """
    Trains a Naive Bayes classifier.

    Args:
        X: Training feature matrix.
        y: Training labels.

    Returns:
        Trained Naive Bayes model.
    """
    nbc = MultinomialNB()
    nbc.fit(X, y)
    return nbc


def save_model(model, vectorizer, filepath):
    """
    Saves the trained model and vectorizer as a pipeline.

    Args:
        model: Trained model to save.
        vectorizer: Fitted TfidfVectorizer to save.
        filepath: Path to save the pipeline.
    """
    pipeline = Pipeline([('tfidf', vectorizer), ('classifier', model)])
    joblib.dump(pipeline, filepath)


def load_model(filepath):
    """
    Loads a model from a file.

    Args:
        filepath: Path to the saved model file.

    Returns:
        Loaded model.
    """
    return joblib.load(filepath)
