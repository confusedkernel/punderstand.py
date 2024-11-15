from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib


def train_knn(X, y, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    return knn


def train_naive_bayes(X, y):
    nbc = MultinomialNB()
    nbc.fit(X, y)
    return nbc


def save_model(model, filepath):
    joblib.dump(model, filepath)


def load_model(filepath):
    return joblib.load(filepath)
