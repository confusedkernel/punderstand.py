import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import numpy as np


def plot_confusion_matrix(y_test, y_pred, model_name):
    """
    Plots a confusion matrix for the given predictions.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                'No Pun', 'Pun'], yticklabels=['No Pun', 'Pun'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()


def compare_metrics(report_knn, report_nbc):
    """
    Creates a bar chart to compare precision, recall, and F1-score between models.
    """
    metrics = ['precision', 'recall', 'f1-score']
    knn_metrics = [report_knn['weighted avg'][metric] for metric in metrics]
    nbc_metrics = [report_nbc['weighted avg'][metric] for metric in metrics]

    # Plot the metrics
    x = range(len(metrics))
    plt.figure(figsize=(8, 5))
    plt.bar(x, knn_metrics, width=0.4, label='kNN', align='center')
    plt.bar([p + 0.4 for p in x], nbc_metrics, width=0.4,
            label='Naive Bayes', align='center')
    plt.xticks([p + 0.2 for p in x], metrics)
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.show()


def plot_knn_neighbors(knn_model, sample_vector, X_test, y_test, n_neighbors=5):
    """
    Visualizes the kNN neighbors for a specific sample.
    """
    # Find k nearest neighbors
    distances, indices = knn_model.kneighbors(
        sample_vector, n_neighbors=n_neighbors)

    # Get neighbor data
    neighbor_vectors = X_test[indices[0]]
    neighbor_labels = [y_test[i] for i in indices[0]]

    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(
        np.vstack([sample_vector.toarray()] + [v.toarray() for v in neighbor_vectors]))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[1:, 0], X_reduced[1:, 1],
                c=neighbor_labels, cmap='coolwarm', label='Neighbors')
    plt.scatter(X_reduced[0, 0], X_reduced[0, 1], c='green',
                label='Sample', edgecolor='black', s=100)
    plt.title('kNN Neighbors Visualization')
    plt.legend()
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

