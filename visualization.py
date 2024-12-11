import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def plot_knn_neighbors(
        knn_model, sample_vector, X_train, y_train, n_neighbors=5):
    """
    Visualizes the kNN neighbors for a specific sample.

    Args:
        knn_model: Trained kNN model.
        sample_vector: Vector for the specific sample to analyze.
        X_train: Training set feature matrix (sparse matrix).
        y_train: Training set labels (list or array).
        n_neighbors: Number of neighbors to visualize.

    Returns:
        None
    """
    # Find k nearest neighbors
    distances, indices = knn_model.kneighbors(
        sample_vector, n_neighbors=n_neighbors)

    # Handle sparse matrix indexing correctly
    # Convert each neighbor vector to dense
    neighbor_vectors = [X_train[i].toarray().flatten() for i in indices[0]]
    # Convert y_train to NumPy array for indexing
    neighbor_labels = np.array(y_train)[indices[0]]

    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    all_vectors = np.vstack(
        [sample_vector.toarray().flatten()] + neighbor_vectors)
    reduced_vectors = pca.fit_transform(all_vectors)

    # Plot neighbors and the target sample
    plt.figure(figsize=(8, 6))
    # Plot target sample
    plt.scatter(reduced_vectors[0, 0], reduced_vectors[0, 1],
                color='red', label='Target Sample', s=150, edgecolor='black')

    # Plot neighbors
    for i, (x, y) in enumerate(reduced_vectors[1:]):
        label = neighbor_labels[i]
        plt.scatter(x, y, c='blue' if label == 0 else 'green',
                    label=f'Neighbor {
                        i + 1} ({"No Pun" if label == 0 else "Pun"})',
                    s=100, edgecolor='black')

    # Improve plot aesthetics
    plt.title(f'{n_neighbors}-Nearest Neighbors Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_metrics(report_knn, report_nbc):
    """
    Creates a bar chart to compare precision, recall, and F1-score
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


def visualize_prediction(input_text, prediction, probability=None):
    """
    Visualizes a prediction with input text and probability (if available).
    """
    plt.figure(figsize=(10, 6))
    plt.barh(['Prediction'], [1 if prediction == 1 else 0],
             color='green' if prediction == 1 else 'blue')
    plt.text(0.5, 0, 'Pun' if prediction == 1 else 'No Pun',
             fontsize=12, va='center', ha='center', color='white')

    if probability is not None:
        plt.barh(['Confidence'], [max(probability) * 100], color='orange')
        plt.text(max(probability) * 50, -0.6, f"{max(probability) * 100:.2f}%",
                 fontsize=12, va='center', ha='center', color='white')

    plt.title("Prediction Visualization")
    plt.xlim(0, 100)
    plt.xlabel('Confidence Percentage')
    plt.yticks([])
    plt.tight_layout()
    plt.show()
