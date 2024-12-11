import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def plot_nbc_feature_contributions(vectorizer, model, input_vector, class_labels, top_n=10):
    """
    Visualizes the most influential features for the predicted class, limited to
    features present in the input text.

    Args:
        vectorizer: Fitted TfidfVectorizer used for transforming text.
        model: Trained Naive Bayes model.
        input_vector: Vectorized input text (sparse matrix).
        class_labels: List of class labels (e.g., ["No Pun", "Pun"]).
        top_n: Number of top features to display.

    Returns:
        None
    """
    feature_names = vectorizer.get_feature_names_out()
    # Log probabilities of features for each class
    log_probs = model.feature_log_prob_
    # Convert sparse matrix to dense array
    input_vector = input_vector.toarray()[0]

    # Get the predicted class
    predicted_class = np.argmax(
        model.predict_proba(input_vector.reshape(1, -1)))

    # Contribution scores for the predicted class
    contributions = log_probs[predicted_class] * input_vector

    # Filter features that are present in the input text
    non_zero_indices = np.nonzero(input_vector)[0]
    filtered_contributions = contributions[non_zero_indices]
    filtered_features = feature_names[non_zero_indices]

    # Get the top N features
    top_indices = np.argsort(np.abs(filtered_contributions))[-top_n:][::-1]
    top_features = [filtered_features[i] for i in top_indices]
    top_scores = [filtered_contributions[i] for i in top_indices]

    # Debugging
    print(f"Top contributions for class '{class_labels[predicted_class]}':")
    for feature, score in zip(top_features, top_scores):
        print(f"{feature}: {score}")

    # Plot the feature contributions
    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_scores,
             color='green' if predicted_class == 1 else 'blue')
    plt.xlabel('Contribution Score')
    plt.ylabel('Feature')
    plt.title(f"Top {top_n} Features Contributing to Class '{class_labels[predicted_class]}'")
    plt.tight_layout()
    plt.show()


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
