from datasets import load_dataset
from preprocessing import preprocess_text
from models import train_knn, train_naive_bayes, save_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load dataset and inspect structure
dataset = load_dataset('frostymelonade/SemEval2017-task7-pun-detection')
print(dataset)  # Debugging step

# Use the 'test' split for training/testing
texts = dataset['test']['text']
labels = dataset['test']['label']

# Preprocess the texts
texts = [preprocess_text(text) for text in texts]

# Split into training and testing data
X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

# Train and evaluate kNN
knn_model = train_knn(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print("kNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

# Train and evaluate Naive Bayes
nbc_model = train_naive_bayes(X_train, y_train)
y_pred_nbc = nbc_model.predict(X_test)
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nbc))

# Save models
save_model(knn_model, 'knn_model.pkl')
save_model(nbc_model, 'nbc_model.pkl')

print("Models saved successfully.")
