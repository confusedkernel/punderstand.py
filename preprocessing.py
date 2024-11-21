import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):
    """
    Preprocesses input text by converting to lowercase, 
    removing punctuation and stopwords, and tokenizing.
    """
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # Remove stopwords
    tokens = [
        word for word in tokens if word not in stopwords.words('english')]
    # Rejoin tokens into a single string
    return " ".join(tokens)
