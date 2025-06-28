import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ssl
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix SSL certificate issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Try to import scikit-learn, but provide fallback if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn is available for advanced text comparison")
except ImportError:
    logger.warning("scikit-learn not available. Using simpler text comparison.")
    SKLEARN_AVAILABLE = False

# Download required NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Get stopwords
try:
    stop_words = set(stopwords.words('english'))
except:
    # Fallback if stopwords not available
    stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when',
                      'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those', 'then',
                      'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of',
                      'while', 'during', 'to', 'from', 'in', 'on', 'at', 'by', 'with'])

def preprocess_text(text):
    """
    Preprocess text for comparison:
    - Convert to lowercase
    - Remove punctuation
    - Remove stopwords
    - Tokenize
    """
    if not text:
        return ""
        
    try:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        filtered_tokens = [w for w in tokens if w not in stop_words]
        return " ".join(filtered_tokens)
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        # Fallback to simple preprocessing
        if not isinstance(text, str):
            text = str(text)
        return " ".join([w for w in text.lower().split() if w not in stop_words])

def calculate_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts using TF-IDF and cosine similarity.
    Falls back to simpler word overlap method if scikit-learn is not available.
    """
    # Ensure we have valid inputs
    if not text1 or not text2:
        return 0.0
        
    # Convert to string if needed
    if not isinstance(text1, str):
        text1 = str(text1)
    if not isinstance(text2, str):
        text2 = str(text2)
        
    # Preprocess texts
    try:
        processed_text1 = preprocess_text(text1)
        processed_text2 = preprocess_text(text2)
        
        # If either processed text is empty, return low similarity
        if not processed_text1 or not processed_text2:
            return 0.1  # Return small non-zero value
    except Exception as e:
        logger.error(f"Error in text preprocessing: {str(e)}")
        return 0.0
    
    if SKLEARN_AVAILABLE:
        # Use scikit-learn for better similarity calculation
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)  # Ensure we return a float
        except Exception as e:
            logger.error(f"Error in TF-IDF calculation: {str(e)}")
            # Fallback if vectorization fails (e.g., empty strings)
            if processed_text1 == processed_text2:
                return 1.0
            # Use the simpler method as fallback
            logger.info("Falling back to simple similarity calculation")
    
    # Simple fallback using word overlap
    try:
        words1 = set(processed_text1.split())
        words2 = set(processed_text2.split())
        
        if not words1 or not words2:
            return 0.1  # Return small non-zero value
            
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return float(intersection / union)  # Ensure we return a float
    except Exception as e:
        logger.error(f"Error in simple similarity calculation: {str(e)}")
        # Last resort fallback
        return 0.0

def evaluate_answer(user_answer, correct_answer):
    """
    Compares user's answer with the correct answer and provides detailed feedback.
    Returns:
        Tuple: (Feedback message, Score between 0 and 1)
    """
    try:
        # Handle empty answers
        if not user_answer or not correct_answer:
            return "❌ No answer provided", 0.0
        
        # Calculate similarity score
        similarity = calculate_similarity(user_answer, correct_answer)
        
        # Ensure we have a valid float score between 0 and 1
        similarity = max(0.0, min(1.0, float(similarity)))
        
        # Determine score and feedback based on similarity
        if similarity >= 0.8:
            return "✅ Excellent answer! Your response matches the key points perfectly.", similarity
        elif similarity >= 0.6:
            return "✓ Good answer! You've captured most of the important points.", similarity
        elif similarity >= 0.4:
            return "⚠️ Partially correct. Your answer contains some relevant information, but misses key points.", similarity
        elif similarity >= 0.2:
            return "⚠️ Your answer is on the right track but needs more specific details.", similarity
        else:
            return "❌ Your answer doesn't match the expected response. Try again with more specific information.", similarity
    except Exception as e:
        logger.error(f"Error in evaluate_answer: {str(e)}")
        # Return a default score and feedback in case of error
        return "⚠️ Error evaluating answer.", 0.1
