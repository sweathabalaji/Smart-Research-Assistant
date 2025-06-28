print("Checking scikit-learn installation...")

try:
    import sklearn
    print(f"scikit-learn is installed. Version: {sklearn.__version__}")
    
    # Check specific modules
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("TfidfVectorizer is available")
    except ImportError:
        print("ERROR: TfidfVectorizer could not be imported")
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        print("cosine_similarity is available")
    except ImportError:
        print("ERROR: cosine_similarity could not be imported")
        
    # Try a simple operation
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer()
        corpus = ['This is the first document.', 'This document is the second document.']
        X = vectorizer.fit_transform(corpus)
        print("scikit-learn functionality test: SUCCESS")
    except Exception as e:
        print(f"scikit-learn functionality test: FAILED - {str(e)}")
        
except ImportError:
    print("ERROR: scikit-learn is not installed or not in the Python path")
    
print("\nChecking Python path:")
import sys
for path in sys.path:
    print(f"  - {path}") 