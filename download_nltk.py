import nltk
import ssl
import os
import shutil

# Fix SSL certificate issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data path to ensure it's found
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# Download required NLTK data
print("Downloading NLTK data...")

# Basic resources
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

# Create necessary files for punkt_tab
try:
    # Create directory structure for punkt_tab if it doesn't exist
    punkt_tab_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab', 'english')
    os.makedirs(punkt_tab_dir, exist_ok=True)
    
    # Copy punkt files to punkt_tab directory
    punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
    if os.path.exists(punkt_dir):
        for file in os.listdir(punkt_dir):
            if file.endswith('.pickle'):
                src = os.path.join(punkt_dir, file)
                dst = os.path.join(punkt_tab_dir, file)
                shutil.copy2(src, dst)
                print(f"Copied {file} to punkt_tab directory")
    
    # Create empty collocations.tab file (required by NLTK)
    collocations_file = os.path.join(punkt_tab_dir, 'collocations.tab')
    with open(collocations_file, 'w', encoding='utf-8') as f:
        f.write("# This is an empty collocations file for NLTK punkt_tab\n")
    print(f"Created empty collocations.tab file at {collocations_file}")
    
    # Create empty abbreviations.tab file (might be needed)
    abbreviations_file = os.path.join(punkt_tab_dir, 'abbreviations.tab')
    with open(abbreviations_file, 'w', encoding='utf-8') as f:
        f.write("# This is an empty abbreviations file for NLTK punkt_tab\n")
    print(f"Created empty abbreviations.tab file at {abbreviations_file}")
    
    print("punkt_tab setup complete")
except Exception as e:
    print(f"Error setting up punkt_tab: {str(e)}")

print("NLTK data download complete!")

# Verify the data is available
try:
    from nltk.tokenize import sent_tokenize
    test_text = "This is a test sentence. This is another test sentence."
    sentences = sent_tokenize(test_text)
    print(f"Sentence tokenization test: {sentences}")
    print("✅ NLTK setup successful!")
except Exception as e:
    print(f"❌ NLTK setup error: {str(e)}")
    print("Please run this script again or manually download the required resources.") 