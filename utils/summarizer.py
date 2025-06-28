from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import re
import os
import ssl
import logging
from typing import Dict, List, Any, Union, Optional

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

# Set NLTK data path to include local directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if os.path.exists(nltk_data_dir):
    nltk.data.path.insert(0, nltk_data_dir)

# Simple sentence tokenizer as fallback
def simple_sent_tokenize(text):
    """
    A simple sentence tokenizer that splits on common sentence terminators.
    Used as fallback if NLTK's sent_tokenize fails.
    """
    # Replace common abbreviations to avoid splitting them
    text = re.sub(r'Mr\.', 'Mr', text)
    text = re.sub(r'Mrs\.', 'Mrs', text)
    text = re.sub(r'Dr\.', 'Dr', text)
    text = re.sub(r'Ph\.D\.', 'PhD', text)
    text = re.sub(r'i\.e\.', 'ie', text)
    text = re.sub(r'e\.g\.', 'eg', text)
    text = re.sub(r'etc\.', 'etc', text)
    
    # Split on sentence terminators followed by space and uppercase letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Handle cases where there's no space after the period
    result = []
    for sentence in sentences:
        # Further split on sentence terminators directly followed by uppercase letter
        subsents = re.split(r'(?<=[.!?])(?=[A-Z])', sentence)
        result.extend(subsents)
    
    # Remove empty strings and strip whitespace
    return [s.strip() for s in result if s.strip()]

# Try to get NLTK's sent_tokenize, or use our fallback
try:
    from nltk.tokenize import sent_tokenize
    # Skip testing here - we'll handle errors during actual use
except (ImportError, LookupError):
    logger.warning("NLTK's sent_tokenize not available. Using simple fallback tokenizer.")
    sent_tokenize = simple_sent_tokenize

# Define a type for summarization results
SummaryResult = Union[List[Dict[str, str]], Dict[str, str], str]

# Load models
try:
    model_path = "./models/facebook-bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Create summarizer pipeline with explicit model and tokenizer instances
    # Use kwargs to avoid type errors
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    logger.info("Summarization model loaded successfully")
except Exception as e:
    logger.error(f"Error loading summarization model: {str(e)}")
    # Define a simple fallback summarizer that just returns the first few sentences
    def fallback_summarize(text, max_length=200, **kwargs) -> List[Dict[str, str]]:
        try:
            sentences = sent_tokenize(text)
            # Take first few sentences as a simple summary
            summary = " ".join(sentences[:3])
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return [{"summary_text": summary}]
        except:
            return [{"summary_text": text[:200] + "..."}]
    
    summarizer = fallback_summarize

def extract_summary_text(result: SummaryResult) -> str:
    """Helper function to extract summary text from different result formats."""
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict) and "summary_text" in result[0]:
            return result[0]["summary_text"]
    elif isinstance(result, dict) and "summary_text" in result:
        return result["summary_text"]
    # Fallback for unexpected formats
    return str(result)

def chunk_text_for_summary(text, max_chunk_size=1000):
    """
    Split text into chunks for summarization.
    """
    # Tokenize into sentences
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.error(f"Error in sentence tokenization: {str(e)}")
        logger.info("Falling back to simple sentence tokenizer")
        sentences = simple_sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        # Get the token count for this sentence
        try:
            sentence_tokens = tokenizer.tokenize(sentence)
            sentence_size = len(sentence_tokens)
        except:
            # If tokenization fails, estimate size based on characters
            sentence_size = len(sentence) // 4  # Rough estimate
        
        # If adding this sentence doesn't exceed the max size, add it
        if current_size + sentence_size <= max_chunk_size:
            current_chunk.append(sentence)
            current_size += sentence_size
        else:
            # If the current chunk is not empty, add it to chunks
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Start a new chunk with this sentence
            current_chunk = [sentence]
            current_size = sentence_size
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def generate_summary(text):
    """
    Generate a summary of the text, handling long documents by chunking.
    """
    if not text or len(text.strip()) == 0:
        return "No text provided for summarization."
    
    # Check if the text is short enough to summarize directly
    try:
        token_count = len(tokenizer.tokenize(text))
        if token_count <= 1024:
            try:
                result = summarizer(
                    text,
                    max_length=350,
                    min_length=150,
                    length_penalty=1.0,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    do_sample=False
                )
                return extract_summary_text(result)
            except Exception as e:
                logger.error(f"Error in summarization: {str(e)}")
                return "Error generating summary. Please try again with a different document."
    except:
        # If tokenization fails, assume it's too long
        pass
    
    # For longer text, chunk and summarize each chunk
    try:
        chunks = chunk_text_for_summary(text)
        chunk_summaries = []
        
        for chunk in chunks:
            try:
                result = summarizer(
                    chunk,
                    max_length=200,  # Shorter summaries for chunks
                    min_length=50,
                    length_penalty=1.0,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    do_sample=False
                )
                
                chunk_summaries.append(extract_summary_text(result))
            except Exception as e:
                logger.error(f"Error summarizing chunk: {str(e)}")
                # Skip failed chunks
                continue
        
        # Combine chunk summaries
        if not chunk_summaries:
            return "Could not generate a summary for this document."
            
        combined_summary = " ".join(chunk_summaries)
        
        # If the combined summary is still too long, summarize it again
        try:
            if len(tokenizer.tokenize(combined_summary)) > 1024:
                result = summarizer(
                    combined_summary,
                    max_length=350,
                    min_length=150,
                    length_penalty=1.0,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    do_sample=False
                )
                
                return extract_summary_text(result)
        except:
            # If re-summarization fails, return the combined summary
            pass
        
        return combined_summary
    except Exception as e:
        logger.error(f"Error in chunked summarization: {str(e)}")
        # Fallback to returning first few sentences
        try:
            sentences = sent_tokenize(text)
            return " ".join(sentences[:5])
        except:
            return text[:300] + "..."
