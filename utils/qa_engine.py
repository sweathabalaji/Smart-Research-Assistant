from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import nltk
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
import os
import ssl
import logging
from typing import Dict, List, Any, Union, Optional, cast

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import PyPDF2  # for PDF files
except ImportError:
    PyPDF2 = None
    logger.warning("PyPDF2 not installed. PDF functionality will be limited.")

try:
    import docx    # for DOCX files
except ImportError:
    docx = None
    logger.warning("python-docx not installed. DOCX functionality will be limited.")

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

# Download nltk data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)

# Define types for QA results
QAResult = Union[Dict[str, Any], str]

# Load tokenizer and model from local folder
try:
    model_path = "./models/distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    # Create the QA pipeline with kwargs to avoid type errors
    qa_pipeline = pipeline(
        task="question-answering", 
        model=model, 
        tokenizer=tokenizer
    )
    logger.info("QA model loaded successfully")
except Exception as e:
    logger.error(f"Error loading QA model: {str(e)}")
    # Define a simple fallback QA function
    def qa_fallback(question=None, context=None, **kwargs) -> Dict[str, str]:
        return {"answer": "Sorry, the QA model could not be loaded. Please try again later."}
    qa_pipeline = qa_fallback

def chunk_text(text, max_chunk_size=512, overlap=100):
    """
    Split text into overlapping chunks of specified size.
    """
    try:
        sentences = sent_tokenize(text)
    except Exception:
        # Simple fallback sentence tokenization
        sentences = text.replace('\n', ' ').split('. ')
        sentences = [s + '.' for s in sentences[:-1]] + [sentences[-1]]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        try:
            sentence_tokens = tokenizer.tokenize(sentence)
            sentence_size = len(sentence_tokens)
        except:
            # If tokenization fails, estimate size based on characters
            sentence_size = len(sentence) // 4  # Rough estimate
        
        if current_size + sentence_size <= max_chunk_size:
            current_chunk.append(sentence)
            current_size += sentence_size
        else:
            # Add the current chunk to our list of chunks
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Start a new chunk with overlap
            overlap_tokens = current_size - overlap
            while overlap_tokens > 0 and current_chunk:
                sentence_to_remove = current_chunk.pop(0)
                try:
                    overlap_tokens -= len(tokenizer.tokenize(sentence_to_remove))
                except:
                    overlap_tokens -= len(sentence_to_remove) // 4
            
            # Add the current sentence to the new chunk
            current_chunk.append(sentence)
            try:
                current_size = sum(len(tokenizer.tokenize(s)) for s in current_chunk)
            except:
                current_size = sum(len(s) // 4 for s in current_chunk)
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_answer_from_result(result: QAResult) -> str:
    """Extract answer text from different result formats."""
    if isinstance(result, dict) and "answer" in result:
        return result["answer"]
    return str(result)

def answer_question(question, context):
    """
    Improved answer_question that handles long contexts by chunking
    and finding the best answer across chunks.
    """
    if not question or not context:
        return "Please provide both a question and document text."
    
    # For short contexts, just use the pipeline directly
    try:
        context_tokens = tokenizer.tokenize(context)
        if len(context_tokens) < 384:
            try:
                # Call QA pipeline with explicit parameters
                result = qa_pipeline(
                    question=question, 
                    context=context
                )
                return get_answer_from_result(result)
            except Exception as e:
                logger.error(f"Error in QA pipeline: {str(e)}")
                return "Could not find an answer to this question."
    except:
        # If tokenization fails, assume it's too long
        pass
    
    # For longer contexts, chunk and find best answer
    try:
        chunks = chunk_text(context)
        best_answer = ""
        best_score = -1.0  # Use float for score
        
        for chunk in chunks:
            try:
                # Call QA pipeline with explicit parameters
                result = qa_pipeline(
                    question=question, 
                    context=chunk
                )
                
                # Handle different return types
                if isinstance(result, dict) and "score" in result:
                    current_score = float(result["score"])  # Convert to float
                else:
                    current_score = 0.0
                
                if current_score > best_score:
                    best_score = current_score
                    best_answer = get_answer_from_result(result)
            except Exception as e:
                logger.error(f"Error processing chunk in QA: {str(e)}")
                continue
        
        # If no good answer found
        if not best_answer or best_score < 0.1:
            return "I couldn't find a reliable answer to this question in the document."
        
        return best_answer
    except Exception as e:
        logger.error(f"Error in chunked QA: {str(e)}")
        return "An error occurred while processing your question."

def generate_questions(context):
    """
    Generate more meaningful questions based on content analysis.
    """
    if not context:
        return ["Please provide a document to generate questions."]
    
    # Chunk the context for processing
    try:
        chunks = chunk_text(context, max_chunk_size=768)
    except:
        # Simple chunking fallback
        words = context.split()
        chunks = [" ".join(words[i:i+200]) for i in range(0, len(words), 150)]
    
    # Base questions that are generally applicable
    base_questions = [
        "What is the main topic discussed in this document?",
        "What are the key findings mentioned in the document?",
        "What conclusions are drawn in this document?"
    ]
    
    # Content-specific questions based on keywords
    content_questions = []
    
    # Keywords and associated questions
    keyword_questions = {
        "research": ["What research methodology was used?", "What were the research findings?"],
        "study": ["What was the purpose of the study?", "What population was studied?"],
        "data": ["What data was collected?", "How was the data analyzed?"],
        "results": ["What were the main results?", "How significant were the results?"],
        "conclusion": ["What conclusions were drawn?", "What limitations were mentioned?"],
        "method": ["What methods were employed?", "Why were these methods chosen?"],
        "analysis": ["What type of analysis was performed?", "What were the outcomes of the analysis?"],
        "experiment": ["What was the experimental design?", "What variables were controlled?"],
        "hypothesis": ["What was the hypothesis?", "Was the hypothesis supported?"],
        "recommendation": ["What recommendations were made?", "Who should implement these recommendations?"],
        "patient": ["What patient population was described?", "What treatments were administered to patients?"],
        "diagnosis": ["What diagnostic criteria were used?", "How was the diagnosis confirmed?"],
        "treatment": ["What treatment options were discussed?", "What was the efficacy of the treatment?"],
        "symptoms": ["What symptoms were reported?", "How were the symptoms managed?"],
        "technology": ["What technologies were discussed?", "How were these technologies applied?"],
        "algorithm": ["What algorithms were mentioned?", "How effective were the algorithms?"],
        "implementation": ["How was the implementation carried out?", "What challenges were faced during implementation?"]
    }
    
    # Check for keywords in each chunk
    for chunk in chunks:
        chunk_lower = chunk.lower()
        for keyword, questions in keyword_questions.items():
            if keyword in chunk_lower:
                # Add relevant questions for this keyword
                content_questions.extend(questions)
    
    # Combine questions and remove duplicates
    all_questions = base_questions + list(set(content_questions))
    
    # Limit to a reasonable number (max 10 questions)
    if len(all_questions) > 10:
        all_questions = all_questions[:10]
    
    return all_questions

# File reading utility
def extract_text_from_file(file_path):
    """
    Extract text from various file formats with error handling for missing dependencies.
    """
    if file_path.endswith(".pdf"):
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")
        
        text = ""
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")

    elif file_path.endswith(".docx"):
        if docx is None:
            raise ImportError("python-docx is required for DOCX support. Install with: pip install python-docx")
        
        try:
            doc = docx.Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Error reading DOCX file: {str(e)}")

    elif file_path.endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding if utf-8 fails
            with open(file_path, "r", encoding="latin-1") as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")

    else:
        raise ValueError("Unsupported file format. Please use PDF, DOCX, or TXT files.")

# Main function
def process_file_and_qa(file_path):
    context = extract_text_from_file(file_path)
    questions = generate_questions(context)
    
    print("\n📄 Extracted Questions and Answers:\n")
    for q in questions:
        try:
            ans = answer_question(q, context)
            print(f"❓ {q}\n➡️ {ans}\n")
        except:
            print(f"❓ {q}\n⚠️ Could not find a good answer.\n")
