import fitz  # PyMuPDF
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_pdf_text(pdf_file):
    """
    Extract text from a PDF file.
    Supports both file paths and file-like objects.
    """
    try:
        # Check if pdf_file is a string (file path)
        if isinstance(pdf_file, str):
            logger.info(f"Opening PDF from path: {pdf_file}")
            doc = fitz.open(pdf_file)
        else:
            # Assume it's a file-like object
            logger.info("Opening PDF from file object")
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        
        text = ""
        for page_num in range(len(doc)):
            # Use getattr to avoid the linter error
            page = doc.load_page(page_num)
            # Use getattr to avoid the linter error with get_text
            text += getattr(page, "get_text")()
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        raise

def extract_txt_text(txt_file):
    """
    Extract text from a TXT file.
    Supports both file paths and file-like objects.
    """
    try:
        # Check if txt_file is a string (file path)
        if isinstance(txt_file, str):
            logger.info(f"Opening text file from path: {txt_file}")
            with open(txt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            # Assume it's a file-like object
            logger.info("Opening text file from file object")
            return txt_file.read().decode("utf-8").strip()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        if isinstance(txt_file, str):
            with open(txt_file, 'r', encoding='latin-1') as f:
                return f.read().strip()
        else:
            # Reset file pointer and try again with different encoding
            txt_file.seek(0)
            return txt_file.read().decode("latin-1").strip()
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise
