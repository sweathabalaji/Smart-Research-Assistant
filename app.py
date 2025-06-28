import streamlit as st
from utils.pdf_utils import extract_pdf_text, extract_txt_text
from utils.summarizer import generate_summary
from utils.qa_engine import answer_question, generate_questions
from utils.evaluator import evaluate_answer
import time
import traceback
import logging
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Smart Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing document text
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'error' not in st.session_state:
    st.session_state.error = None
if 'debug' not in st.session_state:
    st.session_state.debug = ""
if 'document_content' not in st.session_state:
    st.session_state.document_content = None

# App header
st.markdown('<h1 class="main-header">📚 Smart Assistant for Research Summarization</h1>', unsafe_allow_html=True)

# Sidebar with instructions
with st.sidebar:
    st.header("How to use")
    st.markdown("""
    1. Upload a PDF or TXT file
    2. Review the generated summary
    3. Choose a mode:
       - **Ask Anything**: Ask questions about the document
       - **Challenge Me**: Test your knowledge with auto-generated questions
    """)
    
    # Clear data button
    if st.button("Clear All Data"):
        st.session_state.raw_text = None
        st.session_state.summary = None
        st.session_state.file_name = None
        st.session_state.questions = []
        st.session_state.scores = []
        st.session_state.answers = []
        st.session_state.error = None
        st.session_state.debug = ""
        st.session_state.document_content = None
        st.rerun()

# Display any errors
if st.session_state.error:
    st.markdown(f'<div class="error-box">❌ {st.session_state.error}</div>', unsafe_allow_html=True)
    if st.button("Clear Error"):
        st.session_state.error = None
        st.rerun()

# Upload Section
st.markdown('<h2 class="sub-header">📎 Upload Document</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file and (st.session_state.file_name != uploaded_file.name):
    # Reset state when a new file is uploaded
    st.session_state.raw_text = None
    st.session_state.summary = None
    st.session_state.file_name = uploaded_file.name
    st.session_state.questions = []
    st.session_state.scores = []
    st.session_state.answers = []
    st.session_state.error = None
    st.session_state.debug = ""
    st.session_state.document_content = None
    
    # Extract text with progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Processing document...")
    progress_bar.progress(25)
    
    try:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Extract text based on file type
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            st.session_state.raw_text = extract_pdf_text(temp_path)
        elif file_type == "text/plain":
            st.session_state.raw_text = extract_txt_text(temp_path)
        else:
            st.session_state.error = "Unsupported file format. Please use PDF or TXT files."
            os.unlink(temp_path)  # Remove temporary file
            progress_bar.empty()
            status_text.empty()
            st.rerun()
            st.stop()
        
        # Store the document content for later use
        st.session_state.document_content = st.session_state.raw_text
        
        progress_bar.progress(50)
        status_text.text("Generating summary...")
        
        # Generate summary
        st.session_state.summary = generate_summary(st.session_state.raw_text)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        progress_bar.progress(100)
        status_text.text("✅ Document processed successfully!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        st.markdown('<div class="success-box">✅ File uploaded and processed successfully.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.session_state.error = f"Error processing file: {str(e)}"
        # Clean up temporary file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        progress_bar.empty()
        status_text.empty()
        st.rerun()

# Display content if document is loaded
if st.session_state.raw_text:
    # Summary Section
    st.markdown('<h2 class="sub-header">📄 Document Summary</h2>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write(st.session_state.summary)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Document stats
    col1, col2 = st.columns(2)
    with col1:
        # Fix for "split" is not a known attribute of "None"
        doc_length = 0
        if st.session_state.raw_text:
            doc_length = len(st.session_state.raw_text.split())
        st.metric("Document Length", f"{doc_length} words")
    with col2:
        # Fix for potential None summary
        summary_length = 0
        if st.session_state.summary:
            summary_length = len(st.session_state.summary.split())
        st.metric("Summary Length", f"{summary_length} words")
    
    # Mode Selection
    st.markdown('<h2 class="sub-header">🔍 Select Mode</h2>', unsafe_allow_html=True)
    mode = st.radio("Choose a Mode:", ["Ask Anything", "Challenge Me"])
    
    # Mode 1 - Ask Anything
    if mode == "Ask Anything":
        st.markdown('<h3>💬 Ask Questions</h3>', unsafe_allow_html=True)
        question = st.text_input("Enter your question about the document:")
        
        if question:
            try:
                with st.spinner("Finding the best answer..."):
                    # Make sure we're using the document content
                    answer = answer_question(question, st.session_state.document_content)
                
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.session_state.error = f"Error answering question: {str(e)}"
                st.rerun()
    
    # Mode 2 - Challenge Me
    elif mode == "Challenge Me":
        if st.button("🧠 Generate Quiz Questions") or len(st.session_state.questions) == 0:
            try:
                with st.spinner("Creating quiz questions based on document content..."):
                    # Make sure we're using the document content for question generation
                    if st.session_state.document_content:
                        st.session_state.questions = generate_questions(st.session_state.document_content)
                        st.session_state.scores = [None] * len(st.session_state.questions)
                        st.session_state.answers = [None] * len(st.session_state.questions)
                        st.rerun()  # Rerun to update UI with new questions
                    else:
                        st.error("No document content available. Please upload a document.")
            except Exception as e:
                st.session_state.error = f"Error generating questions: {str(e)}"
                st.rerun()
        
        if st.session_state.questions:
            st.markdown('<h3>✍️ Answer the Questions</h3>', unsafe_allow_html=True)
            
            # Display overall score
            valid_scores = [s for s in st.session_state.scores if s is not None]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                st.progress(avg_score)
                st.markdown(f"**Current Score:** {int(avg_score * 100)}%")
            
            # Display questions
            for i, question in enumerate(st.session_state.questions):
                st.markdown(f"**Question {i+1}:** {question}")
                
                # Use a key that includes the question to ensure uniqueness
                user_ans_key = f"user_ans_{i}_{hash(question)}"
                user_ans = st.text_area(f"Your answer for Q{i+1}", key=user_ans_key, height=100)
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    # Use a unique key for the submit button
                    submit_key = f"submit_{i}_{hash(question)}"
                    if st.button(f"Submit Answer #{i+1}", key=submit_key):
                        try:
                            with st.spinner("Evaluating answer..."):
                                # Get the correct answer from the document content
                                correct_ans = answer_question(question, st.session_state.document_content)
                                
                                # Store the correct answer for reference
                                st.session_state.answers[i] = correct_ans
                                
                                # Log for debugging
                                logger.info(f"Question: {question}")
                                logger.info(f"User Answer: {user_ans}")
                                logger.info(f"Correct Answer: {correct_ans}")
                                
                                # Evaluate the user's answer
                                feedback, score = evaluate_answer(user_ans, correct_ans)
                                
                                # Store the score
                                st.session_state.scores[i] = float(score)
                                
                                # Debug information
                                st.session_state.debug = f"Question: {question}\nUser Answer: {user_ans}\nCorrect Answer: {correct_ans}\nScore: {score}\nFeedback: {feedback}"
                                
                                # Force a rerun to update the UI
                                st.rerun()
                        except Exception as e:
                            error_msg = f"Error evaluating answer: {str(e)}\n{traceback.format_exc()}"
                            logger.error(error_msg)
                            st.session_state.error = error_msg
                            st.rerun()
                
                # Display feedback if we have a score for this question
                if i < len(st.session_state.scores) and st.session_state.scores[i] is not None:
                    with col2:
                        # Get the feedback based on the score
                        score = st.session_state.scores[i]
                        if score >= 0.8:
                            feedback = "✅ Excellent answer! Your response matches the key points perfectly."
                        elif score >= 0.6:
                            feedback = "✓ Good answer! You've captured most of the important points."
                        elif score >= 0.4:
                            feedback = "⚠️ Partially correct. Your answer contains some relevant information, but misses key points."
                        elif score >= 0.2:
                            feedback = "⚠️ Your answer is on the right track but needs more specific details."
                        else:
                            feedback = "❌ Your answer doesn't match the expected response. Try again with more specific information."
                        
                        st.markdown(f"**Feedback:** {feedback}")
                        st.markdown(f"**Score:** {int(score * 100)}%")
                    
                    # Show correct answer if score is low
                    if score < 0.5:
                        with st.expander("See suggested answer"):
                            if i < len(st.session_state.answers) and st.session_state.answers[i]:
                                st.write(st.session_state.answers[i])
                            else:
                                try:
                                    correct_ans = answer_question(question, st.session_state.document_content)
                                    st.write(correct_ans)
                                except Exception as e:
                                    st.write("Error retrieving answer.")
                
                st.markdown("---")
            
            # Debug section (uncomment for debugging)
            with st.expander("Debug Information"):
                st.text(st.session_state.debug)
else:
    st.info("Please upload a document to get started.")
