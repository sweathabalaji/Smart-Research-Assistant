# 🧠 Smart Assistant for Research Summarization

An AI-powered assistant that helps users interact with uploaded documents (PDF or TXT) by:
- Summarizing the content automatically
- Answering user questions with contextual understanding
- Generating logic-based questions and evaluating responses

---

## 🚀 Features

- 📄 **Document Upload** (PDF/TXT)
- ✨ **Automatic Summary** (≤ 150 words)
- 💬 **Ask Anything**: Ask any question and get document-based answers
- 🧠 **Challenge Me**: Get logic/comprehension questions, answer them, and receive intelligent feedback
- ✅ Grounded answers with references to document sections
- 🌐 **Web UI** using Streamlit

---

## 🛠️ Tech Stack

| Component          | Library/Tool             |
|-------------------|--------------------------|
| Language           | Python 3.7+              |
| Frontend UI        | Streamlit                |
| NLP Models         | Hugging Face Transformers (BERT, GPT-2) |
| PDF Text Extraction| pdfminer.six             |
| Text Processing    | NLTK                     |
| Evaluation Logic   | Custom Python Logic      |

---

## Create and Activate Virtual Environment
**Windows:**
python -m venv assistant_env
assistant_env\Scripts\activate
**Mac/Linux:**
python3 -m venv assistant_env
source assistant_env/bin/activate
## Install Dependencies
pip install -r requirements.txt
## 🧪 Running the Application
streamlit run app.py

## Usage Guide
**Uploading a Document**
- Use the upload button to add a .pdf or .txt file.
- Once uploaded, a summary of the document will be shown automatically.
**Ask Anything Mode**
- Type any question based on the uploaded document.
- Get accurate answers with justification and references to document text.
**Challenge Me Mode**
- Automatically generates 3 logic-based/comprehension questions.
- Submit your answers and receive evaluations with document references.

## Local Deployment
streamlit run app.py

