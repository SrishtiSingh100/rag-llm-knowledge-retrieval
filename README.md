Retrieval-Augmented Generation (RAG) using LLMs

An end-to-end AI-powered medical chatbot that answers user queries by retrieving relevant information from medical literature and generating accurate, context-aware responses using Large Language Models.


Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline over medical documents.
Instead of relying solely on a language model’s memory, the system retrieves relevant medical knowledge from a PDF corpus and uses it as context to generate reliable answers.

The chatbot is designed with:

Scalability

Modularity

Reproducibility

Research-oriented system design


Key Features

1. PDF-based medical knowledge ingestion

2. Semantic search using vector embeddings

3. Context-aware answers with LLMs

4. Fast similarity search using FAISS

5. Interactive Flask web interface

6. Modern dark-themed UI

7. Modular and extensible codebase


 System Architecture

Medical PDF
    ↓
Document Loader
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
LLM (Answer Generation)
    ↓
Flask Web App (Chat Interface)

Demo & Screenshots
 Chat Interface Screenshot

(Add a screenshot of the chatbot UI here)

![Chatbot UI](screenshots/chat_ui.png)

🔹 Screen Recording (Demo)

(Add a short demo video or GIF here)



Tech Stack
Backend & AI

Python

LangChain

FAISS

Hugging Face Embeddings

Open-source LLMs


Web Framework

Flask

HTML / CSS / JavaScript


Tools & Libraries

Conda (environment management)

NumPy

PyPDF

Git


Project Structure
├── data
│   └── Medical_book.pdf
│
├── research
│   └── trials.ipynb
│
├── src
│   ├── __init__.py
│   ├── helper.py
│   └── prompt.py
│
├── static
│   └── style.css
│
├── templates
│   └── chat.html
│
├── faiss_index
│   ├── index.faiss
│   └── index.pkl
│
├── app.py
├── requirements.txt
├── setup.py
├── template.sh
├── LICENSE
└── README.md

How It Works

Document Loading
Medical PDFs are loaded and parsed into raw text.

Text Chunking
Documents are split into smaller, overlapping chunks for better semantic retrieval.

Embedding Generation
Each chunk is converted into a dense vector representation.

Vector Storage
Embeddings are stored in a FAISS vector database for efficient similarity search.

Query Processing
User queries are embedded and matched against the vector store.

Answer Generation
The most relevant chunks are passed to the LLM to generate grounded answers.

🚀 Getting Started
1️⃣ Create Conda Environment
conda create -n rag-llm python=3.10
conda activate rag-llm

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Build Vector Store (if not already created)

Run the notebook:

research/trials.ipynb

4️⃣ Run the Application
python app.py

5️⃣ Open in Browser
http://127.0.0.1:5000

💬 Sample Queries

What are the symptoms of diabetes?

Explain hypertension in simple terms.

What are common treatments for asthma?

🔒 Disclaimer

⚠️ This chatbot is for educational and research purposes only.
It is not a substitute for professional medical advice.

📈 Future Enhancements

✅ Source citations with each response

✅ Chat memory for multi-turn conversations

✅ Evaluation metrics for retrieval quality

✅ Deployment to cloud platforms (AWS / Render / HF Spaces)

✅ Support for multiple document uploads

👩‍💻 Author

Srishti Singh
📎 GitHub: https://github.com/SrishtiSingh100

📎 LinkedIn: https://www.linkedin.com/in/srishtisingh01/


