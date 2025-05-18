# VITUAL-STUDY-ASSISTANT-USING-AI
Student-Chatbot
It is a chatbot that takes input of documents which are in the form of .ppt or .pdf and after uploading it you can ask any questions regarding that document and it will answer those question if it is available in that document.

University Student Support Chatbot
A chatbot that answers academic queries using official university PDFs and PPTs. Built with LangChain, Groq LLM, and ChromaDB for vector-based search.

Features
PDF and PowerPoint ingestion
Context-aware question answering
Uses MistralAI embeddings + LLaMA3-70B via Groq
Fallback to LLM knowledge when needed
How to Run
Clone the repo
Install requirements: pip install -r requirements.txt
Set your GROQ_API_KEY in environment variables
Place documents in the Data/ folder
Run: python main.py

SUMMARIZER:
This project is a command-line based text summarizer built using Hugging Face's transformers library and the powerful facebook/bart-large-cnn model. It allows users to input long-form text and receive a concise, readable summary using state-of-the-art NLP.

 Features
 Uses the BART-large-CNN model for high-quality abstractive summarization.

 Handles long texts by automatically chunking them.

 Configurable summarization parameters for better control over output length and diversity.

 Simple command-line interface – just paste your text and get a summary.
 Tech Stack
Python 3.x

Hugging Face Transformers

textwrap (standard library)

 How It Works
Prompts the user to paste in a block of text.

Splits the text into manageable chunks (if necessary).

Summarizes each chunk using facebook/bart-large-cnn.

Outputs a final combined summary.
