# RAG Chat App 

A simple Retrieval-Augmented Generation (RAG) chatbot that allows users to chat with pre-loaded documents.

## Features
- Pre-loaded document knowledge base
- Vector embeddings using OpenAI's text-embedding-3-small
- Vector storage with Chroma
- RAG pipeline with LangChain
- Interactive Streamlit interface

## Project Structure
```
rag_chat_app/
├── src/                    # Source code modules
├── data/                   # Pre-loaded documents for demo
├── main.py                 # Streamlit app entry point
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Set OpenAI API key: `export OPENAI_API_KEY=your_key_here`
3. Run the app: `streamlit run main.py`

## Demo Use Case
Perfect for conference demos showing how AI can answer questions about pre-loaded documents with source citations.