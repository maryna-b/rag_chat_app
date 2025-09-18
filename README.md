# RAG Chat App

A Retrieval-Augmented Generation (RAG) chatbot that allows you to ask questions about your documents and get intelligent answers with source citations.

## Features
- **Document Processing**: Automatically loads and processes text documents from the data folder
- **Smart Search**: Uses vector embeddings to find relevant document sections
- **AI-Powered Answers**: Leverages OpenAI's GPT models to generate contextual responses
- **Source Citations**: Shows which documents were used to generate each answer
- **Interactive UI**: Web-based chat interface built with Streamlit
- **Persistent Storage**: Saves processed documents in a vector database for fast retrieval

## Project Structure

```
rag_chat_app/
├── src/                         # Source code modules
│   ├── interfaces.py           # Component interfaces
│   ├── config.py               # Configuration management
│   ├── logger.py               # Logging setup
│   ├── document_loader.py      # Text file loading
│   ├── document_processor.py   # Document chunking
│   ├── vector_store.py         # Vector database operations
│   ├── rag_pipeline.py         # Question-answering pipeline
│   └── factory.py              # Component initialization
├── data/                       # Your documents go here
├── tests.py                    # Test suite
├── main.py                     # Streamlit web app
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API key:**
   ```bash
   # Create .env file with your API key
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

3. **Run tests to verify setup:**
   ```bash
   python tests.py
   ```

4. **Run the application:**
   ```bash
   streamlit run main.py
   ```

## How It Works

1. **Document Loading**: The app scans the `data/` folder for text files and loads them
2. **Text Processing**: Documents are split into smaller chunks for better search accuracy
3. **Vector Embeddings**: Each chunk is converted to vector embeddings using OpenAI's API
4. **Storage**: Embeddings are stored in a ChromaDB vector database
5. **Query Processing**: When you ask a question, it finds the most relevant document chunks
6. **Answer Generation**: The relevant context is sent to GPT to generate a comprehensive answer

## Configuration

Set up your environment with these variables:

- `OPENAI_API_KEY` - Required: Your OpenAI API key
- `OPENAI_MODEL` - Optional: GPT model to use (default: gpt-3.5-turbo)
- `CHUNK_SIZE` - Optional: Size of document chunks (default: 1000)
- `CHUNK_OVERLAP` - Optional: Overlap between chunks (default: 200)
- `SEARCH_K` - Optional: Number of relevant chunks to retrieve (default: 3)

## Testing

Verify everything works correctly:
```bash
python tests.py
```

## Adding Your Documents

Simply place your text files in the `data/` folder. Supported formats:
- `.txt` files
- The app will automatically process them when you run it

## Deployment to Streamlit Cloud

### Step 1: Push to GitHub
1. Remove your `.env` file from the repo (it's already in `.gitignore`)
2. Push your code to a public GitHub repository

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set main file path: `main.py`

### Step 3: Add Secrets
In your Streamlit Cloud app:
1. Go to **Settings** → **Secrets**
2. Add your configuration:
```toml
OPENAI_API_KEY = "sk-your-actual-api-key-here"

# Optional: Override settings for production
OPENAI_MODEL = "gpt-4"
CHUNK_SIZE = "300"
SEARCH_K = "1"
```

### Step 4: Deploy
Your app will automatically deploy and be publicly accessible while keeping your API key secure!

## Troubleshooting Deployment

### ChromaDB SQLite3 Issue
If you see an SQLite3 version error during deployment, the app includes an automatic fix:
- `pysqlite3-binary` dependency provides a newer SQLite3 version
- Automatic SQLite3 patching for Streamlit Cloud compatibility
- ChromaDB will work seamlessly without any additional configuration needed

The deployment should work without any manual intervention!
