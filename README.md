# RAG Chat App (Refactored with SOLID Principles)

A Retrieval-Augmented Generation (RAG) chatbot refactored to follow SOLID principles and clean code practices. Chat with pre-loaded documents using a maintainable, extensible architecture.

## Features
- **Maintainable Architecture**: Follows SOLID principles with proper separation of concerns
- **Dependency Injection**: Clean factory pattern for component creation and wiring  
- **Configuration Management**: Centralized configuration with validation
- **Proper Error Handling**: Structured logging and error management
- **Extensible Design**: Interface-based design allowing easy component replacement
- **Vector embeddings**: Using OpenAI's text-embedding-3-small
- **Vector storage**: Persistent Chroma database
- **Interactive UI**: Streamlit interface with chat history

## Refactored Architecture

```
rag_chat_app/
├── src/                         # Clean, modular source code
│   ├── interfaces.py           # Abstract interfaces (SOLID)
│   ├── config.py               # Configuration management
│   ├── logger.py               # Structured logging
│   ├── document_loader.py      # Document loading (SRP)
│   ├── document_processor.py   # Document chunking (SRP)  
│   ├── vector_store.py         # Vector operations (SRP)
│   ├── rag_pipeline.py         # Main RAG logic (SRP)
│   └── factory.py              # Dependency injection (DIP)
├── data/                       # Sample documents
├── tests.py                    # Comprehensive test suite
├── main.py                     # Streamlit app (refactored)
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## SOLID Principles Applied

### Single Responsibility Principle (SRP)
- **DocumentLoader**: Only handles loading text files with encoding detection
- **DocumentChunker**: Only handles text chunking and metadata
- **VectorStore**: Only handles vector operations and similarity search
- **RAGPipeline**: Only handles question-answering logic
- **ConfigManager**: Only handles configuration and validation

### Open/Closed Principle (OCP)
- Interface-based design allows extending functionality without modifying existing code
- New document types can be added by implementing `DocumentLoaderInterface`
- New vector stores can be added by implementing `VectorStoreInterface`

### Liskov Substitution Principle (LSP)
- All implementations can be substituted for their interfaces without breaking functionality
- `ChromaVectorStore` can be replaced with any `VectorStoreInterface` implementation

### Interface Segregation Principle (ISP)
- Focused interfaces: `DocumentLoaderInterface`, `VectorStoreInterface`, `RAGInterface`
- Clients only depend on methods they actually use

### Dependency Inversion Principle (DIP)
- High-level modules depend on abstractions, not concretions
- Factory pattern provides clean dependency injection
- Configuration and logging are injected dependencies

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

## Configuration

The app uses environment variables for configuration:

- `OPENAI_API_KEY` - Required: Your OpenAI API key
- `OPENAI_MODEL` - Optional: Model to use (default: gpt-3.5-turbo)
- `CHUNK_SIZE` - Optional: Document chunk size (default: 1000)
- `CHUNK_OVERLAP` - Optional: Chunk overlap (default: 200)
- `SEARCH_K` - Optional: Number of search results (default: 3)

## Testing

Run the comprehensive test suite:
```bash
python tests.py
```

The test suite validates:
- Project structure and file integrity
- Configuration management
- Factory pattern and dependency injection  
- Document processing pipeline
- RAG functionality (requires API key)

## Code Quality

This refactored version emphasizes:
- **Maintainability**: Clean, focused classes with single responsibilities
- **Testability**: Dependency injection enables easy unit testing
- **Extensibility**: Interface-based design supports new implementations
- **Reliability**: Proper error handling and logging throughout
- **Configurability**: Centralized configuration management

## Use Cases

Perfect for:
- Learning SOLID principles in practice
- Building maintainable RAG applications
- Conference demos with clean, professional code
- Educational examples of proper software architecture