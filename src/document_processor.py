from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .interfaces import DocumentProcessorInterface, LoggerInterface
from .config import RAGConfig


class DocumentChunker(DocumentProcessorInterface):
    """Handles document chunking with configurable parameters."""
    
    def __init__(self, config: RAGConfig, logger: LoggerInterface):
        self._config = config
        self._logger = logger
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks with metadata.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            self._logger.warning("No documents provided for chunking")
            return []
        
        self._logger.info(f"Chunking {len(documents)} documents")
        chunked_docs = []
        
        for doc in documents:
            chunks = self._text_splitter.split_documents([doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['total_chunks'] = len(chunks)
                chunk.metadata['original_source'] = doc.metadata.get('source', 'Unknown')
            
            chunked_docs.extend(chunks)
        
        self._logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs


class DocumentPipeline:
    """Orchestrates the document processing pipeline."""
    
    def __init__(self, loader, processor, logger: LoggerInterface):
        self._loader = loader
        self._processor = processor
        self._logger = logger
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Complete document processing pipeline: load and chunk documents.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of processed and chunked Document objects
        """
        try:
            self._logger.info(f"Starting document pipeline for: {directory_path}")
            
            # Load documents
            documents = self._loader.load_documents(directory_path)
            
            if not documents:
                self._logger.warning("No documents loaded from directory")
                return []
            
            # Process (chunk) documents
            chunked_documents = self._processor.process_documents(documents)
            
            self._logger.info(f"Pipeline completed: {len(chunked_documents)} total chunks")
            return chunked_documents
            
        except Exception as e:
            self._logger.error(f"Document pipeline failed", e)
            raise