"""
Document loading functionality following single responsibility principle.
"""
import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader

from .interfaces import DocumentLoaderInterface, LoggerInterface
from .config import RAGConfig


class TextDocumentLoader(DocumentLoaderInterface):
    """Loads text documents from a directory with encoding handling."""
    
    def __init__(self, config: RAGConfig, logger: LoggerInterface):
        self._config = config
        self._logger = logger
    
    def load_documents(self, directory_path: str) -> List[Document]:
        """Load all text files from a directory with proper encoding handling."""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        documents = []
        txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        
        if not txt_files:
            self._logger.warning(f"No .txt files found in {directory_path}")
            return documents
        
        self._logger.info(f"Found {len(txt_files)} text files to process")
        
        for filename in txt_files:
            file_path = os.path.join(directory_path, filename)
            try:
                doc = self._load_single_file(file_path, filename)
                if doc:
                    documents.extend(doc)
                    self._logger.info(f"Successfully loaded: {filename}")
            except Exception as e:
                self._logger.error(f"Failed to load {filename}", e)
        
        return documents
    
    def _load_single_file(self, file_path: str, filename: str) -> List[Document]:
        """Load a single file trying multiple encodings."""
        for encoding in self._config.supported_encodings:
            try:
                loader = TextLoader(file_path, encoding=encoding)
                docs = loader.load()
                
                # Add source metadata
                for doc in docs:
                    doc.metadata['source'] = filename
                    doc.metadata['file_path'] = file_path
                    doc.metadata['encoding_used'] = encoding
                
                return docs
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self._logger.error(f"Error loading {filename} with {encoding}", e)
                continue
        
        raise UnicodeDecodeError(
            f"Could not decode {filename} with any supported encoding: "
            f"{', '.join(self._config.supported_encodings)}"
        )