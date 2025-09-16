"""
Abstract interfaces for the RAG system following SOLID principles.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.schema import Document


class DocumentLoaderInterface(ABC):
    """Interface for document loading operations."""
    
    @abstractmethod
    def load_documents(self, source: str) -> List[Document]:
        """Load documents from a source."""
        pass


class DocumentProcessorInterface(ABC):
    """Interface for document processing operations."""
    
    @abstractmethod
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents (chunking, metadata addition, etc.)."""
        pass


class VectorStoreInterface(ABC):
    """Interface for vector store operations."""
    
    @abstractmethod
    def create_store(self, documents: List[Document]) -> Any:
        """Create a new vector store."""
        pass
    
    @abstractmethod
    def load_store(self) -> Optional[Any]:
        """Load existing vector store."""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing store."""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        pass


class RAGInterface(ABC):
    """Interface for RAG operations."""
    
    @abstractmethod
    def initialize(self, data_source: str) -> bool:
        """Initialize the RAG system."""
        pass
    
    @abstractmethod
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system."""
        pass


class ConfigurationInterface(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate configuration."""
        pass


class LoggerInterface(ABC):
    """Interface for logging operations."""
    
    @abstractmethod
    def info(self, message: str) -> None:
        """Log info message."""
        pass
    
    @abstractmethod
    def error(self, message: str, exception: Exception = None) -> None:
        """Log error message."""
        pass
    
    @abstractmethod
    def debug(self, message: str) -> None:
        """Log debug message."""
        pass