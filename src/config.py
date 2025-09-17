import os
from typing import Any, Dict
from dataclasses import dataclass
from dotenv import load_dotenv

from .interfaces import ConfigurationInterface


@dataclass
class RAGConfig:
    """Configuration data class for RAG system."""
    
    # OpenAI settings
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.1
    embedding_model: str = "text-embedding-3-small"
    
    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Vector store
    vector_store_path: str = "./chroma_db"
    collection_name: str = "meeting_documents"
    
    # Search settings
    search_k: int = 3
    score_threshold: float = 0.3  # Lower threshold for better retrieval (0.0 = perfect match)
    
    # Data settings
    data_directory: str = "data"
    
    # Supported encodings for document loading
    supported_encodings: list = None
    
    def __post_init__(self):
        if self.supported_encodings is None:
            self.supported_encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']


class ConfigurationManager(ConfigurationInterface):
    """Manages application configuration with validation."""
    
    def __init__(self):
        load_dotenv()
        self._config = self._load_config()
    
    def _load_config(self) -> RAGConfig:
        """Load configuration from environment variables."""
        return RAGConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            vector_store_path=os.getenv("VECTOR_STORE_PATH", "./chroma_db"),
            collection_name=os.getenv("COLLECTION_NAME", "meeting_documents"),
            search_k=int(os.getenv("SEARCH_K", "3")),
            score_threshold=float(os.getenv("SCORE_THRESHOLD", "0.3")),
            data_directory=os.getenv("DATA_DIRECTORY", "data")
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return getattr(self._config, key, default)
    
    def get_config(self) -> RAGConfig:
        """Get the complete configuration object."""
        return self._config
    
    def validate(self) -> bool:
        """Validate that required configuration is present."""
        if not self._config.openai_api_key:
            return False
        
        if not os.path.exists(self._config.data_directory):
            return False
        
        return True
    
    def get_validation_errors(self) -> list:
        """Get detailed validation errors."""
        errors = []
        
        if not self._config.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
        
        if not os.path.exists(self._config.data_directory):
            errors.append(f"Data directory not found: {self._config.data_directory}")
        
        return errors