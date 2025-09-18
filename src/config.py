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
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Vector store
    vector_store_path: str = "./chroma_db"
    collection_name: str = "meeting_documents"

    # Search settings
    search_k: int = 2
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

    def _get_config_value(self, key: str, default: str = "") -> str:
        """Get configuration value from environment variables or Streamlit secrets."""
        # Try environment variables first (for local development)
        value = os.getenv(key, "")

        # If not found and we're in Streamlit, try st.secrets
        if not value:
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and key in st.secrets:
                    value = st.secrets[key]
            except ImportError:
                pass  # Not in Streamlit environment
            except Exception:
                pass  # Secrets not available or other error

        return value if value else default

    def _load_config(self) -> RAGConfig:
        """Load configuration from environment variables or Streamlit secrets."""
        return RAGConfig(
            openai_api_key=self._get_config_value("OPENAI_API_KEY"),
            openai_model=self._get_config_value("OPENAI_MODEL", "gpt-3.5-turbo"),
            openai_temperature=float(self._get_config_value("OPENAI_TEMPERATURE", "0.1")),
            embedding_model=self._get_config_value("EMBEDDING_MODEL", "text-embedding-3-small"),
            chunk_size=int(self._get_config_value("CHUNK_SIZE", "500")),
            chunk_overlap=int(self._get_config_value("CHUNK_OVERLAP", "100")),
            vector_store_path=self._get_config_value("VECTOR_STORE_PATH", "./chroma_db"),
            collection_name=self._get_config_value("COLLECTION_NAME", "meeting_documents"),
            search_k=int(self._get_config_value("SEARCH_K", "2")),
            score_threshold=float(self._get_config_value("SCORE_THRESHOLD", "0.3")),
            data_directory=self._get_config_value("DATA_DIRECTORY", "data")
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