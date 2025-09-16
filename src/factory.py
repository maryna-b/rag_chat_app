"""
Factory pattern implementation for creating and wiring RAG system components.
Follows dependency inversion principle.
"""
from .config import ConfigurationManager, RAGConfig
from .logger import RAGLogger
from .document_loader import TextDocumentLoader
from .document_processor import DocumentChunker, DocumentPipeline
from .vector_store import ChromaVectorStore, VectorStoreManager
from .rag_pipeline import RAGPipeline


class RAGSystemFactory:
    """Factory for creating and wiring RAG system components."""
    
    def __init__(self):
        # Create shared configuration and logger
        self._config_manager = ConfigurationManager()
        self._logger = RAGLogger()
        
        if not self._config_manager.validate():
            errors = self._config_manager.get_validation_errors()
            for error in errors:
                self._logger.error(error)
            raise ValueError("Configuration validation failed")
        
        self._config = self._config_manager.get_config()
        self._logger.info("RAG system factory initialized")
    
    def create_document_loader(self):
        """Create document loader instance."""
        return TextDocumentLoader(self._config, self._logger)
    
    def create_document_processor(self):
        """Create document processor instance."""
        return DocumentChunker(self._config, self._logger)
    
    def create_document_pipeline(self):
        """Create document pipeline with all dependencies."""
        loader = self.create_document_loader()
        processor = self.create_document_processor()
        return DocumentPipeline(loader, processor, self._logger)
    
    def create_vector_store(self):
        """Create vector store instance."""
        return ChromaVectorStore(self._config, self._logger)
    
    def create_vector_store_manager(self):
        """Create vector store manager with dependencies."""
        vector_store = self.create_vector_store()
        return VectorStoreManager(vector_store, self._logger)
    
    def create_rag_pipeline(self):
        """Create complete RAG pipeline with all dependencies."""
        document_pipeline = self.create_document_pipeline()
        vector_store_manager = self.create_vector_store_manager()
        
        return RAGPipeline(
            config=self._config,
            document_pipeline=document_pipeline,
            vector_store_manager=vector_store_manager,
            logger=self._logger
        )
    
    def get_config(self) -> RAGConfig:
        """Get configuration object."""
        return self._config
    
    def get_logger(self):
        """Get logger instance."""
        return self._logger