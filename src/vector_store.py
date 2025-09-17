import os
from typing import List, Optional, Any
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from .interfaces import VectorStoreInterface, LoggerInterface
from .config import RAGConfig


class ChromaVectorStore(VectorStoreInterface):
    """Vector store implementation using Chroma database."""
    
    def __init__(self, config: RAGConfig, logger: LoggerInterface):
        self._config = config
        self._logger = logger
        self._embeddings = OpenAIEmbeddings(model=config.embedding_model)
        self._vector_store = None
    
    def create_store(self, documents: List[Document]) -> Any:
        """Create a new vector store from documents."""
        if not documents:
            raise ValueError("Cannot create vector store with empty documents list")
        
        self._logger.info(f"Creating vector store with {len(documents)} documents")
        
        try:
            self._vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self._embeddings,
                persist_directory=self._config.vector_store_path,
                collection_name=self._config.collection_name
            )
            
            self._logger.info(f"Vector store created and persisted to: {self._config.vector_store_path}")
            return self._vector_store
            
        except Exception as e:
            self._logger.error("Failed to create vector store", e)
            raise
    
    def load_store(self) -> Optional[Any]:
        """Load existing vector store from disk."""
        if not os.path.exists(self._config.vector_store_path):
            self._logger.info("No existing vector store found")
            return None
        
        try:
            self._vector_store = Chroma(
                persist_directory=self._config.vector_store_path,
                embedding_function=self._embeddings,
                collection_name=self._config.collection_name
            )
            
            # Test if the collection has any documents
            count = self._vector_store._collection.count()
            if count > 0:
                self._logger.info(f"Loaded existing vector store with {count} documents")
                return self._vector_store
            else:
                self._logger.warning("Vector store exists but is empty")
                return None
                
        except Exception as e:
            self._logger.error("Error loading existing vector store", e)
            return None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the existing vector store."""
        if self._vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if not documents:
            self._logger.warning("No documents provided to add")
            return
        
        try:
            self._logger.info(f"Adding {len(documents)} documents to vector store")
            self._vector_store.add_documents(documents)
            self._logger.info("Documents added successfully")
            
        except Exception as e:
            self._logger.error("Failed to add documents to vector store", e)
            raise
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search on the vector store."""
        if self._vector_store is None:
            raise ValueError("Vector store not initialized")
        
        try:
            return self._vector_store.similarity_search(query, k=k)
        except Exception as e:
            self._logger.error(f"Search failed for query: {query}", e)
            raise
    
    def search_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """Perform similarity search with similarity scores."""
        if self._vector_store is None:
            raise ValueError("Vector store not initialized")
        
        try:
            return self._vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            self._logger.error(f"Search with scores failed for query: {query}", e)
            raise
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: dict = None):
        """Get a retriever for the vector store."""
        if self._vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if search_kwargs is None:
            search_kwargs = {"k": self._config.search_k}
        
        self._logger.debug(f"Creating retriever with search_type={search_type}, kwargs={search_kwargs}")
        
        retriever = self._vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        return retriever
    
    def delete_collection(self) -> None:
        """Delete the vector store collection and files."""
        try:
            if self._vector_store:
                self._vector_store.delete_collection()
            
            import shutil
            if os.path.exists(self._config.vector_store_path):
                shutil.rmtree(self._config.vector_store_path)
                self._logger.info(f"Deleted vector store: {self._config.vector_store_path}")
                
        except Exception as e:
            self._logger.error("Error deleting vector store", e)


class VectorStoreManager:
    """Manager class that handles vector store operations with proper lifecycle management."""
    
    def __init__(self, vector_store: VectorStoreInterface, logger: LoggerInterface):
        self._vector_store = vector_store
        self._logger = logger
    
    def get_or_create_store(self, documents: List[Document] = None) -> Any:
        """Get existing vector store or create a new one."""
        # Try to load existing vector store first
        existing_store = self._vector_store.load_store()
        
        if existing_store is not None:
            return existing_store
        
        # Create new vector store if none exists
        if documents is None:
            raise ValueError("No existing vector store found and no documents provided")
        
        return self._vector_store.create_store(documents)