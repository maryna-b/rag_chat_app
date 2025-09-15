"""
Vector store module for handling embeddings and Chroma database operations.
"""
import os
from typing import List, Optional
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb


class VectorStoreManager:
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "meeting_documents",
                 embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Directory to persist the Chroma database
            collection_name: Name of the Chroma collection
            embedding_model: OpenAI embedding model to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize or load existing vector store
        self.vector_store = None
        
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to embed and store
            
        Returns:
            Chroma vector store instance
        """
        print(f"Creating vector store with {len(documents)} documents...")
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        print(f"Vector store created successfully!")
        print(f"Persisted to: {self.persist_directory}")
        
        return self.vector_store
    
    def load_existing_vector_store(self) -> Optional[Chroma]:
        """
        Load an existing vector store from disk.
        
        Returns:
            Chroma vector store instance if exists, None otherwise
        """
        if os.path.exists(self.persist_directory):
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                
                # Test if the collection has any documents
                count = self.vector_store._collection.count()
                if count > 0:
                    print(f"Loaded existing vector store with {count} documents")
                    return self.vector_store
                else:
                    print("Vector store exists but is empty")
                    return None
                    
            except Exception as e:
                print(f"Error loading existing vector store: {str(e)}")
                return None
        else:
            print("No existing vector store found")
            return None
    
    def get_or_create_vector_store(self, documents: List[Document] = None) -> Chroma:
        """
        Get existing vector store or create a new one.
        
        Args:
            documents: Documents to use if creating a new vector store
            
        Returns:
            Chroma vector store instance
        """
        # Try to load existing vector store first
        existing_store = self.load_existing_vector_store()
        
        if existing_store is not None:
            return existing_store
        
        # Create new vector store if none exists
        if documents is None:
            raise ValueError("No existing vector store found and no documents provided to create new one")
        
        return self.create_vector_store(documents)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to the existing vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call get_or_create_vector_store first.")
        
        print(f"Adding {len(documents)} documents to vector store...")
        self.vector_store.add_documents(documents)
        print("Documents added successfully!")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call get_or_create_vector_store first.")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform similarity search with similarity scores.
        
        Args:
            query: Search query
            k: Number of similar documents to return
            
        Returns:
            List of tuples (document, score)
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call get_or_create_vector_store first.")
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def delete_collection(self) -> None:
        """Delete the vector store collection and files."""
        try:
            if self.vector_store:
                self.vector_store.delete_collection()
            
            # Remove the persist directory
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                print(f"Deleted vector store: {self.persist_directory}")
        except Exception as e:
            print(f"Error deleting vector store: {str(e)}")


def main():
    """Test the vector store functionality"""
    from document_processor import DocumentProcessor
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY environment variable")
        return
    
    # Load and process documents
    processor = DocumentProcessor()
    documents = processor.process_documents("data")
    
    # Initialize vector store manager
    vector_manager = VectorStoreManager()
    
    # Create or load vector store
    vector_store = vector_manager.get_or_create_vector_store(documents)
    
    # Test similarity search
    test_query = "What were the budget details discussed in meetings?"
    print(f"\nTesting similarity search with query: '{test_query}'")
    
    results = vector_manager.similarity_search(test_query, k=3)
    
    print(f"\nFound {len(results)} relevant chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()