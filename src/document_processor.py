"""
Document processing module for loading and chunking text files.
"""
import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """
        Load all text files from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of Document objects
        """
        documents = []
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                try:
                    # Try multiple encodings
                    for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                        try:
                            loader = TextLoader(file_path, encoding=encoding)
                            docs = loader.load()
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise UnicodeDecodeError("Could not decode file with any supported encoding")
                    
                    # Add source metadata
                    for doc in docs:
                        doc.metadata['source'] = filename
                        doc.metadata['file_path'] = file_path
                    
                    documents.extend(docs)
                    print(f"[OK] Loaded: {filename}")
                    
                except Exception as e:
                    print(f"[ERROR] Error loading {filename}: {str(e)}")
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['total_chunks'] = len(chunks)
            
            chunked_docs.extend(chunks)
        
        return chunked_docs
    
    def process_documents(self, directory_path: str) -> List[Document]:
        """
        Complete document processing pipeline: load and chunk documents.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of processed and chunked Document objects
        """
        print(f"Loading documents from: {directory_path}")
        documents = self.load_documents_from_directory(directory_path)
        
        print(f"Loaded {len(documents)} documents")
        print(f"Chunking documents...")
        
        chunked_documents = self.chunk_documents(documents)
        
        print(f"Created {len(chunked_documents)} chunks")
        
        return chunked_documents


def main():
    """Test the document processor"""
    processor = DocumentProcessor()
    docs = processor.process_documents("data")
    
    print(f"\nProcessing Summary:")
    print(f"Total chunks: {len(docs)}")
    
    if docs:
        print(f"First chunk preview:")
        print(f"Source: {docs[0].metadata['source']}")
        print(f"Content: {docs[0].page_content[:200]}...")


if __name__ == "__main__":
    main()