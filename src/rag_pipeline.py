"""
RAG Pipeline module for implementing retrieval-augmented generation.
"""
import os
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager


class RAGPipeline:
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1):
        """
        Initialize the RAG pipeline.
        
        Args:
            model_name: OpenAI model to use for generation
            temperature: Temperature for response generation
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_manager = VectorStoreManager()
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize vector store
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
        # Custom prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following pieces of context from meeting documents to answer the question. 
Focus only on information directly relevant to the question asked.
If you don't know the answer based on the context, just say that you don't know.
Provide a clear, well-formatted response.

Context from meeting documents:
{context}

Question: {question}

Answer:"""
        )
    
    def setup_vector_store(self, data_directory: str = "data") -> bool:
        """
        Set up the vector store with documents from the specified directory.
        
        Args:
            data_directory: Directory containing the documents
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Setting up vector store from: {data_directory}")
            
            # Process documents
            documents = self.document_processor.process_documents(data_directory)
            
            if not documents:
                print("No documents found to process")
                return False
            
            # Create or load vector store
            self.vector_store = self.vector_manager.get_or_create_vector_store(documents)
            
            # Create retriever with more selective parameters
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,  # Reduce number of chunks
                    "score_threshold": 0.7  # Only include highly relevant chunks
                }
            )
            
            print(f"Vector store setup complete with {len(documents)} document chunks")
            return True
            
        except Exception as e:
            print(f"Error setting up vector store: {str(e)}")
            return False
    
    def create_qa_chain(self):
        """Create the question-answering chain."""
        if not self.retriever:
            raise ValueError("Vector store not initialized. Call setup_vector_store first.")
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
        
        print("QA chain created successfully")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and source information
        """
        if not self.qa_chain:
            self.create_qa_chain()
        
        try:
            # Get response from QA chain
            result = self.qa_chain({"query": question})
            
            # Check if we got any source documents
            source_docs = result.get("source_documents", [])
            
            # If no documents found with similarity threshold, try regular similarity search
            if not source_docs:
                print("No documents found with similarity threshold, trying regular search...")
                regular_retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 2}
                )
                # Create a new QA chain with regular retriever
                from langchain.chains import RetrievalQA
                fallback_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=regular_retriever,
                    chain_type_kwargs={"prompt": self.prompt_template},
                    return_source_documents=True
                )
                result = fallback_chain({"query": question})
                source_docs = result.get("source_documents", [])
            
            # Format response
            response = {
                "answer": result["result"],
                "sources": [],
                "source_documents": source_docs
            }
            
            # Extract source information with better formatting
            for doc in source_docs:
                # Clean up the content preview
                content = doc.page_content.strip()
                preview = content[:150] + "..." if len(content) > 150 else content
                
                source_info = {
                    "filename": doc.metadata.get("source", "Unknown"),
                    "content_preview": preview
                }
                response["sources"].append(source_info)
            
            return response
            
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "source_documents": []
            }
    
    def get_relevant_documents(self, question: str, k: int = 4) -> List[Document]:
        """
        Get relevant documents for a question without generating an answer.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.retriever:
            raise ValueError("Vector store not initialized. Call setup_vector_store first.")
        
        return self.retriever.get_relevant_documents(question)


def main():
    """Test the RAG pipeline"""
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY environment variable")
        return
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Setup vector store
    if not rag.setup_vector_store():
        print("Failed to setup vector store")
        return
    
    # Test queries
    test_questions = [
        "What budget information was discussed in the meetings?",
        "Who are the team members mentioned?",
        "What are the main action items?",
        "When is the next meeting scheduled?"
    ]
    
    print("\nTesting RAG Pipeline:")
    print("=" * 50)
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 30)
        
        response = rag.query(question)
        print(f"Answer: {response['answer']}")
        
        if response['sources']:
            print("\nSources:")
            for i, source in enumerate(response['sources'], 1):
                print(f"{i}. {source['filename']}")
                print(f"   Preview: {source['content_preview']}")


if __name__ == "__main__":
    main()