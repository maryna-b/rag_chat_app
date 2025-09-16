"""
Simple test script for the RAG pipeline functionality.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_rag_pipeline():
    """Test the RAG pipeline with a simple question."""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY in a .env file")
        print("Copy .env.example to .env and add your OpenAI API key")
        return False
    
    try:
        from rag_pipeline import RAGPipeline
        
        print("Testing RAG Pipeline...")
        print("=" * 40)
        
        # Initialize RAG pipeline
        rag = RAGPipeline()
        
        # Setup vector store
        print("Setting up vector store...")
        if not rag.setup_vector_store():
            print("Failed to setup vector store")
            return False
        
        # Test a simple question
        test_question = "What budget information was discussed?"
        print(f"\nTesting question: {test_question}")
        print("-" * 30)
        
        result = rag.query(test_question)
        
        print(f"Answer: {result['answer']}")
        
        if result['sources']:
            print("\nSources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['filename']}")
        
        print("\n" + "=" * 40)
        print("RAG pipeline test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"Error testing RAG pipeline: {e}")
        return False

if __name__ == "__main__":
    success = test_rag_pipeline()
    if success:
        print("\n[OK] RAG pipeline is working! You can now run: streamlit run main.py")
    else:
        print("\n[ERROR] RAG pipeline test failed. Please fix the issues above.")