"""
RAG Chat App - Main Streamlit Application
A demo chatbot that answers questions about pre-loaded meeting documents.
"""
import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from rag_pipeline import RAGPipeline
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="RAG Chat App",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("Chat with Your Meeting Documents")
    st.markdown("Ask questions about your meeting transcripts and get accurate, source-cited answers!")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set your OPENAI_API_KEY in a .env file")
        st.info("Copy .env.example to .env and add your OpenAI API key")
        st.stop()
    
    # Initialize RAG pipeline (with caching)
    @st.cache_resource
    def initialize_rag_pipeline():
        try:
            rag = RAGPipeline()
            if rag.setup_vector_store():
                return rag
            else:
                return None
        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {e}")
            return None
    
    rag_pipeline = initialize_rag_pipeline()
    
    if not rag_pipeline:
        st.error("Failed to initialize RAG pipeline. Please check your setup.")
        st.stop()
    
    # Sidebar with document info
    with st.sidebar:
        st.header("Knowledge Base")
        st.markdown("**Available Documents:**")
        
        # Show available documents
        data_dir = "data"
        if os.path.exists(data_dir):
            documents = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
            for doc in documents:
                st.markdown(f"• {doc}")
        else:
            st.error("Data directory not found!")
        
        st.markdown("---")
        st.markdown("**Example Questions:**")
        st.markdown("• What was discussed about budget?")
        st.markdown("• Who are the team members mentioned?")
        st.markdown("• What are the action items?")
        st.markdown("• When is the next meeting?")
    
    # Main chat interface
    st.header("Ask a Question")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your meeting documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response using RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching through documents..."):
                try:
                    # Query the RAG pipeline
                    result = rag_pipeline.query(prompt)
                    
                    # Format the response
                    response_text = result["answer"]
                    
                    # Add source information
                    if result["sources"]:
                        response_text += "\n\n**Sources:**\n"
                        for i, source in enumerate(result["sources"], 1):
                            response_text += f"{i}. **{source['filename']}**\n"
                            response_text += f"   _{source['content_preview']}_\n\n"
                    
                    st.markdown(response_text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Footer
    st.markdown("---")
    st.markdown("**Demo Status:** ✅ RAG pipeline active! Ask questions about your meeting documents.")

if __name__ == "__main__":
    main()