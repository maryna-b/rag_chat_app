import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.factory import RAGSystemFactory
    from src.config import ConfigurationManager
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure all dependencies are installed: pip install -r requirements.txt")
    st.stop()

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="RAG Chat App",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("Chat with Your Documents")
    st.markdown("Ask questions about your documents and get accurate, source-cited answers!")
    
    # Check configuration
    try:
        config_manager = ConfigurationManager()
        validation_errors = config_manager.get_validation_errors()
        
        if validation_errors:
            st.error("Configuration Issues:")
            for error in validation_errors:
                st.error(f"• {error}")
            
            if any("OPENAI_API_KEY" in error for error in validation_errors):
                st.info("Please set your OPENAI_API_KEY in a .env file")
                st.info("You can copy .env.example to .env and add your OpenAI API key")
            
            st.stop()
        
        config = config_manager.get_config()
        
    except Exception as e:
        st.error(f"Configuration error: {e}")
        st.stop()
    
    # Initialize RAG pipeline (with caching)
    @st.cache_resource
    def initialize_rag_system():
        try:
            factory = RAGSystemFactory()
            rag_pipeline = factory.create_rag_pipeline()
            
            if rag_pipeline.initialize(config.data_directory):
                return rag_pipeline, factory.get_logger()
            else:
                return None, None
                
        except Exception as e:
            error_msg = str(e)
            if "chromadb" in error_msg.lower():
                st.error("Missing ChromaDB dependency!")
                st.info("Please install ChromaDB to enable citations:")
                st.code("pip install chromadb")
                st.warning("Without ChromaDB, the app will work but won't show source citations.")
            else:
                st.error(f"Error initializing RAG system: {e}")
            return None, None
    
    rag_pipeline, logger = initialize_rag_system()
    
    if not rag_pipeline:
        st.error("Failed to initialize RAG system. Please check your configuration and logs.")
        st.stop()
    
    # Sidebar with document info
    with st.sidebar:
        st.header("Knowledge Base")
        st.markdown("**Available Documents:**")
        
        # Show available documents
        if os.path.exists(config.data_directory):
            documents = [f for f in os.listdir(config.data_directory) if f.endswith('.txt')]
            if documents:
                for doc in documents:
                    st.markdown(f"• {doc}")
            else:
                st.warning("No .txt files found in data directory")
        else:
            st.error(f"Data directory not found: {config.data_directory}")
        
        st.markdown("---")
        st.markdown("**Configuration:**")
        st.markdown(f"• Model: {config.openai_model}")
        st.markdown(f"• Chunk Size: {config.chunk_size}")
        st.markdown(f"• Search Results: {config.search_k}")
        
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
                    
                    # Add enhanced source information with similarity scores
                    if result["sources"]:
                        response_text += "\n\n---\n**📚 Sources & Citations:**\n\n"
                        for i, source in enumerate(result["sources"], 1):
                            response_text += f"**[{i}] {source['filename']}**"
                            if source.get('chunk_id') != 'N/A':
                                response_text += f" _(chunk {source['chunk_id']})_"
                            # Add similarity score if available
                            if source.get('similarity_score') is not None:
                                response_text += f" - **{source['similarity_score']}% similarity**"
                            response_text += "\n"
                            response_text += f"> _{source['content_preview']}_\n\n"
                    else:
                        # Only show note for answers that seem to have content but no sources
                        answer_lower = result["answer"].lower()
                        no_info_phrases = ["don't have information", "no mention", "cannot determine", "not possible to determine"]
                        has_no_info = any(phrase in answer_lower for phrase in no_info_phrases)
                        
                        if not has_no_info:
                            response_text += "\n\n_📝 Note: This response was generated without specific document references._"
                    
                    st.markdown(response_text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Query failed: {prompt}", e)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Footer
    st.markdown("---")
    st.markdown("**System Status:** ✅ Refactored RAG system active with SOLID principles!")

if __name__ == "__main__":
    main()