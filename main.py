"""
RAG Chat App - Main Streamlit Application
A demo chatbot that answers questions about pre-loaded meeting documents.
"""
import os
import streamlit as st
from dotenv import load_dotenv

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
        
        # Generate response (placeholder for now)
        with st.chat_message("assistant"):
            with st.spinner("Searching through documents..."):
                # TODO: Implement actual RAG pipeline
                response = f"""
I received your question: "{prompt}"

**Status:** RAG pipeline is being built! 

**Current Progress:**
✅ Document processing ready
✅ Vector store configured  
🔄 Building retriever and RAG pipeline...

**Coming soon:** Actual answers with source citations!
                """
                
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.markdown("**Demo Status:** Under Development - RAG pipeline coming soon!")

if __name__ == "__main__":
    main()