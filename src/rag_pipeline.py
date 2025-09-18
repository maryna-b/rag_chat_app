from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from .interfaces import RAGInterface, LoggerInterface
from .config import RAGConfig
from .document_processor import DocumentPipeline


class RAGPipeline(RAGInterface):
    """Main RAG pipeline implementing question-answering over documents."""
    
    def __init__(self, 
                 config: RAGConfig,
                 document_pipeline: DocumentPipeline,
                 vector_store_manager,
                 logger: LoggerInterface):
        self._config = config
        self._document_pipeline = document_pipeline
        self._vector_store_manager = vector_store_manager
        self._logger = logger
        
        # Initialize LLM
        self._llm = ChatOpenAI(
            model_name=config.openai_model,
            temperature=config.openai_temperature
        )
        
        # Initialize components
        self._vector_store = None
        self._retriever = None
        self._qa_chain = None
        
        # Custom prompt template
        self._prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following pieces of context from meeting documents to answer the question. 

IMPORTANT INSTRUCTIONS:
- Only use information that is directly relevant to answer the specific question
- If the context does not contain information to answer the question, respond with: "I don't have information about this in the meeting documents."
- If you provide an answer based on the context, make sure it's actually supported by the provided text
- Do not make assumptions or provide general knowledge not found in the context

Context from meeting documents:
{context}

Question: {question}

Answer:"""
        )
    
    def initialize(self, data_source: str) -> bool:
        """Initialize the RAG system with documents from data source."""
        try:
            self._logger.info(f"Initializing RAG system from: {data_source}")
            
            # Process documents
            documents = self._document_pipeline.process_directory(data_source)
            
            if not documents:
                self._logger.warning("No documents found to process")
                return False
            
            # Create or load vector store
            self._vector_store = self._vector_store_manager.get_or_create_store(documents)
            
            # Create retriever - use basic similarity for more reliable results
            # The score threshold approach was too restrictive
            self._retriever = self._vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self._config.search_k}
            )
            
            self._logger.info("Using similarity-based retrieval for reliable source citations")
            
            self._logger.info(f"RAG system initialized with {len(documents)} document chunks")
            return True
            
        except Exception as e:
            self._logger.error("Failed to initialize RAG system", e)
            return False
    
    def _create_qa_chain(self):
        """Create the question-answering chain."""
        if not self._retriever:
            raise ValueError("Vector store not initialized. Call initialize first.")
        
        self._qa_chain = RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type="stuff",
            retriever=self._retriever,
            chain_type_kwargs={"prompt": self._prompt_template},
            return_source_documents=True
        )
        
        self._logger.debug("QA chain created successfully")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG pipeline with a question."""
        if not self._qa_chain:
            self._create_qa_chain()

        try:
            self._logger.debug(f"Processing query: {question}")

            # First get documents with similarity scores directly from vector store
            docs_with_scores = self._vector_store_manager._vector_store.search_with_scores(question, k=self._config.search_k)

            # Get response from QA chain using regular retrieval
            result = self._qa_chain({"query": question})
            source_docs = result.get("source_documents", [])

            self._logger.debug(f"Search returned {len(source_docs)} source documents")

            # Log source document details for debugging
            if source_docs:
                for i, doc in enumerate(source_docs):
                    self._logger.debug(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')} - {len(doc.page_content)} chars")
            else:
                self._logger.warning("No source documents returned - this will result in no citations")

            # Format response with smart citation filtering and similarity scores
            response = self._format_response_with_validation(result, source_docs, question, docs_with_scores)
            self._logger.info(f"Query processed successfully with {len(response['sources'])} relevant sources")

            return response

        except Exception as e:
            self._logger.error(f"Query processing failed: {question}", e)
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "source_documents": []
            }
    
    
    def _format_response_with_validation(self, result: Dict, source_docs: list, question: str, docs_with_scores: list = None) -> Dict[str, Any]:
        """Format response with smart citation validation."""
        answer = result["result"]

        # Check if the answer indicates no relevant information was found
        no_info_indicators = [
            "don't have information",
            "no mention",
            "not mentioned",
            "no information",
            "cannot determine",
            "not possible to determine",
            "based on the information provided",
            "there is no mention",
            "i don't have information",
            "no information about",
            "not mentioned in"
        ]

        answer_lower = answer.lower()
        has_no_info = any(indicator in answer_lower for indicator in no_info_indicators)

        if has_no_info:
            self._logger.debug("Answer indicates no relevant information found - filtering out citations")
            return {
                "answer": answer,
                "sources": [],
                "source_documents": []
            }

        # Additional check: see if answer actually references content from the sources
        if source_docs:
            content_overlap = self._check_content_relevance(answer, source_docs, question)
            if not content_overlap:
                self._logger.debug("No content overlap detected - filtering out citations")
                return {
                    "answer": answer,
                    "sources": [],
                    "source_documents": []
                }

        # If answer seems to reference the context, include sources
        return self._format_response(result, source_docs, docs_with_scores)
    
    def _check_content_relevance(self, answer: str, source_docs: list, question: str) -> bool:
        """Check if the answer actually references content from the source documents."""
        if not source_docs:
            return False
        
        # Extract key terms from the question (simple approach)
        question_words = set(question.lower().split())
        # Remove common words
        stop_words = {'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during'}
        question_keywords = question_words - stop_words
        
        # Check if any source document contains keywords from the question AND the answer references those docs
        for doc in source_docs:
            doc_words = set(doc.page_content.lower().split())
            # If document contains question keywords AND answer is longer than a simple "no info" response
            if question_keywords.intersection(doc_words) and len(answer) > 50:
                return True
        
        return False
    
    def _format_response(self, result: Dict, source_docs: list, docs_with_scores: list = None) -> Dict[str, Any]:
        """Format the response with detailed source information including similarity scores."""
        response = {
            "answer": result["result"],
            "sources": [],
            "source_documents": source_docs
        }

        # Create a mapping of document content to similarity scores if available
        score_mapping = {}
        if docs_with_scores:
            for doc, score in docs_with_scores:
                # Use content beginning as key to match with source_docs
                content_key = doc.page_content[:100]
                score_mapping[content_key] = score

        # Extract and deduplicate source information
        seen_sources = set()

        for i, doc in enumerate(source_docs):
            content = doc.page_content.strip()
            filename = doc.metadata.get("source", "Unknown")

            # Create a key to avoid duplicate sources from same file
            source_key = f"{filename}_{content[:50]}"

            if source_key not in seen_sources:
                seen_sources.add(source_key)

                # Create more detailed preview
                preview = content[:200] + "..." if len(content) > 200 else content

                # Find similarity score for this document
                similarity_score = None
                content_key = content[:100]
                if content_key in score_mapping:
                    raw_score = score_mapping[content_key]
                    if raw_score is not None:
                        # Convert distance to similarity percentage (Chroma uses distance, lower = more similar)
                        # Typical distance range is 0-2, so we convert to percentage
                        similarity_percentage = max(0, (1 - raw_score / 2) * 100)
                        similarity_score = round(similarity_percentage, 1)

                source_info = {
                    "filename": filename,
                    "content_preview": preview,
                    "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                    "relevance_rank": i + 1,
                    "similarity_score": similarity_score
                }
                response["sources"].append(source_info)

        # Sort sources by similarity score if available (highest first)
        if any(source.get("similarity_score") is not None for source in response["sources"]):
            response["sources"].sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            # Update relevance rank after sorting
            for i, source in enumerate(response["sources"]):
                source["relevance_rank"] = i + 1

        # Log source information for debugging
        self._logger.debug(f"Formatted {len(response['sources'])} unique sources with similarity scores")

        return response

