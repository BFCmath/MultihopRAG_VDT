from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import logging

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MultiHopRagStructure(ABC):
    """Abstract base class for multi-hop RAG structures"""
    
    def __init__(self, 
                 name: str,
                 vector_store=None,
                 reranker=None, 
                 search_engine=None,
                 config=None):
        
        self.name = name
        self.vector_store = vector_store
        self.reranker = reranker
        self.search_engine = search_engine
        self.config = config
        
        # Performance tracking
        self.last_query_time = 0
        self.total_queries = 0
        self.total_documents_retrieved = 0
        
        # Set up logger for this instance
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{name}")
    
    @abstractmethod
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the multi-hop query process.
        
        Args:
            question: The input question to answer
            **kwargs: Additional parameters specific to the structure
            
        Returns:
            Dict containing:
            - question: Original question
            - answer: Generated answer
            - reasoning_chain: List of reasoning steps
            - retrieved_documents: All documents used
            - metadata: Additional information about the process
        """
        pass
    
    def initialize(self) -> bool:
        """Initialize the structure with required components"""
        try:
            # Validate required components
            if not self.vector_store:
                self.logger.warning(f"{self.name} initialized without vector store")
            
            if not self.config:
                self.logger.warning(f"{self.name} initialized without configuration")
                
            self.logger.info(f"Initialized {self.name} structure")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing {self.name}: {e}")
            return False
    
    def get_structure_info(self) -> Dict[str, Any]:
        """Get information about this structure"""
        return {
            'name': self.name,
            'has_vector_store': self.vector_store is not None,
            'has_reranker': self.reranker is not None,
            'has_search_engine': self.search_engine is not None,
            'has_config': self.config is not None,
            'total_queries': self.total_queries,
            'total_documents_retrieved': self.total_documents_retrieved,
            'last_query_time': self.last_query_time
        }
    
    def validate_components(self) -> Dict[str, bool]:
        """Validate that all required components are available"""
        return {
            'vector_store': self.vector_store is not None,
            'reranker': self.reranker is not None,
            'search_engine': self.search_engine is not None,
            'config': self.config is not None
        }
    
    def _format_documents_for_display(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents for display in prompts"""
        if not documents:
            return "No documents available."
        
        formatted_docs = []
        for i, doc in enumerate(documents):
            # Assuming 'doc' can be a Langchain Document object or a dict
            page_content = getattr(doc, 'page_content', doc.get('content', '')) 
            doc_metadata = getattr(doc, 'metadata', doc.get('metadata', {}))

            content_to_display = page_content[:500] + "..." if len(page_content) > 500 else page_content
            doc_str = f"Document {i+1}:\n{content_to_display}"
            
            if doc_metadata:
                title = doc_metadata.get('title', '').strip()
                # Check if the title is the generic one assigned by DocumentProcessor
                # The passage_id in metadata is 0-indexed.
                passage_id_in_metadata = doc_metadata.get('passage_id', -1) # default to -1 if not found
                is_default_title = False
                if passage_id_in_metadata != -1:
                    is_default_title = title == f"Passage {passage_id_in_metadata + 1}"

                if title and not is_default_title:
                    display_source = title
                else:
                    # Fallback to the file path if title is default/empty, or if passage_id was missing for the check
                    display_source = doc_metadata.get('source', 'Unknown Source')
                
                doc_str += f"\nSource: {display_source}"
            
            formatted_docs.append(doc_str)
        
        return "\n\n".join(formatted_docs)
    
    def _update_metrics(self, num_documents: int, query_time: float):
        """Update performance metrics"""
        self.total_queries += 1
        self.total_documents_retrieved += num_documents
        self.last_query_time = query_time
    
    def _log_step(self, step_name: str, details: Dict[str, Any]):
        """Log a step in the multi-hop process"""
        self.logger.info(f"[{self.name}] {step_name}: {details}")
    
    def reset_metrics(self):
        """Reset performance tracking metrics"""
        self.total_queries = 0
        self.total_documents_retrieved = 0
        self.last_query_time = 0
        self.logger.info(f"Reset metrics for {self.name}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if self.total_queries == 0:
            return {'message': 'No queries executed yet'}
        
        avg_docs_per_query = self.total_documents_retrieved / self.total_queries
        
        return {
            'structure_name': self.name,
            'total_queries': self.total_queries,
            'total_documents_retrieved': self.total_documents_retrieved,
            'average_documents_per_query': avg_docs_per_query,
            'last_query_time_seconds': self.last_query_time
        } 