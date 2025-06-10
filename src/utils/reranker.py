import os
# import google.generativeai as genai # No longer directly used for LLM calls
from typing import List, Dict, Any, Optional, Tuple
import re

from langchain_core.language_models import BaseLanguageModel # Import Langchain LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Reranker:
    """General reranker API for multi-hop RAG systems, using a Langchain LLM"""
    
    def __init__(self, 
                 llm: BaseLanguageModel, # Expect a Langchain LLM object
                 min_relevance_score: float = 0.5):
        
        self.llm = llm # Store the Langchain LLM object
        self.min_relevance_score = min_relevance_score
        self.output_parser = StrOutputParser()
        
        # Google API key configuration is now handled by the LLM object if it's Google-based
        # No need for direct genai.configure or self.model initialization here
    
    def rerank_documents(self, 
                        query: str, 
                        documents: List[Dict[str, Any]], 
                        top_k: Optional[int] = None,
                        custom_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to query"""
        if not documents:
            return []
        
        scored_docs = []
        
        for doc in documents:
            score = self.score_document(query, doc, custom_prompt)
            
            # Add score to document
            scored_doc = doc.copy()
            scored_doc['relevance_score'] = score
            scored_docs.append(scored_doc)
        
        # Sort by relevance score (descending)
        scored_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Filter by minimum relevance score
        filtered_docs = [
            doc for doc in scored_docs 
            if doc.get('relevance_score', 0) >= self.min_relevance_score
        ]
        
        # Return top k documents if specified
        if top_k is not None:
            return filtered_docs[:top_k]
        
        return filtered_docs
    
    def score_document(self, 
                      query: str, 
                      document: Dict[str, Any],
                      custom_prompt: Optional[str] = None) -> float:
        """Score a single document's relevance to the query using Langchain LLM"""
        try:
            content = document.get('content', '')
            metadata = document.get('metadata', {})
            doc_text = self._prepare_document_text(content, metadata)
            
            prompt_text = custom_prompt or self._get_default_scoring_prompt(query, doc_text)
            
            # Use a Langchain prompt template and chain
            prompt_template = PromptTemplate.from_template(prompt_text)
            chain = prompt_template | self.llm | self.output_parser
            
            response_text = chain.invoke({"query": query, "document": doc_text})
            score = self._extract_score(response_text)
            
            return score
            
        except Exception as e:
            print(f"Error scoring document with Langchain: {e}")
            return 0.0
    
    def _prepare_document_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """Prepare document text for scoring"""
        # Truncate content if too long
        max_content_length = 1000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        # Add relevant metadata
        metadata_str = ""
        if metadata:
            relevant_keys = ['title', 'source', 'date', 'author']
            metadata_parts = []
            for key in relevant_keys:
                if key in metadata:
                    metadata_parts.append(f"{key}: {metadata[key]}")
            if metadata_parts:
                metadata_str = f"[{', '.join(metadata_parts)}]\n"
        
        return f"{metadata_str}{content}"
    
    def _get_default_scoring_prompt(self, query: str, document: str) -> str:
        """Get default document scoring prompt"""
        return f"""Rate the relevance of this document to the given question on a scale of 0.0 to 1.0.

Question: {query}

Document:
{document}

Consider:
- Direct relevance to the question
- Quality and accuracy of information
- Specificity and detail level
- Credibility of the source

Provide only a numerical score between 0.0 and 1.0 (e.g., 0.8):"""
    
    def _extract_score(self, response_text: str) -> float:
        """Extract numerical score from LLM response"""
        try:
            # Look for decimal numbers between 0 and 1
            import re
            
            # First try to find a decimal like 0.8, 0.75, etc.
            decimal_pattern = r'\b0\.\d+\b|\b1\.0+\b'
            decimal_matches = re.findall(decimal_pattern, response_text)
            
            if decimal_matches:
                return float(decimal_matches[0])
            
            # Try to find a number out of 10 and convert
            out_of_ten_pattern = r'\b([0-9](?:\.[0-9])?)\s*(?:/|out of)\s*10\b'
            ten_matches = re.findall(out_of_ten_pattern, response_text, re.IGNORECASE)
            
            if ten_matches:
                return float(ten_matches[0]) / 10.0
            
            # Try to find any number and normalize
            number_pattern = r'\b\d+(?:\.\d+)?\b'
            number_matches = re.findall(number_pattern, response_text)
            
            if number_matches:
                score = float(number_matches[0])
                # Normalize based on likely scale
                if score > 10:
                    return min(score / 100.0, 1.0)  # Assume out of 100
                elif score > 1:
                    return min(score / 10.0, 1.0)   # Assume out of 10
                else:
                    return min(score, 1.0)          # Assume out of 1
            
            # Default to medium relevance if can't parse
            return 0.5
            
        except Exception as e:
            print(f"Error extracting score: {e}")
            return 0.5
    
    def filter_relevant_documents(self, 
                                 query: str, 
                                 documents: List[Dict[str, Any]], 
                                 threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Filter documents by relevance threshold"""
        threshold = threshold or self.min_relevance_score
        
        relevant_docs = []
        for doc in documents:
            score = self.score_document(query, doc)
            if score >= threshold:
                doc_copy = doc.copy()
                doc_copy['relevance_score'] = score
                relevant_docs.append(doc_copy)
        
        return relevant_docs
    
    def compare_documents(self, 
                         query: str, 
                         doc1: Dict[str, Any], 
                         doc2: Dict[str, Any],
                         custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Compare two documents and determine which is more relevant"""
        try:
            doc1_text = self._prepare_document_text(
                doc1.get('content', ''), 
                doc1.get('metadata', {})
            )
            doc2_text = self._prepare_document_text(
                doc2.get('content', ''), 
                doc2.get('metadata', {})
            )
            
            if custom_prompt:
                prompt = custom_prompt.format(
                    query=query, 
                    document1=doc1_text, 
                    document2=doc2_text
                )
            else:
                prompt = f"""Which document is more relevant to the question? Respond with "1" or "2".

Question: {query}

Document 1:
{doc1_text}

Document 2:
{doc2_text}

More relevant document (1 or 2):"""
            
            # Use a Langchain prompt template and chain for comparison
            prompt_template = PromptTemplate.from_template(prompt)
            chain = prompt_template | self.llm | self.output_parser
            response_text = chain.invoke({
                "query": query, 
                "document1": doc1_text, 
                "document2": doc2_text
            })
            
            if "1" in response_text:
                return {"winner": doc1, "loser": doc2, "choice": 1}
            elif "2" in response_text:
                return {"winner": doc2, "loser": doc1, "choice": 2}
            else:
                return {"winner": doc1, "loser": doc2, "choice": 1}  # Default to first
                
        except Exception as e:
            print(f"Error comparing documents: {e}")
            return {"winner": doc1, "loser": doc2, "choice": 1}
    
    def batch_score_documents(self, 
                             query: str, 
                             documents: List[Dict[str, Any]],
                             batch_size: int = 5) -> List[Dict[str, Any]]:
        """Score documents in batches for efficiency"""
        scored_docs = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            for doc in batch_docs:
                score = self.score_document(query, doc)
                doc_copy = doc.copy()
                doc_copy['relevance_score'] = score
                scored_docs.append(doc_copy)
        
        return scored_docs
    
    def get_top_documents(self, 
                         query: str, 
                         documents: List[Dict[str, Any]], 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top k most relevant documents"""
        scored_docs = self.batch_score_documents(query, documents)
        scored_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return scored_docs[:top_k]
    
    def explain_relevance(self, 
                         query: str, 
                         document: Dict[str, Any]) -> str:
        """Get explanation for why a document is relevant using Langchain LLM"""
        try:
            content = document.get('content', '')
            metadata = document.get('metadata', {})
            doc_text = self._prepare_document_text(content, metadata)
            
            prompt_text = f"""Explain why this document is relevant (or not relevant) to the given question. Be specific about which parts of the document address the question.

Question: {query}

Document:
{doc_text}

Explanation:"""
            
            # Use a Langchain prompt template and chain for explanation
            prompt_template = PromptTemplate.from_template(prompt_text)
            chain = prompt_template | self.llm | self.output_parser
            explanation = chain.invoke({"query": query, "document": doc_text})
            
            return explanation.strip()
            
        except Exception as e:
            print(f"Error explaining relevance with Langchain: {e}")
            return "Unable to generate explanation." 