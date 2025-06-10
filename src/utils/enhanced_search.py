#!/usr/bin/env python3
"""
Enhanced Search Module
Combines semantic, lexical, and fuzzy search capabilities for improved retrieval
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from difflib import SequenceMatcher
from collections import defaultdict
import unicodedata

class EnhancedSearchEngine:
    """
    Enhanced search engine that combines semantic, lexical, and fuzzy search
    """
    
    def __init__(self, vector_store=None, corpus_documents: List[Dict] = None):
        self.vector_store = vector_store
        self.corpus_documents = corpus_documents or []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Build indexes for faster searching
        self._build_lexical_index()
        
    def _build_lexical_index(self):
        """Build inverted index for lexical search"""
        self.lexical_index = defaultdict(set)  # word -> set of document indices
        self.document_texts = []  # Store normalized document texts
        
        for i, doc in enumerate(self.corpus_documents):
            content = doc.get('content', '')
            self.document_texts.append(content)
            
            # Normalize and tokenize
            normalized_content = self._normalize_text(content)
            words = self._tokenize(normalized_content)
            
            for word in words:
                self.lexical_index[word].add(i)
        
        self.logger.info(f"Built lexical index with {len(self.lexical_index)} unique terms for {len(self.corpus_documents)} documents")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent searching"""
        # Convert to lowercase
        text = text.lower()
        # Remove accents and special characters
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, preserving important patterns"""
        # Split on whitespace and punctuation, but preserve some patterns
        # Keep numbers, dates, and hyphenated words intact
        pattern = r'\b\w+(?:-\w+)*\b|\d+(?:\.\d+)?|\w+'
        tokens = re.findall(pattern, text)
        return [token for token in tokens if len(token) > 1]  # Filter very short tokens
    
    def semantic_search(self, query: str, top_k: int = 10, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Perform semantic search using vector store"""
        if not self.vector_store:
            self.logger.warning("No vector store available for semantic search")
            return []
        
        try:
            results = self.vector_store.similarity_search(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            # Add search method metadata
            for result in results:
                result['search_method'] = 'semantic'
                result['search_query'] = query
            
            self.logger.debug(f"Semantic search found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def lexical_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform lexical/keyword search"""
        if not self.corpus_documents:
            self.logger.warning("No corpus documents available for lexical search")
            return []
        
        try:
            # Normalize query and extract keywords
            normalized_query = self._normalize_text(query)
            query_words = self._tokenize(normalized_query)
            
            if not query_words:
                return []
            
            # Score documents based on keyword matches
            doc_scores = defaultdict(float)
            
            for word in query_words:
                # Exact matches
                if word in self.lexical_index:
                    for doc_idx in self.lexical_index[word]:
                        doc_scores[doc_idx] += 1.0
                
                # Partial matches for longer words
                if len(word) > 4:
                    for indexed_word in self.lexical_index:
                        if word in indexed_word or indexed_word in word:
                            similarity = SequenceMatcher(None, word, indexed_word).ratio()
                            if similarity > 0.8:
                                for doc_idx in self.lexical_index[indexed_word]:
                                    doc_scores[doc_idx] += similarity * 0.5
            
            # Convert to results format
            results = []
            for doc_idx, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                if score > 0:
                    doc = self.corpus_documents[doc_idx].copy()
                    doc['similarity_score'] = score
                    doc['search_method'] = 'lexical'
                    doc['search_query'] = query
                    results.append(doc)
            
            self.logger.debug(f"Lexical search found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"Lexical search failed: {e}")
            return []
    
    def fuzzy_search(self, query: str, top_k: int = 10, min_similarity: float = 0.6) -> List[Dict[str, Any]]:
        """Perform fuzzy search for approximate matching"""
        if not self.corpus_documents:
            self.logger.warning("No corpus documents available for fuzzy search")
            return []
        
        try:
            normalized_query = self._normalize_text(query)
            results = []
            
            for i, doc in enumerate(self.corpus_documents):
                content = self._normalize_text(doc.get('content', ''))
                
                # Calculate similarity using multiple methods
                similarity_scores = []
                
                # Overall content similarity
                overall_similarity = SequenceMatcher(None, normalized_query, content).ratio()
                similarity_scores.append(overall_similarity)
                
                # Sliding window similarity for queries
                if len(normalized_query) > 20:
                    window_size = len(normalized_query)
                    max_window_similarity = 0
                    
                    for j in range(max(0, len(content) - window_size + 1)):
                        window = content[j:j + window_size]
                        window_similarity = SequenceMatcher(None, normalized_query, window).ratio()
                        max_window_similarity = max(max_window_similarity, window_similarity)
                    
                    similarity_scores.append(max_window_similarity)
                
                # Phrase-level similarity
                query_phrases = normalized_query.split()
                if len(query_phrases) > 1:
                    for phrase in query_phrases:
                        if len(phrase) > 3:
                            if phrase in content:
                                similarity_scores.append(1.0)
                            else:
                                # Find best matching substring
                                best_phrase_similarity = 0
                                for k in range(len(content) - len(phrase) + 1):
                                    substring = content[k:k + len(phrase)]
                                    phrase_similarity = SequenceMatcher(None, phrase, substring).ratio()
                                    best_phrase_similarity = max(best_phrase_similarity, phrase_similarity)
                                similarity_scores.append(best_phrase_similarity)
                
                # Take the maximum similarity score
                if similarity_scores:
                    max_similarity = max(similarity_scores)
                    
                    if max_similarity >= min_similarity:
                        doc_result = doc.copy()
                        doc_result['similarity_score'] = max_similarity
                        doc_result['search_method'] = 'fuzzy'
                        doc_result['search_query'] = query
                        results.append(doc_result)
            
            # Sort by similarity and return top_k
            results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)[:top_k]
            
            self.logger.debug(f"Fuzzy search found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"Fuzzy search failed: {e}")
            return []
    
    def combined_search(self, 
                       query: str, 
                       search_methods: List[str] = None,
                       top_k: int = 10,
                       semantic_weight: float = 0.6,
                       lexical_weight: float = 0.3,
                       fuzzy_weight: float = 0.1) -> List[Dict[str, Any]]:
        """
        Perform combined search using multiple methods and merge results
        
        Args:
            query: Search query
            search_methods: List of methods to use ['semantic', 'lexical', 'fuzzy']
            top_k: Number of results to return
            semantic_weight: Weight for semantic search results
            lexical_weight: Weight for lexical search results  
            fuzzy_weight: Weight for fuzzy search results
        """
        if search_methods is None:
            search_methods = ['semantic', 'lexical', 'fuzzy']
        
        all_results = []
        
        # Collect results from each search method
        if 'semantic' in search_methods:
            semantic_results = self.semantic_search(query, top_k=top_k * 2)
            for result in semantic_results:
                result['weighted_score'] = result.get('similarity_score', 0) * semantic_weight
            all_results.extend(semantic_results)
        
        if 'lexical' in search_methods:
            lexical_results = self.lexical_search(query, top_k=top_k * 2)
            for result in lexical_results:
                result['weighted_score'] = result.get('similarity_score', 0) * lexical_weight
            all_results.extend(lexical_results)
        
        if 'fuzzy' in search_methods:
            fuzzy_results = self.fuzzy_search(query, top_k=top_k * 2)
            for result in fuzzy_results:
                result['weighted_score'] = result.get('similarity_score', 0) * fuzzy_weight
            all_results.extend(fuzzy_results)
        
        # Merge and deduplicate results
        merged_results = self._merge_results(all_results, top_k)
        
        self.logger.info(f"Combined search using {search_methods} found {len(merged_results)} merged results")
        return merged_results
    
    def _merge_results(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from different search methods"""
        # Group by document content hash for deduplication
        content_groups = defaultdict(list)
        
        for result in results:
            content = result.get('content', '')
            content_hash = hash(content.strip())
            content_groups[content_hash].append(result)
        
        # Merge grouped results
        merged_results = []
        for content_hash, group in content_groups.items():
            if not group:
                continue
            
            # Take the result with highest weighted score
            best_result = max(group, key=lambda x: x.get('weighted_score', 0))
            
            # Combine search methods and scores
            search_methods = list(set(r.get('search_method', '') for r in group))
            total_weighted_score = sum(r.get('weighted_score', 0) for r in group)
            
            merged_result = best_result.copy()
            merged_result['search_methods'] = search_methods
            merged_result['combined_score'] = total_weighted_score
            merged_result['method_count'] = len(search_methods)
            
            merged_results.append(merged_result)
        
        # Sort by combined score and method diversity
        merged_results.sort(key=lambda x: (x.get('combined_score', 0), x.get('method_count', 0)), reverse=True)
        
        return merged_results[:top_k]
    
    def analyze_query_for_search_methods(self, query: str) -> Dict[str, Any]:
        """Analyze query to suggest optimal search methods"""
        query_lower = query.lower()
        
        analysis = {
            'suggested_methods': [],
            'reasoning': [],
            'has_specific_terms': False,
            'has_numeric_data': False,
            'has_names': False,
            'is_conceptual': False
        }
        
        # Check for specific indicators
        
        # Numeric data (dates, numbers, etc.)
        if re.search(r'\b\d{4}\b|\b\d+\b|january|february|march|april|may|june|july|august|september|october|november|december', query_lower):
            analysis['has_numeric_data'] = True
            analysis['suggested_methods'].append('lexical')
            analysis['reasoning'].append('Query contains numeric/date information - lexical search recommended')
        
        # Proper names (capitalized words)
        if re.search(r'\b[A-Z][a-z]+\b', query):
            analysis['has_names'] = True
            analysis['suggested_methods'].append('lexical')
            analysis['suggested_methods'].append('fuzzy')
            analysis['reasoning'].append('Query contains proper names - lexical and fuzzy search recommended')
        
        # Specific technical terms or exact phrases
        if '"' in query or len(query.split()) <= 3:
            analysis['has_specific_terms'] = True
            analysis['suggested_methods'].append('lexical')
            analysis['reasoning'].append('Query has specific terms - lexical search recommended')
        
        # Conceptual/semantic queries
        if any(word in query_lower for word in ['how', 'why', 'what', 'explain', 'describe', 'concept', 'theory']):
            analysis['is_conceptual'] = True
            analysis['suggested_methods'].append('semantic')
            analysis['reasoning'].append('Conceptual query - semantic search recommended')
        
        # Default to semantic if no specific indicators
        if not analysis['suggested_methods']:
            analysis['suggested_methods'] = ['semantic']
            analysis['reasoning'].append('General query - semantic search recommended')
        
        # Always include semantic as fallback
        if 'semantic' not in analysis['suggested_methods']:
            analysis['suggested_methods'].append('semantic')
        
        # Remove duplicates while preserving order
        analysis['suggested_methods'] = list(dict.fromkeys(analysis['suggested_methods']))
        
        return analysis 