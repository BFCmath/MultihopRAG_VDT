import os
import time
import hashlib
import logging
import re # Add re for parsing structured outputs
import json # Add json for report generation
from datetime import datetime # Add datetime for timestamps
from collections import defaultdict # Add defaultdict import
# import google.generativeai as genai # No longer directly needed for LLM calls
from typing import Dict, Any, List, Optional, Tuple # Add Tuple to imports

from langchain_core.language_models import BaseLanguageModel # For LLM type hint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from multihop_structure import MultiHopRagStructure
from prompts import PromptTemplates
from utils.parser import IRCOTResponseParser

class IRCOTStructure(MultiHopRagStructure):
    """
    IR-COT (Interleaved Retrieval Chain-of-Thought) implementation
    Following the updated specification with fact-based reasoning
    """
    
    def __init__(self, 
                 vector_store=None,
                 reranker=None,
                 search_engine=None,
                 config=None,
                 llm: Optional[BaseLanguageModel] = None, # Allow passing an LLM object
                 model_name: Optional[str] = None): # model_name is backup if llm not given
        
        super().__init__(
            name="IR-COT",
            vector_store=vector_store,
            reranker=reranker,
            search_engine=search_engine,
            config=config
        )
        # Ensure logger is initialized (it should be by superclass)
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.name}")

        self.max_iterations = getattr(config, 'MAX_ITERATIONS', 5)
        self.top_k = getattr(config, 'TOP_K_DOCUMENTS', 10)
        self.expansion_factor = getattr(config, 'EXPANSION_FACTOR', 5)
        self.no_repeat_retrieve_enable = getattr(config, 'NO_REPEAT_RETRIEVE_ENABLE', True)
        self.remove_sources_enable = getattr(config, 'REMOVE_SOURCES_ENABLE', True)
        self.output_parser = StrOutputParser()
        
        # Initialize metrics attributes
        self.total_queries = 0
        self.total_documents = 0
        self.total_query_time = 0.0
        self.avg_documents_per_query = 0.0
        self.avg_query_time = 0.0

        # Initialize enhanced search and rerank settings
        self.enhanced_search = getattr(vector_store, 'enhanced_search', None)
        # If vector_store doesn't have enhanced_search, try to create one
        if not self.enhanced_search and vector_store:
            try:
                from utils.enhanced_search import EnhancedSearchEngine
                # Try to get corpus documents from vector store if available
                corpus_docs = getattr(vector_store, 'documents', [])
                self.enhanced_search = EnhancedSearchEngine(
                    vector_store=vector_store,
                    corpus_documents=corpus_docs
                )
                self.logger.info("Created EnhancedSearchEngine for IR-COT")
            except Exception as e:
                self.logger.warning(f"Could not create EnhancedSearchEngine: {e}")
                self.enhanced_search = None
        
        self.rerank_enable = getattr(config, 'RERANK_ENABLE', False)
        self.enhanced_search_enable = getattr(config, 'ENHANCED_SEARCH_ENABLE', True)
        
        # Initialize response parser
        self.parser = IRCOTResponseParser()

        # Initialize separate LLMs for different components
        self.model_name = model_name or "gemini-1.5-flash-latest"
        
        if llm:
            # If a single LLM is provided, use it for all components
            self.llm_query_generator = llm
            self.llm_reasoner = llm
            self.llm_final_generator = llm
            self.model_name = getattr(llm, 'model_name', self.model_name)
        else:
            # Create separate LLMs for different components using different API keys
            self.llm_query_generator = self._create_llm_for_component('query_generator')
            self.llm_reasoner = self._create_llm_for_component('reasoner')
            self.llm_final_generator = self._create_llm_for_component('final_generator')

    def _create_llm_for_component(self, component: str) -> BaseLanguageModel:
        """Create an LLM instance for a specific component using its designated API key"""
        try:
            api_key = self.config.get_api_key_for_component(component)
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            llm = ChatGoogleGenerativeAI(
                model=self.model_name, 
                google_api_key=str(api_key)
            )
            
            self.logger.info(f"Created LLM for {component} using API key: {api_key[:10]}...")
            return llm
            
        except Exception as e:
            self.logger.error(f"Failed to create LLM for {component}: {e}")
            # Fallback to general API key
            fallback_key = getattr(self.config, 'GOOGLE_API_KEY_1', None)
            if fallback_key:
                self.logger.warning(f"Falling back to general API key for {component}")
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(model=self.model_name, google_api_key=str(fallback_key))
            else:
                raise ValueError(f"No API key available for {component}")
                
    def query(self, question: str, generate_individual_report: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Execute IR-COT query process following the updated specification:
        0. Preprocessing: Remove sources from the original question
        1. Initial query generation and retrieval 
        2. Iterative interleaving of fact extraction and query generation
        3. Final answer generation
        4. Generate detailed JSON report (optional)
        
        Args:
            question: The question to answer
            generate_individual_report: Whether to generate individual JSON report for this query
            **kwargs: Additional keyword arguments
        """
        start_time = time.time()
        
        try:
            # Step 0: Preprocessing - Remove sources from the original question (if enabled)
            original_question_with_sources = question
            
            if self.remove_sources_enable:
                self.logger.info("Preprocessing Step: Removing sources from original question")
                cleaned_question = self._remove_sources_from_question(original_question_with_sources)
                # Use the cleaned question for the rest of the process
                question = cleaned_question
                self.logger.info(f"Original question: {original_question_with_sources}")
                self.logger.info(f"Cleaned question: {cleaned_question}")
            else:
                self.logger.info("Source removal disabled - using original question")
                cleaned_question = original_question_with_sources
            
            # Track established facts and detailed iteration information for reporting
            established_facts = []
            iteration_details = []
            fact_list_for_report = []  # Detailed fact list for JSON report
            all_documents = []
            retrieved_document_hashes = set() if self.no_repeat_retrieve_enable else None  # Track hashes only if deduplication is enabled
            
            # Step 1: Generate initial search query
            self.logger.info("Initial Query Generation Step")
            initial_search_query, initial_query_reasoning, initial_search_methods = self._generate_search_query_with_reasoning(
                question=question,
                current_facts=established_facts
            )
            
            if not initial_search_query or initial_search_query.strip().upper() == "NO_QUERY_NEEDED":
                self.logger.info(f"No initial search query generated. Query: '{initial_search_query}'")
                return self._create_error_result(question, "No initial search query could be generated", start_time)
            
            # Step 2: Initial retrieval
            self.logger.info("Initial Retrieval Step")
            initial_docs = self._retrieve_documents(initial_search_query, initial_search_methods, retrieved_document_hashes)
            all_documents = initial_docs.copy()
            
            # Step 3: Iterative Interleaving Loop
            current_iteration_for_termination_reason = 0
            no_new_docs_for_termination_reason = False
            no_query_generated_for_termination_reason = False
            current_search_query = initial_search_query
            current_docs = initial_docs
            current_query_reasoning = initial_query_reasoning
            current_search_methods = initial_search_methods

            for iteration in range(self.max_iterations):
                current_iteration_for_termination_reason = iteration
                self.logger.info(f"Iteration {iteration + 1}: Fact Extraction Step")
                
                # Step (i): Extract New Fact from Current Documents
                new_fact, fact_reasoning = self._extract_new_fact_with_reasoning(
                    question=question,
                    current_facts=established_facts,
                    current_search_query=current_search_query,
                    retrieved_docs=current_docs
                )

                if not new_fact:
                    self.logger.info("Early termination: No new fact extracted")
                    break
                
                established_facts.append(new_fact)
                
                # Add detailed information to fact list for report
                fact_entry = {
                    "reason_query": current_query_reasoning,
                    "query_to_search": current_search_query,
                    "search_methods": current_search_methods,
                    "keywords": getattr(self, '_current_keywords', 'NO_NEED'),
                    "documents": self._format_documents_for_report(current_docs),
                    "reason_fact": fact_reasoning,
                    "new_fact": new_fact
                }
                fact_list_for_report.append(fact_entry)
                
                # Check if max iterations reached after fact extraction
                if iteration >= self.max_iterations - 1:
                    self.logger.info(f"Termination: Max iterations ({self.max_iterations}) reached after fact extraction")
                    break
                
                # Step (ii): Generate Next Search Query
                self.logger.info(f"Iteration {iteration + 1}: Query Generation Step")
                next_search_query, next_query_reasoning, next_search_methods = self._generate_search_query_with_reasoning(
                    question=question,
                    current_facts=established_facts
                )

                if not next_search_query or next_search_query.strip().upper() == "NO_QUERY_NEEDED":
                    self.logger.info(f"Termination: No new search query generated or 'NO_QUERY_NEEDED'. Query: '{next_search_query}'")
                    no_query_generated_for_termination_reason = True
                    break
                
                # Step (iii): Retrieve Documents for Next Query
                self.logger.info(f"Iteration {iteration + 1}: Retrieval Step")
                new_docs = self._retrieve_documents(next_search_query, next_search_methods, retrieved_document_hashes)
                no_new_docs_for_termination_reason = not new_docs
                
                # Add new documents to cumulated docs (avoid duplicates)
                before_count = len(all_documents)
                all_documents = self._accumulate_documents(all_documents, new_docs)
                new_count = len(all_documents) - before_count
                
                iteration_details.append({
                    'iteration': iteration + 1,
                    'extracted_fact': new_fact,
                    'search_query': next_search_query,
                    'search_methods': next_search_methods,
                    'keywords': getattr(self, '_current_keywords', 'NO_NEED'),
                    'new_documents_found': len(new_docs),
                    'new_documents_added': new_count,
                    'total_documents': len(all_documents)
                })
                
                # Update for next iteration
                current_search_query = next_search_query
                current_docs = new_docs
                current_query_reasoning = next_query_reasoning
                current_search_methods = next_search_methods
                
                if not new_docs: 
                    self.logger.info("Early termination: No new documents found in this iteration") 
                    break
            
            # Step 4: Final Answer Generation
            self.logger.info("Final Answer Generation Step")
            
            # Determine termination reason for prompt selection
            termination_reason = self._get_termination_reason(
                established_facts, 
                current_iteration_for_termination_reason, 
                no_new_docs_for_termination_reason,
                no_query_generated_for_termination_reason
            )
            
            final_answer, answer_reasoning, confidence = self._generate_final_answer_with_reasoning(
                question=question,
                established_facts=established_facts,
                termination_reason=termination_reason
            )
            
            # Calculate metrics
            query_time = time.time() - start_time
            self._update_metrics(len(all_documents), query_time)
            
            # Log the final answer itself to reasoning_chain.txt
            self.logger.info(f"Final Answer: {final_answer}")

            # Prepare result
            result = {
                'question': question,
                'answer': final_answer,
                'answer_reasoning': answer_reasoning,  # Add answer reasoning to result
                'confidence_score': confidence,  # Add confidence score to result
                'reasoning_chain': established_facts,  # Facts are the reasoning chain
                'retrieved_documents': all_documents,
                'metadata': {
                    'structure': self.name,
                    'iterations': len(iteration_details), 
                    'iteration_details': iteration_details,
                    'total_documents': len(all_documents),
                    'query_time_seconds': query_time,
                    'termination_reason': termination_reason
                }
            }
            
            # Generate detailed JSON report only if requested
            if generate_individual_report and getattr(self.config, 'JSON_REPORT_ENABLE', True):
                self._generate_json_report(
                    original_question=original_question_with_sources,
                    cleaned_question=question,
                    answer=final_answer,
                    answer_reasoning=answer_reasoning,
                    confidence_score=confidence,
                    fact_list=fact_list_for_report,
                    metadata=result['metadata']
                )
            else:
                self.logger.info("Individual JSON report generation skipped")
            
            return result
            
        except Exception as e:
            error_time = time.time() - start_time
            self.logger.error(f"Error in IR-COT query for question '{question[:75]}...': {e}", exc_info=True) 
            
            return self._create_error_result(question, str(e), start_time)

    def _generate_search_query_with_reasoning(self,
                                             question: str,
                                             current_facts: List[str]) -> Tuple[str, str, List[str]]:
        """Generate the next search query using the IR_COT_QUERY_GENERATOR_PROMPT and return query, reasoning, and search methods"""
        try:
            current_facts_text = "\n".join(f"- {fact}" for fact in current_facts) if current_facts else "(empty)"
            
            self.logger.info(f"Generating Search Query with Input:\nOriginal Question: {question}\nCurrent Facts:\n{current_facts_text}")

            prompt_template = PromptTemplate(
                template=PromptTemplates.IR_COT_QUERY_GENERATOR_PROMPT,
                input_variables=["question", "current_cot"]
            )
            
            chain = prompt_template | self.llm_query_generator | self.output_parser
            
            response = chain.invoke({
                "question": question,
                "current_cot": current_facts_text
            })
            
            # Use parser to extract all components
            parsed_response = self.parser.parse_search_query_response(response)
            
            search_query = parsed_response['query']
            reasoning = parsed_response['reasoning']
            keywords = parsed_response['keywords']
            search_methods = parsed_response['search_methods']
            
            # Store keywords for use in retrieval
            self._current_keywords = keywords
            
            self.logger.info(f"Generated Search Query: {search_query}")
            self.logger.info(f"Extracted Keywords: {keywords}")
            self.logger.info(f"Selected Search Methods: {search_methods}")
            return search_query, reasoning, search_methods
            
        except Exception as e:
            self.logger.error(f"Failed to generate search query: {e}", exc_info=True)
            return "", "", ["semantic"]  # Default fallback

    def _generate_json_report(self, original_question: str, cleaned_question: str, answer: str, answer_reasoning: str, confidence_score: int, fact_list: List[Dict], metadata: Dict) -> None:
        """Generate a detailed JSON report for the query"""
        try:
            # Create report data structure
            report = {
                "timestamp": datetime.now().isoformat(),
                "original_question": original_question,
                "cleaned_question": cleaned_question,
                "answer": answer,
                "answer_reasoning": answer_reasoning,
                "confidence_score": confidence_score,
                "fact_list": fact_list,
                "metadata": {
                    "total_iterations": metadata.get('iterations', 0),
                    "total_documents": metadata.get('total_documents', 0),
                    "query_time_seconds": metadata.get('query_time_seconds', 0),
                    "termination_reason": metadata.get('termination_reason', 'Unknown'),
                    "structure": metadata.get('structure', 'IR-COT')
                }
            }
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create a safe filename from the question (first 50 chars, replace special chars)
            safe_question = re.sub(r'[^\w\s-]', '', cleaned_question[:50]).strip()
            safe_question = re.sub(r'[-\s]+', '_', safe_question)
            filename = f"report_{timestamp}_{safe_question}.json"
            
            # Ensure reports directory exists
            reports_dir = getattr(self.config, 'BASE_DIR', '.') / "reports" if hasattr(self.config, 'BASE_DIR') else "reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            # Write report to file
            report_path = os.path.join(reports_dir, filename)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Generated detailed JSON report: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {e}", exc_info=True)

    def _format_documents_for_report(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format documents for inclusion in JSON report"""
        formatted_docs = []
        for doc in documents:
            formatted_doc = {
                "content": doc.get('content', ''),
                "metadata": doc.get('metadata', {}),
                "similarity_score": doc.get('similarity_score', 0.0),
                "doc_id": doc.get('doc_id', ''),
                "search_method": doc.get('search_method', 'semantic')
            }
            
            # Add balanced search information if available
            if 'balanced_score' in doc:
                formatted_doc['balanced_score'] = doc.get('balanced_score', 0.0)
                formatted_doc['methods_found'] = doc.get('methods_found', [])
                formatted_doc['method_scores'] = doc.get('method_scores', {})
            
            formatted_docs.append(formatted_doc)
        return formatted_docs

    def _create_error_result(self, question: str, error_msg: str, start_time: float) -> Dict[str, Any]:
        """Create a standardized error result"""
        error_time = time.time() - start_time
        return {
            'question': question,
            'answer': f"Error occurred during processing: {error_msg}",
            'reasoning_chain': [],
            'retrieved_documents': [],
            'metadata': {
                'structure': self.name,
                'error': error_msg,
                'query_time_seconds': error_time
            }
        }
    
    def _retrieve_documents(self, query: str, search_methods: List[str] = None, retrieved_document_hashes: Optional[set] = None) -> List[Dict[str, Any]]:
        """Retrieve documents using enhanced search with configurable expansion and balanced reranking"""
        self.logger.info(f"Retrieval Query: {query}")
        self.logger.info(f"Search Methods: {search_methods}")
        start_time = time.time()
        
        if search_methods is None:
            search_methods = ['semantic']
        
        # Get keywords for lexical/fuzzy search
        keywords = getattr(self, '_current_keywords', None)
        
        # Use enhanced search if available, otherwise fallback to basic semantic search
        if self.enhanced_search and self.enhanced_search_enable:
            try:
                expanded_top_k = self.top_k * self.expansion_factor
                
                if len(search_methods) == 1 and search_methods[0] == 'semantic':
                    # Single semantic search - use original logic
                    self.logger.debug(f"Single semantic search: fetching {expanded_top_k} documents")
                    expanded_results = self.enhanced_search.semantic_search(
                        query=query,
                        top_k=expanded_top_k
                    )
                    
                else:
                    # Multiple search methods - use balanced retrieval and reranking
                    self.logger.debug(f"Multi-method search using {search_methods}: implementing balanced retrieval")
                    expanded_results = self._balanced_multi_search(query, search_methods, expanded_top_k, keywords)
                
                if not expanded_results:
                    self.logger.info("No documents retrieved from enhanced search.")
                    return []
                
            except Exception as e:
                self.logger.error(f"Enhanced search failed, falling back to basic semantic search: {e}")
                # Fallback to basic semantic search
                expanded_results = self._fallback_semantic_search(query, self.top_k * self.expansion_factor)
        else:
            # No enhanced search available or disabled, use basic semantic search
            if not self.enhanced_search_enable:
                self.logger.info("Enhanced search disabled by config, using basic semantic search")
            else:
                self.logger.warning("Enhanced search not available, using basic semantic search")
            expanded_results = self._fallback_semantic_search(query, self.top_k * self.expansion_factor)
        
        # Apply reranking if reranker is available and enabled
        if self.reranker and self.rerank_enable and expanded_results:
            self.logger.debug(f"Reranking {len(expanded_results)} documents")
            rerank_top_k = getattr(self.config, 'RERANK_TOP_K', self.top_k * 2)  # Rerank more than final top_k
            reranked_results = self.reranker.rerank_documents(
                query=query,
                documents=expanded_results,
                top_k=rerank_top_k
            )
            self.logger.info(f"Reranking: {len(expanded_results)} -> {len(reranked_results)} documents")
            expanded_results = reranked_results
        
        # Apply duplicate filtering if enabled and hash set is provided
        if self.no_repeat_retrieve_enable and retrieved_document_hashes is not None:
            # Filter out documents that have already been retrieved based on content hash
            new_documents = []
            filtered_count = 0
            
            for doc in expanded_results:
                content = doc.get('content', '').strip()
                if content:
                    # Create content hash for deduplication
                    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    
                    if content_hash not in retrieved_document_hashes:
                        # This is a new document
                        new_documents.append(doc)
                        retrieved_document_hashes.add(content_hash)
                        
                        # Stop when we have enough new documents
                        if len(new_documents) >= self.top_k:
                            break
                    else:
                        filtered_count += 1
            
            # Log retrieval statistics
            search_info = f" using {search_methods}" if search_methods != ['semantic'] else ""
            rerank_info = " (with reranking)" if (self.reranker and self.rerank_enable) else ""
            self.logger.info(f"Enhanced retrieval results{search_info}{rerank_info} (with deduplication): Retrieved {len(expanded_results)} docs, "
                           f"filtered out {filtered_count} duplicates, returning {len(new_documents)} new docs")
            
            final_documents = new_documents
        else:
            # No duplicate filtering - just take the top_k documents
            final_documents = expanded_results[:self.top_k]
            
            # Log retrieval statistics
            search_info = f" using {search_methods}" if search_methods != ['semantic'] else ""
            rerank_info = " (with reranking)" if (self.reranker and self.rerank_enable) else ""
            self.logger.info(f"Enhanced retrieval results{search_info}{rerank_info} (no deduplication): Retrieved {len(expanded_results)} docs, "
                           f"returning top {len(final_documents)} docs")
        
        if final_documents:
            formatted_retrieved_docs = self._format_documents_for_display(final_documents)
            self.logger.info(f"Retrieved Documents:\n{formatted_retrieved_docs}")
        else:
            self.logger.info("No documents found after processing.")

        return final_documents
    
    def _balanced_multi_search(self, query: str, search_methods: List[str], total_top_k: int, keywords: str = None) -> List[Dict[str, Any]]:
        """Perform balanced multi-method search and rerank results"""
        
        # Get weights from config instead of hardcoding
        method_weights = getattr(self.config, 'SEARCH_METHOD_WEIGHTS', {
            'semantic': 1.0,   # Primary score
            'lexical': 0.02,   # Small keyword boost
            'fuzzy': 0.02      # Small fuzzy keyword boost
        })
        
        all_results = []
        
        # Retrieve from each search method
        if 'semantic' in search_methods:
            semantic_results = self.enhanced_search.semantic_search(
                query=query,
                top_k=total_top_k
            )
            for result in semantic_results:
                result['method_weights'] = {'semantic': method_weights['semantic']}
                result['base_score'] = result.get('similarity_score', 0.0)
            all_results.extend(semantic_results)
            self.logger.debug(f"Semantic search returned {len(semantic_results)} results")
        
        if 'lexical' in search_methods:
            # Use keywords for lexical search if available
            lexical_query = self._extract_keywords_for_search(keywords) if keywords else query
            lexical_results = self.enhanced_search.lexical_search(
                query=lexical_query,
                top_k=total_top_k
            )
            for result in lexical_results:
                result['method_weights'] = {'lexical': method_weights['lexical']}
                result['base_score'] = result.get('similarity_score', 0.0)
            all_results.extend(lexical_results)
            self.logger.debug(f"Lexical search with query '{lexical_query}' returned {len(lexical_results)} results")
        
        if 'fuzzy' in search_methods:
            # Use keywords for fuzzy search if available 
            fuzzy_query = self._extract_keywords_for_search(keywords) if keywords else query
            fuzzy_results = self.enhanced_search.fuzzy_search(
                query=fuzzy_query,
                top_k=total_top_k
            )
            for result in fuzzy_results:
                result['method_weights'] = {'fuzzy': method_weights['fuzzy']}
                result['base_score'] = result.get('similarity_score', 0.0)
            all_results.extend(fuzzy_results)
            self.logger.debug(f"Fuzzy search with query '{fuzzy_query}' returned {len(fuzzy_results)} results")
        
        # Merge and rerank results using balanced scoring
        merged_results = self._rerank_balanced_results(all_results, method_weights)
        
        self.logger.info(f"Balanced multi-search completed: {len(all_results)} total results merged to {len(merged_results)} final results")
        
        return merged_results
    
    def _extract_keywords_for_search(self, keywords_text: str) -> str:
        """Extract and clean keywords from the keyword search text"""
        return self.parser.extract_keywords_for_search(keywords_text)
    
    def _rerank_balanced_results(self, all_results: List[Dict[str, Any]], method_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Rerank results using balanced scoring across multiple search methods"""
        
        # Group results by content hash for deduplication
        content_groups = defaultdict(list)
        
        for result in all_results:
            content = result.get('content', '').strip()
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            content_groups[content_hash].append(result)
        
        # Calculate balanced scores for each unique document
        reranked_results = []
        
        for content_hash, group in content_groups.items():
            if not group:
                continue
            
            # Use the first result as the base document
            base_doc = group[0].copy()
            
            # Calculate weighted score from all methods that found this document
            total_weighted_score = 0.0
            methods_found = set()
            method_scores = {}
            
            for result in group:
                method_weight_dict = result.get('method_weights', {})
                base_score = result.get('base_score', 0.0)
                
                for method, weight in method_weight_dict.items():
                    # Normalize base_score to 0-1 range if needed
                    normalized_score = min(base_score, 1.0) if base_score > 1.0 else base_score
                    weighted_score = normalized_score * weight
                    total_weighted_score += weighted_score
                    methods_found.add(method)
                    method_scores[method] = normalized_score
            
            # Simple additive scoring - semantic is primary (1.0), others provide small boosts (0.02)
            final_score = total_weighted_score
            
            # Update document with balanced scoring information
            base_doc['balanced_score'] = final_score
            base_doc['methods_found'] = list(methods_found)
            base_doc['method_scores'] = method_scores
            base_doc['search_method'] = 'balanced_multi_search'
            
            reranked_results.append(base_doc)
        
        # Sort by balanced score
        reranked_results.sort(key=lambda x: x.get('balanced_score', 0.0), reverse=True)
        
        # Log scoring details for top results
        for i, result in enumerate(reranked_results[:5]):
            methods = result.get('methods_found', [])
            score = result.get('balanced_score', 0.0)
            method_scores = result.get('method_scores', {})
            self.logger.debug(f"Top result #{i+1}: Score={score:.4f}, Methods={methods}, Breakdown={method_scores}")
        
        return reranked_results

    def _remove_sources_from_question(self, question: str) -> str:
        """Remove source attributions from the question using the REMOVE_SOURCE_PROMPT"""
        try:
            prompt_template = PromptTemplate(
                template=PromptTemplates.REMOVE_SOURCE_PROMPT,
                input_variables=["query"]
            )
            
            chain = prompt_template | self.llm_query_generator | self.output_parser
            
            response = chain.invoke({
                "query": question
            })
            
            # Use parser to extract the reformulated query
            cleaned_question = self.parser.parse_source_removal_response(response)
            return cleaned_question
                
        except Exception as e:
            self.logger.error(f"Failed to remove sources from question: {e}", exc_info=True)
            # Fallback: return original question on error
            return question

    def _extract_new_fact_with_reasoning(self, 
                                       question: str,
                                       current_facts: List[str],
                                       current_search_query: str,
                                       retrieved_docs: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Extract a new fact from retrieved documents using the IR_COT_REASONING_PROMPT"""
        try:
            previous_cot = "\n".join(f"- {fact}" for fact in current_facts) if current_facts else "(empty)"
            formatted_docs = self._format_documents_for_display(retrieved_docs)
            
            prompt_template = PromptTemplate(
                template=PromptTemplates.IR_COT_REASONING_PROMPT,
                input_variables=["question", "previous_cot", "current_search_query", "retrieved_documents_for_query"]
            )
            
            chain = prompt_template | self.llm_reasoner | self.output_parser
            
            response = chain.invoke({
                "question": question,
                "previous_cot": previous_cot,
                "current_search_query": current_search_query,
                "retrieved_documents_for_query": formatted_docs
            })
            
            # Use parser to extract reasoning and new fact
            parsed_response = self.parser.parse_fact_extraction_response(response)
            new_fact = parsed_response['fact']
            reasoning = parsed_response['reasoning']
            
            if not new_fact:
                self.logger.warning("Could not extract new fact from reasoning response")
                return "", ""
            
            return new_fact, reasoning
            
        except Exception as e:
            self.logger.error(f"Failed to extract new fact: {e}", exc_info=True)
            return "", ""

    def _accumulate_documents(self, existing_docs: List[Dict[str, Any]], new_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Accumulate new documents with existing ones, avoiding duplicates"""
        if not new_docs:
            return existing_docs
        
        # Create a set of existing document content hashes for deduplication
        existing_hashes = set()
        for doc in existing_docs:
            content = doc.get('content', '').strip()
            if content:
                content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                existing_hashes.add(content_hash)
        
        # Add new documents that don't already exist
        accumulated_docs = existing_docs.copy()
        added_count = 0
        
        for doc in new_docs:
            content = doc.get('content', '').strip()
            if content:
                content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                if content_hash not in existing_hashes:
                    accumulated_docs.append(doc)
                    existing_hashes.add(content_hash)
                    added_count += 1
        
        self.logger.debug(f"Accumulated {added_count} new documents, total: {len(accumulated_docs)}")
        return accumulated_docs

    def _parse_search_query_from_response(self, response: str) -> str:
        """Parse the search query from LLM response"""
        try:
            if "### Next Search Query:" in response:
                lines = response.split("### Next Search Query:")
                if len(lines) >= 2:
                    query_part = lines[1].split("###")[0].strip()
                    return query_part
            return ""
        except Exception as e:
            self.logger.error(f"Error parsing search query: {e}")
            return ""

    def _parse_query_reasoning_from_response(self, response: str) -> str:
        """Parse the query reasoning from LLM response"""
        try:
            if "### Reasoning for Next Query:" in response:
                lines = response.split("### Reasoning for Next Query:")
                if len(lines) >= 2:
                    reasoning_part = lines[1].split("###")[0].strip()
                    return reasoning_part
            return ""
        except Exception as e:
            self.logger.error(f"Error parsing query reasoning: {e}")
            return ""

    def _parse_keyword_search_from_response(self, response: str) -> str:
        """Parse the keyword search from LLM response"""
        try:
            if "### Keyword Search:" in response:
                lines = response.split("### Keyword Search:")
                if len(lines) >= 2:
                    keyword_part = lines[1].split("###")[0].strip()
                    return keyword_part
            return "NO_NEED"
        except Exception as e:
            self.logger.error(f"Error parsing keyword search: {e}")
            return "NO_NEED"

    def _parse_search_methods_from_response(self, response: str) -> List[str]:
        """Parse the search methods from LLM response based on keyword search"""
        try:
            keywords = self._parse_keyword_search_from_response(response)
            
            # Determine search methods based on keywords and query content
            search_methods = ['semantic']  # Always include semantic search
            
            if keywords and keywords.upper() != "NO_NEED":
                search_methods.append('lexical')
                search_methods.append('fuzzy')
            
            return search_methods
        except Exception as e:
            self.logger.error(f"Error parsing search methods: {e}")
            return ['semantic']

    def _get_termination_reason(self, 
                              established_facts: List[str],
                              current_iteration: int,
                              no_new_docs: bool,
                              no_query_generated: bool) -> str:
        """Determine the reason for termination"""
        if no_query_generated:
            return "no_query_needed"
        elif no_new_docs:
            return "no_new_documents"
        elif current_iteration >= self.max_iterations - 1:
            return "max_iterations_reached"
        elif not established_facts:
            return "no_facts_extracted"
        else:
            return "normal_completion"

    def _generate_final_answer_with_reasoning(self, 
                                            question: str,
                                            established_facts: List[str],
                                            termination_reason: str) -> Tuple[str, str, int]:
        """Generate the final answer using established facts and apply confidence threshold"""
        try:
            current_cot = "\n".join(f"- {fact}" for fact in established_facts) if established_facts else "(empty)"
            
            # Use the unified TERMINATION_FINAL_ANSWER_PROMPT for all cases
            prompt_template = PromptTemplate(
                template=PromptTemplates.TERMINATION_FINAL_ANSWER_PROMPT,
                input_variables=["question", "current_cot"]
            )
            
            chain = prompt_template | self.llm_final_generator | self.output_parser
            
            response = chain.invoke({
                "question": question,
                "current_cot": current_cot
            })
            
            # Use parser to extract reasoning, final answer, and confidence score
            parsed_response = self.parser.parse_final_answer_response(response)
            raw_final_answer = parsed_response['answer']
            reasoning = parsed_response['reasoning']
            confidence = parsed_response.get('confidence', 0)
            
            # Get confidence threshold from config
            confidence_threshold = getattr(self.config, 'CONFIDENCE_THRESHOLD', 3)
            
            # Apply confidence threshold logic
            if confidence < confidence_threshold:
                final_answer = "Insufficient information."
                self.logger.info(f"Confidence score {confidence} below threshold {confidence_threshold}, returning 'Insufficient information'")
            else:
                final_answer = raw_final_answer
                self.logger.info(f"Confidence score {confidence} meets threshold {confidence_threshold}, returning final answer")
            
            # Log confidence information
            self.logger.info(f"Final Answer Confidence: {confidence}/5 (threshold: {confidence_threshold})")
            
            if not raw_final_answer:
                final_answer = "Unable to generate a complete answer based on the available information."
                reasoning = f"Termination reason: {termination_reason}"
            
            return final_answer, reasoning, confidence
            
        except Exception as e:
            self.logger.error(f"Failed to generate final answer: {e}", exc_info=True)
            error_answer = f"Error generating answer: {str(e)}"
            error_reasoning = f"Failed to process due to: {str(e)}"
            return error_answer, error_reasoning, 0

    def _fallback_semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback semantic search when enhanced search is not available"""
        try:
            if self.vector_store:
                similarity_threshold = getattr(self.config, 'SIMILARITY_THRESHOLD', 0.0)
                results = self.vector_store.similarity_search(
                    query=query,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold
                )
                
                # Add search method metadata
                for result in results:
                    result['search_method'] = 'semantic_fallback'
                    result['search_query'] = query
                
                self.logger.debug(f"Fallback semantic search found {len(results)} results")
                return results
            else:
                self.logger.error("No vector store available for fallback search")
                return []
                
        except Exception as e:
            self.logger.error(f"Fallback semantic search failed: {e}")
            return []