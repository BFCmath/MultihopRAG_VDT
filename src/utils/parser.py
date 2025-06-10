#!/usr/bin/env python3
"""
Parser Module for IR-COT Structure
Handles parsing of LLM responses for various IR-COT components
"""

import logging
from typing import List, Dict, Any, Tuple
import re

class IRCOTResponseParser:
    """Parser for IR-COT LLM responses"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def parse_search_query_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the complete query generator response to extract all components
        
        Returns:
            Dict containing:
            - reasoning: str - The reasoning for the next query
            - query: str - The search query
            - keywords: str - The keyword search terms
            - search_methods: List[str] - Determined search methods
        """
        try:
            result = {
                'reasoning': self._parse_query_reasoning(response),
                'query': self._parse_search_query(response),
                'keywords': self._parse_keyword_search(response),
            }
            
            # Determine search methods based on keywords
            result['search_methods'] = self._determine_search_methods(result['keywords'])
            
            self.logger.debug(f"Parsed query response: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing search query response: {e}")
            return {
                'reasoning': "",
                'query': "",
                'keywords': "NO_NEED",
                'search_methods': ['semantic']
            }
    
    def parse_fact_extraction_response(self, response: str) -> Dict[str, str]:
        """
        Parse the fact extraction response to get reasoning and new fact
        
        Returns:
            Dict containing:
            - reasoning: str - The reasoning process
            - fact: str - The extracted fact
        """
        try:
            reasoning = ""
            fact = ""
            
            if "### Reasoning:" in response and "### New Fact:" in response:
                parts = response.split("### New Fact:")
                if len(parts) >= 2:
                    reasoning_part = parts[0].replace("### Reasoning:", "").strip()
                    fact = parts[1].strip()
                    reasoning = reasoning_part
            
            if not fact:
                self.logger.warning("Could not extract new fact from reasoning response")
                return {"reasoning": "", "fact": ""}
            
            return {"reasoning": reasoning, "fact": fact}
            
        except Exception as e:
            self.logger.error(f"Error parsing fact extraction response: {e}")
            return {"reasoning": "", "fact": ""}
    
    def parse_final_answer_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the final answer response to get reasoning, answer, and confidence score
        
        Returns:
            Dict containing:
            - reasoning: str - The reasoning for the final answer
            - answer: str - The final answer
            - confidence: int - The confidence score (0-5)
        """
        try:
            reasoning = ""
            final_answer = ""
            confidence = 0
            
            # Check for the new format with confidence score
            if ("### Reasoning for Final Answer:" in response and 
                "### Final Answer:" in response and 
                "### Confidence Score:" in response):
                
                # Split by sections
                parts = response.split("### Final Answer:")
                if len(parts) >= 2:
                    # Extract reasoning
                    reasoning_part = parts[0].replace("### Reasoning for Final Answer:", "").strip()
                    reasoning = reasoning_part
                    
                    # Split the second part to get answer and confidence
                    answer_confidence_parts = parts[1].split("### Confidence Score:")
                    if len(answer_confidence_parts) >= 2:
                        final_answer = answer_confidence_parts[0].strip()
                        confidence_text = answer_confidence_parts[1].strip()
                        
                        # Extract numeric confidence score
                        confidence = self._extract_confidence_score(confidence_text)
                    else:
                        final_answer = parts[1].strip()
                        
            # Fallback for old format without confidence score
            elif "### Reasoning for Final Answer:" in response and "### Final Answer:" in response:
                parts = response.split("### Final Answer:")
                if len(parts) >= 2:
                    reasoning_part = parts[0].replace("### Reasoning for Final Answer:", "").strip()
                    final_answer = parts[1].strip()
                    reasoning = reasoning_part
                    confidence = 3  # Default confidence for old format
            else:
                # Fallback: use entire response as answer
                final_answer = response.strip()
                reasoning = "Generated from established facts"
                confidence = 1  # Low confidence for fallback
            
            if not final_answer:
                final_answer = "Unable to generate a complete answer based on the available information."
                reasoning = "No clear answer could be generated"
                confidence = 0
            
            return {
                "reasoning": reasoning, 
                "answer": final_answer,
                "confidence": confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing final answer response: {e}")
            return {
                "reasoning": f"Failed to process due to: {str(e)}", 
                "answer": f"Error generating answer: {str(e)}",
                "confidence": 0
            }
    
    def _extract_confidence_score(self, confidence_text: str) -> int:
        """Extract numeric confidence score from text"""
        try:
            # Look for numbers in the confidence text
            import re
            numbers = re.findall(r'\d+', confidence_text)
            if numbers:
                score = int(numbers[0])
                # Clamp to valid range 0-5
                return max(0, min(5, score))
            return 0
        except Exception as e:
            self.logger.error(f"Error extracting confidence score: {e}")
            return 0
    
    def parse_source_removal_response(self, response: str) -> str:
        """
        Parse the source removal response to get cleaned question
        
        Returns:
            str - The cleaned question without source attributions
        """
        try:
            if "### Reformulated Query:" in response:
                cleaned_question = response.split("### Reformulated Query:")[-1].strip()
                return cleaned_question
            else:
                # Return original response if format is not as expected
                self.logger.warning("Could not parse reformulated query, using original response")
                return response.strip()
                
        except Exception as e:
            self.logger.error(f"Error parsing source removal response: {e}")
            return response  # Fallback to original
    
    # Private helper methods for parsing specific sections
    
    def _parse_search_query(self, response: str) -> str:
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
    
    def _parse_query_reasoning(self, response: str) -> str:
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
    
    def _parse_keyword_search(self, response: str) -> str:
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
    
    def _determine_search_methods(self, keywords: str) -> List[str]:
        """Determine search methods based on keyword search"""
        try:
            # Always include semantic search
            search_methods = ['semantic']
            
            # Add lexical and fuzzy search if keywords are needed
            if keywords and keywords.upper() != "NO_NEED":
                search_methods.append('lexical')
                search_methods.append('fuzzy')
            
            return search_methods
        except Exception as e:
            self.logger.error(f"Error determining search methods: {e}")
            return ['semantic']
    
    def extract_keywords_for_search(self, keywords_text: str) -> str:
        """Extract and clean keywords from the keyword search text"""
        if not keywords_text or keywords_text.strip().upper() == "NO_NEED":
            return ""
        
        # Remove quotes and split by commas
        keywords = []
        for keyword in keywords_text.split(','):
            cleaned = keyword.strip().strip('"').strip("'")
            if cleaned:
                keywords.append(cleaned)
        
        return " ".join(keywords)
    
    def validate_query_response(self, response: str) -> bool:
        """Validate that a query response has the expected format"""
        required_sections = [
            "### Reasoning for Next Query:",
            "### Next Search Query:",
            "### Keyword Search:"
        ]
        
        for section in required_sections:
            if section not in response:
                self.logger.warning(f"Missing section in query response: {section}")
                return False
        
        return True
    
    def validate_fact_response(self, response: str) -> bool:
        """Validate that a fact extraction response has the expected format"""
        required_sections = [
            "### Reasoning:",
            "### New Fact:"
        ]
        
        for section in required_sections:
            if section not in response:
                self.logger.warning(f"Missing section in fact response: {section}")
                return False
        
        return True
    
    def validate_final_answer_response(self, response: str) -> bool:
        """Validate that a final answer response has the expected format"""
        required_sections = [
            "### Reasoning for Final Answer:",
            "### Final Answer:",
            "### Confidence Score:"
        ]
        
        for section in required_sections:
            if section not in response:
                self.logger.warning(f"Missing section in final answer response: {section}")
                return False
        
        return True 