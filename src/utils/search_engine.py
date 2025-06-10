import requests
import time
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
import json

class SearchEngine:
    """General search engine API for multi-hop RAG systems"""
    
    def __init__(self, 
                 max_results: int = 5,
                 search_timeout: int = 10,
                 rate_limit_delay: float = 1.0):
        
        self.max_results = max_results
        self.search_timeout = search_timeout
        self.rate_limit_delay = rate_limit_delay
        self.last_search_time = 0
    
    def search(self, 
               query: str, 
               num_results: Optional[int] = None,
               search_type: str = "general") -> List[Dict[str, Any]]:
        """Perform web search and return results"""
        num_results = num_results or self.max_results
        
        # Rate limiting
        self._rate_limit()
        
        try:
            if search_type == "news":
                return self._search_news(query, num_results)
            elif search_type == "academic":
                return self._search_academic(query, num_results)
            else:
                return self._search_general(query, num_results)
                
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def _rate_limit(self):
        """Implement rate limiting between searches"""
        current_time = time.time()
        time_since_last = current_time - self.last_search_time
        
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_search_time = time.time()
    
    def _search_general(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """General web search using DuckDuckGo"""
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=self.search_timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract abstract if available
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', 'DuckDuckGo Abstract'),
                    'content': data.get('Abstract', ''),
                    'url': data.get('AbstractURL', ''),
                    'source': 'DuckDuckGo',
                    'type': 'abstract'
                })
            
            # Extract related topics
            for topic in data.get('RelatedTopics', [])[:num_results-len(results)]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('Text', '')[:100] + '...',
                        'content': topic.get('Text', ''),
                        'url': topic.get('FirstURL', ''),
                        'source': 'DuckDuckGo',
                        'type': 'related_topic'
                    })
            
            # If no results, try alternative search
            if not results:
                return self._search_alternative(query, num_results)
            
            return results[:num_results]
            
        except Exception as e:
            print(f"Error in general search: {e}")
            return self._search_alternative(query, num_results)
    
    def _search_alternative(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Alternative search method when primary fails"""
        try:
            # Use a simple Wikipedia search as fallback
            wiki_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
            encoded_query = quote_plus(query.replace(' ', '_'))
            
            response = requests.get(
                f"{wiki_url}{encoded_query}", 
                timeout=self.search_timeout,
                headers={'User-Agent': 'MultiHopRAG/1.0'}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('extract'):
                    return [{
                        'title': data.get('title', query),
                        'content': data.get('extract', ''),
                        'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                        'source': 'Wikipedia',
                        'type': 'summary'
                    }]
            
            # If all else fails, return a placeholder
            return [{
                'title': f"Search results for: {query}",
                'content': f"No specific search results found for '{query}'. This query may require more specific terms or alternative search approaches.",
                'url': '',
                'source': 'System',
                'type': 'placeholder'
            }]
            
        except Exception as e:
            print(f"Error in alternative search: {e}")
            return []
    
    def _search_news(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search for news articles"""
        try:
            # Use DuckDuckGo news search
            # Note: This is a simplified implementation
            # In production, you might want to use dedicated news APIs
            
            url = "https://api.duckduckgo.com/"
            params = {
                'q': f"{query} news",
                'format': 'json',
                'no_html': '1'
            }
            
            response = requests.get(url, params=params, timeout=self.search_timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Process news results
            for item in data.get('RelatedTopics', [])[:num_results]:
                if isinstance(item, dict) and item.get('Text'):
                    results.append({
                        'title': item.get('Text', '')[:100] + '...',
                        'content': item.get('Text', ''),
                        'url': item.get('FirstURL', ''),
                        'source': 'News Search',
                        'type': 'news'
                    })
            
            return results
            
        except Exception as e:
            print(f"Error in news search: {e}")
            return []
    
    def _search_academic(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search for academic/research content"""
        try:
            # This is a placeholder for academic search
            # In production, you might integrate with:
            # - arXiv API
            # - PubMed API
            # - Google Scholar (though it's more restricted)
            # - Semantic Scholar API
            
            # For now, use general search with academic keywords
            academic_query = f"{query} research study academic paper"
            return self._search_general(academic_query, num_results)
            
        except Exception as e:
            print(f"Error in academic search: {e}")
            return []
    
    def search_multiple_queries(self, 
                               queries: List[str], 
                               max_results_per_query: int = 3) -> List[Dict[str, Any]]:
        """Search multiple queries and combine results"""
        all_results = []
        
        for query in queries:
            results = self.search(query, max_results_per_query)
            for result in results:
                result['search_query'] = query
            all_results.extend(results)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            url = result.get('url', '')
            content_hash = hash(result.get('content', ''))
            
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
            elif not url and content_hash not in seen_urls:
                seen_urls.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for display or processing"""
        if not results:
            return "No search results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            content = result.get('content', 'No content')
            url = result.get('url', 'No URL')
            source = result.get('source', 'Unknown')
            
            # Truncate content if too long
            if len(content) > 300:
                content = content[:300] + "..."
            
            formatted.append(f"""
Result {i}:
Title: {title}
Source: {source}
Content: {content}
URL: {url}
""".strip())
        
        return "\n\n".join(formatted)
    
    def extract_key_information(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key information from search results"""
        if not results:
            return {}
        
        # Combine all content
        all_content = " ".join([result.get('content', '') for result in results])
        
        # Extract URLs
        urls = [result.get('url') for result in results if result.get('url')]
        
        # Extract sources
        sources = list(set([result.get('source') for result in results if result.get('source')]))
        
        return {
            'total_results': len(results),
            'combined_content': all_content,
            'urls': urls,
            'sources': sources,
            'avg_content_length': len(all_content) / len(results) if results else 0
        }
    
    def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions based on the query"""
        # Generate related search queries
        suggestions = [
            f"{query} definition",
            f"{query} examples", 
            f"{query} recent developments",
            f"{query} comparison",
            f"what is {query}",
            f"how does {query} work"
        ]
        
        return suggestions
    
    def validate_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean search results"""
        valid_results = []
        
        for result in results:
            # Check if result has minimum required fields
            if result.get('content') and len(result.get('content', '').strip()) > 10:
                # Clean the result
                cleaned_result = {
                    'title': result.get('title', 'Untitled').strip(),
                    'content': result.get('content', '').strip(),
                    'url': result.get('url', '').strip(),
                    'source': result.get('source', 'Unknown').strip(),
                    'type': result.get('type', 'general')
                }
                valid_results.append(cleaned_result)
        
        return valid_results 