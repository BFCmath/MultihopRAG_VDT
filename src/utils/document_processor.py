import re
import json
import csv
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Compile regex patterns once for efficiency
DATE_PATTERN = re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b')
ENTITY_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
WHITESPACE_PATTERN = re.compile(r'\s+')
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s.,!?;:()-]')

@dataclass
class Document:
    """Document data structure"""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'metadata': self.metadata,
            'doc_id': self.doc_id
        }

class DocumentProcessor:
    """General document processor for multi-hop RAG systems"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, max_acceptable_size: int = 800):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_acceptable_size = max_acceptable_size  # Size below which we keep whole passage
        # Simple cache for chunked documents
        self._chunk_cache = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def parse_corpus_file(self, file_path: str) -> List[Document]:
        """Parse a corpus file and return list of documents"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse documents based on Title: and <endofpassage> markers
            # Split by <endofpassage> to get individual passages
            passages = content.split('<endofpassage>')
            
            for i, passage in enumerate(passages):
                passage = passage.strip()
                if not passage:  # Skip empty passages
                    continue
                
                # Extract title if it starts with "Title:"
                title = ""
                content_text = passage
                
                if passage.startswith("Title:"):
                    lines = passage.split('\n', 1)
                    if len(lines) > 1:
                        title = lines[0].replace("Title:", "").strip()
                        content_text = lines[1].strip()
                    else:
                        title = lines[0].replace("Title:", "").strip()
                        content_text = ""
                
                if content_text:  # Only create document if there's content
                    metadata = {
                        'source': file_path, 
                        'passage_id': i,
                        'title': title if title else f"Passage {i+1}",
                        'original_length': len(content_text)
                    }
                    
                    doc = Document(
                        content=content_text,
                        metadata=metadata,
                        doc_id=f"passage_{i}"
                    )
                    documents.append(doc)
                    
        except Exception as e:
            self.logger.error(f"Error parsing corpus file {file_path}: {e}", exc_info=True)
            
        return documents
    
    def parse_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse a JSON file and return the data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data if isinstance(data, list) else [data]
        except Exception as e:
            self.logger.error(f"Error parsing JSON file {file_path}: {e}", exc_info=True)
            return []
    
    def parse_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse a CSV file and return list of dictionaries"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                return list(reader)
        except Exception as e:
            self.logger.error(f"Error parsing CSV file {file_path}: {e}", exc_info=True)
            return []
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks with overlap"""
        # Use a simple cache key based on document content and settings
        cache_key = (
            hash(tuple(doc.content for doc in documents)),
            self.chunk_size,
            self.chunk_overlap,
            self.max_acceptable_size
        )
        
        if cache_key in self._chunk_cache:
            return self._chunk_cache[cache_key]
        
        chunked_docs = []
        
        for i, doc in enumerate(documents):
            # Check if passage is acceptable size as-is
            if len(doc.content) <= self.max_acceptable_size:
                # Keep the whole passage
                chunked_docs.append(doc)
                self.logger.debug(f"Keeping whole passage {doc.doc_id} (length: {len(doc.content)})")
            else:
                # Need to chunk this passage
                chunks = self._chunk_text_with_overlap(doc.content)
                self.logger.debug(f"Chunking passage {doc.doc_id} into {len(chunks)} chunks")
                
                for j, chunk_text in enumerate(chunks):
                    chunk_doc = Document(
                        content=chunk_text,
                        metadata={
                            **doc.metadata,
                            'parent_doc_id': doc.doc_id,
                            'chunk_index': j,
                            'total_chunks': len(chunks),
                            'chunking_method': 'overlap_chunking'
                        },
                        doc_id=f"{doc.doc_id}_chunk_{j}"
                    )
                    chunked_docs.append(chunk_doc)
            
            # Add overlap between passages (if not the last document)
            if i < len(documents) - 1 and chunked_docs:
                self._add_passage_overlap(chunked_docs, documents[i+1])
        
        # Cache the result
        self._chunk_cache[cache_key] = chunked_docs
        return chunked_docs
    
    def _chunk_text_with_overlap(self, text: str) -> List[str]:
        """Split text into chunks with overlap using dynamic sizing and word boundaries"""
        # Calculate optimal dynamic chunk size
        estimated_chunks = max(1, len(text) // self.chunk_size)
        dynamic_chunk_size = len(text) // estimated_chunks
        
        # Ensure dynamic chunk size is reasonable
        dynamic_chunk_size = max(self.chunk_size // 2, min(dynamic_chunk_size, self.chunk_size * 2))
        
        self.logger.debug(f"Dynamic chunk size: {dynamic_chunk_size} for text length: {len(text)}")
        
        # Split text into chunks
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + dynamic_chunk_size
            
            # If this is the last chunk or we're near the end, take everything
            if end >= len(text) or len(text) - end < dynamic_chunk_size // 3:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # Try to find a good breaking point (word boundary)
            # Look backwards from the end position for whitespace
            best_end = end
            for i in range(end, max(start + dynamic_chunk_size // 2, end - 100), -1):
                if text[i].isspace():
                    best_end = i
                    break
            
            chunk = text[start:best_end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = max(start + 1, best_end - self.chunk_overlap)
        
        return chunks
    
    def _add_passage_overlap(self, chunked_docs: List[Document], next_passage: Document):
        """Add overlap between the last chunk of current passage and beginning of next passage"""
        if not chunked_docs:
            return
        
        last_chunk = chunked_docs[-1]
        next_passage_start = next_passage.content[:self.chunk_overlap].strip()
        
        if next_passage_start:
            # Create an overlap chunk
            overlap_content = last_chunk.content + " " + next_passage_start
            overlap_doc = Document(
                content=overlap_content,
                metadata={
                    **last_chunk.metadata,
                    'chunk_type': 'passage_overlap',
                    'overlap_with': next_passage.doc_id
                },
                doc_id=f"{last_chunk.doc_id}_overlap_{next_passage.doc_id}"
            )
            chunked_docs.append(overlap_doc)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text using pre-compiled regex patterns"""
        # Remove extra whitespace
        text = WHITESPACE_PATTERN.sub(' ', text)
        # Remove special characters but keep basic punctuation
        text = SPECIAL_CHARS_PATTERN.sub('', text)
        return text.strip()
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text (titles, dates, etc.)"""
        metadata = {}
        
        # Extract potential title (first line if it's short)
        lines = text.split('\n')
        if lines and len(lines[0]) < 100:
            metadata['title'] = lines[0].strip()
        
        # Extract dates
        dates = DATE_PATTERN.findall(text)
        if dates:
            metadata['dates'] = dates
        
        # Extract potential entities (capitalized words)
        entities = ENTITY_PATTERN.findall(text)
        if entities:
            metadata['entities'] = list(set(entities))[:10]  # Top 10 unique entities
        
        return metadata
    
    def format_document_for_retrieval(self, doc: Document) -> str:
        """Format document for retrieval display"""
        content = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
        metadata_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items() if k != 'chunk_index'])
        return f"[{metadata_str}]\n{content}"
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        if not documents:
            return {}
        
        total_docs = len(documents)
        total_chars = sum(len(doc.content) for doc in documents)
        avg_length = total_chars / total_docs
        
        # Count different chunking methods used
        chunking_methods = {}
        whole_passages = 0
        
        for doc in documents:
            if 'chunking_method' in doc.metadata:
                method = doc.metadata['chunking_method']
                chunking_methods[method] = chunking_methods.get(method, 0) + 1
            else:
                whole_passages += 1
        
        return {
            'total_documents': total_docs,
            'total_characters': total_chars,
            'average_length': avg_length,
            'longest_doc': max(len(doc.content) for doc in documents),
            'shortest_doc': min(len(doc.content) for doc in documents),
            'whole_passages_kept': whole_passages,
            'chunking_methods_used': chunking_methods
        } 