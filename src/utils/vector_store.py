import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional

from langchain_core.embeddings import Embeddings # Import Langchain Embeddings

from .document_processor import Document

class VectorStore:
    """FAISS-based vector store for multi-hop RAG systems, using Langchain Embeddings"""
    
    def __init__(self, 
                 embeddings: Embeddings,
                 index_path: str = "./faiss_db/index.faiss",
                 metadata_path: str = "./faiss_db/metadata.pkl",
                 index_type: str = "IndexFlatIP"):
        
        self.embeddings_model = embeddings
        self.index_path = str(index_path)
        self.metadata_path = str(metadata_path)
        self.index_type = index_type
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Initialize FAISS index and metadata
        self.index = None
        self.metadata = {}  # Store document metadata and content
        self.embedding_dimension = None
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                if self.index.ntotal > 0:
                    self.embedding_dimension = self.index.d
                    print(f"INFO: Loaded existing FAISS index with {self.index.ntotal} documents")
                else:
                    print("INFO: Loaded empty FAISS index")
            else:
                print("INFO: No existing FAISS index found, will create new one")
        except Exception as e:
            print(f"ERROR: Failed to load FAISS index: {e}")
            self.index = None
            self.metadata = {}
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(self.metadata, f)
                print("INFO: Saved FAISS index and metadata")
                return True
        except Exception as e:
            print(f"ERROR: Failed to save FAISS index: {e}")
            return False
    
    def _create_index(self, dimension: int):
        """Create a new FAISS index"""
        try:
            if self.index_type == "IndexFlatIP":
                # Inner Product index (good for normalized embeddings/cosine similarity)
                self.index = faiss.IndexFlatIP(dimension)
            elif self.index_type == "IndexFlatL2":
                # L2 distance index
                self.index = faiss.IndexFlatL2(dimension)
            else:
                # Default to Inner Product
                self.index = faiss.IndexFlatIP(dimension)
            
            self.embedding_dimension = dimension
            print(f"INFO: Created new FAISS index with dimension {dimension}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to create FAISS index: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        if not documents:
            print("WARNING: No documents provided to add")
            return True
            
        try:
            texts = [doc.content for doc in documents]
            
            # Generate embeddings
            embeddings = self._get_embeddings(texts)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Create index if it doesn't exist
            if self.index is None:
                if not self._create_index(embeddings_array.shape[1]):
                    return False
            
            # Normalize embeddings for cosine similarity if using IndexFlatIP
            if self.index_type == "IndexFlatIP":
                faiss.normalize_L2(embeddings_array)
            
            # Get starting index for new documents
            start_idx = self.index.ntotal
            
            # Add embeddings to index
            self.index.add(embeddings_array)
            
            # Store metadata
            for i, doc in enumerate(documents):
                doc_idx = start_idx + i
                doc_id = doc.doc_id or f"doc_{doc_idx}"
                self.metadata[doc_idx] = {
                    'doc_id': doc_id,
                    'content': doc.content,
                    'metadata': doc.metadata or {}
                }
            
            # Save to disk
            self._save_index()
            
            print(f"INFO: Added {len(documents)} documents to FAISS vector store")
            return True
            
        except ValueError as e:
            print(f"ERROR: Invalid input for adding documents: {e}")
            return False
        except Exception as e:
            print(f"ERROR: Error adding documents to vector store: {e}")
            return False
    
    def similarity_search(self, 
                         query: str, 
                         top_k: int = 10,
                         similarity_threshold: float = 0.0,
                         metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not query.strip():
            print("WARNING: Empty query provided for similarity search")
            return []
            
        if self.index is None or self.index.ntotal == 0:
            print("WARNING: No documents in vector store")
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Normalize for cosine similarity if using IndexFlatIP
            if self.index_type == "IndexFlatIP":
                faiss.normalize_L2(query_vector)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            # Format results
            formatted_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                # Convert score to similarity (FAISS returns different metrics based on index type)
                if self.index_type == "IndexFlatIP":
                    similarity = float(score)  # Already similarity for IP
                else:
                    # For L2 distance, convert to similarity
                    similarity = 1 / (1 + float(score))
                
                if similarity >= similarity_threshold:
                    doc_data = self.metadata.get(idx, {})
                    
                    # Apply metadata filter if provided
                    if metadata_filter:
                        doc_metadata = doc_data.get('metadata', {})
                        if not self._matches_filter(doc_metadata, metadata_filter):
                            continue
                    
                    result = {
                        'doc_id': doc_data.get('doc_id', f"doc_{idx}"),
                        'content': doc_data.get('content', ''),
                        'metadata': doc_data.get('metadata', {}),
                        'similarity_score': similarity,
                        'distance': float(score) if self.index_type != "IndexFlatIP" else 1 - similarity
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except ValueError as e:
            print(f"ERROR: Invalid query for similarity search: {e}")
            return []
        except Exception as e:
            print(f"ERROR: Error during similarity search: {e}")
            return []
    
    def _matches_filter(self, doc_metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if document metadata matches the filter"""
        for key, value in filter_dict.items():
            if key not in doc_metadata or doc_metadata[key] != value:
                return False
        return True
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the provided Langchain Embeddings object"""
        try:
            # Clean texts before embedding
            cleaned_texts = []
            for text in texts:
                clean_text = text.replace('\n', ' ').strip()
                if not clean_text:
                    clean_text = "Empty document"
                cleaned_texts.append(clean_text)

            if not cleaned_texts:
                print("WARNING: No valid texts provided for embedding")
                raise ValueError("No valid texts provided for embedding")
                
            # Use the embed_documents method from Langchain Embeddings
            embeddings = self.embeddings_model.embed_documents(cleaned_texts)
            return embeddings
            
        except Exception as e:
            print(f"ERROR: Error generating embeddings with Langchain: {e}")
            raise e
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query using Langchain Embeddings object"""
        try:
            if not query.strip():
                raise ValueError("Empty query provided for embedding")
            return self.embeddings_model.embed_query(query)
        except Exception as e:
            print(f"ERROR: Error generating query embedding with Langchain: {e}")
            raise e
    
    def delete_collection(self) -> bool:
        """Delete the entire index and metadata"""
        try:
            self.index = None
            self.metadata = {}
            
            # Remove files
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
                
            print("INFO: Deleted FAISS index and metadata")
            return True
        except Exception as e:
            print(f"ERROR: Error deleting index: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the index"""
        try:
            count = self.index.ntotal if self.index is not None else 0
            return {
                'name': 'faiss_index',
                'document_count': count,
                'persist_directory': os.path.dirname(self.index_path),
                'index_type': self.index_type,
                'dimension': self.embedding_dimension
            }
        except Exception as e:
            print(f"ERROR: Error getting index info: {e}")
            return {}
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the index"""
        try:
            for doc_data in self.metadata.values():
                if doc_data.get('doc_id') == doc_id:
                    return True
            return False
        except Exception as e:
            print(f"ERROR: Error checking document existence: {e}")
            return False
    
    def update_document(self, doc_id: str, document: Document) -> bool:
        """Update an existing document (requires rebuilding index)"""
        try:
            # Find the document index
            doc_idx = None
            for idx, doc_data in self.metadata.items():
                if doc_data.get('doc_id') == doc_id:
                    doc_idx = idx
                    break
            
            if doc_idx is None:
                print(f"ERROR: Document {doc_id} not found")
                return False
            
            # Update metadata
            self.metadata[doc_idx] = {
                'doc_id': doc_id,
                'content': document.content,
                'metadata': document.metadata or {}
            }
            
            # For simplicity, we'll need to rebuild the entire index
            # In a production system, you might want a more efficient approach
            return self._rebuild_index()
            
        except Exception as e:
            print(f"ERROR: Error updating document: {e}")
            return False
    
    def _rebuild_index(self) -> bool:
        """Rebuild the entire FAISS index from metadata"""
        try:
            if not self.metadata:
                return True
            
            # Extract all documents
            texts = [doc_data['content'] for doc_data in self.metadata.values()]
            
            # Generate embeddings
            embeddings = self._get_embeddings(texts)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Create new index
            if not self._create_index(embeddings_array.shape[1]):
                return False
            
            # Normalize if needed
            if self.index_type == "IndexFlatIP":
                faiss.normalize_L2(embeddings_array)
            
            # Add all embeddings
            self.index.add(embeddings_array)
            
            # Save to disk
            self._save_index()
            
            print("INFO: Rebuilt FAISS index")
            return True
            
        except Exception as e:
            print(f"ERROR: Error rebuilding index: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a specific document (requires rebuilding index)"""
        try:
            # Find and remove from metadata
            doc_idx = None
            for idx, doc_data in self.metadata.items():
                if doc_data.get('doc_id') == doc_id:
                    doc_idx = idx
                    break
            
            if doc_idx is None:
                print(f"ERROR: Document {doc_id} not found")
                return False
            
            del self.metadata[doc_idx]
            
            # Rebuild index without the deleted document
            return self._rebuild_index()
            
        except Exception as e:
            print(f"ERROR: Error deleting document: {e}")
            return False
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from the index"""
        try:
            documents = []
            
            for doc_data in self.metadata.values():
                doc = {
                    'doc_id': doc_data.get('doc_id', ''),
                    'content': doc_data.get('content', ''),
                    'metadata': doc_data.get('metadata', {})
                }
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"ERROR: Error getting all documents: {e}")
            return []
    
    def reset_collection(self) -> bool:
        """Reset the index (delete and recreate empty)"""
        try:
            self.delete_collection()
            self.index = None
            self.metadata = {}
            print("INFO: Reset FAISS index")
            return True
        except Exception as e:
            print(f"ERROR: Error resetting index: {e}")
            return False 