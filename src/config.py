import os
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

class Config:
    """Central configuration class for the multi-hop RAG system"""
    
    # Base directory for relative paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # File paths using pathlib for robust path handling
    CORPUS_FILE = BASE_DIR / "data" / "multihoprag_corpus.txt" #cursor cannot access this, but it exists, can refer to multihoprag_corpus_exp.txt as an example
    QUERIES_FILE = BASE_DIR / "data" / "query.csv"
    MULTIHOP_DATA_FILE = BASE_DIR / "data" / "MultiHopRAG_exp.json"
    
    # Mode configuration - can be "SANITY_CHECK", "RUN_ALL", "INTERACTIVE", or "RUN_AND_REPORT"
    MODE = os.getenv("MODE", "RUN_ALL").upper()  # Default to RUN_ALL for testing the new functionality
    
    # Model configurations with separate API keys for different components
    GOOGLE_API_KEY_1 = os.getenv("GOOGLE_API_KEY_1")  # Query generator
    GOOGLE_API_KEY_2 = os.getenv("GOOGLE_API_KEY_2")  # Reasoner (fact generator)
    GOOGLE_API_KEY_3 = os.getenv("GOOGLE_API_KEY_3")  # Reranker + indexer + final generator
    
    LLM_MODEL = "models/gemini-2.5-flash-preview-05-20"  # Standardized model name
    EMBEDDING_MODEL = "models/embedding-001" # thử thay thế text-embedding-004
    
    # Vector store settings (FAISS)
    FAISS_INDEX_PATH = BASE_DIR / "faiss_db" / "index.faiss"
    FAISS_METADATA_PATH = BASE_DIR / "faiss_db" / "metadata.pkl"
    FAISS_INDEX_TYPE = "IndexFlatIP"  # Inner Product (cosine similarity)
    
    # Document processing settings
    CHUNK_SIZE = 1536 # faiss
    CHUNK_OVERLAP = 256
    
    #  hop 1: 5 text -> giúp trả lời câu hỏi 1, vô tình bỏ qua cho câu hỏi 2 (nó nằm trong main question, chỉ trả lời đc cho câu hỏi nhỏ hiện tại)
    # hop 2: 5 text (bỏ qua 5 text hop 1) -> dù ret đc nhưng mà không chọn -> ko trả lời đc fact
    # flow: 2 agent nó khác biệt -> summarization 
    # IR-COT specific settings
    MAX_ITERATIONS = 5# 10
    TOP_K_DOCUMENTS = 5
    EXPANSION_FACTOR = 5  # Multiplier for enhanced retrieval (fetch N times more documents)
    
    NO_REPEAT_RETRIEVE_ENABLE = False  # Enable duplicate filtering across iterations
    REMOVE_SOURCES_ENABLE = False  # Enable source removal preprocessing
    SIMILARITY_THRESHOLD = 0.5
    
    # Search settings (disabled for now)
    WEB_SEARCH_ENABLED = False
    MAX_WEB_RESULTS = 5
    
    # Enhanced search settings
    ENHANCED_SEARCH_ENABLE = True  # Enable/disable multi-method search (semantic + lexical + fuzzy)
    
    # Confidence score settings
    CONFIDENCE_THRESHOLD = 3  # Minimum confidence score (0-5) to return final answer, else return "Insufficient information"
    
    # Reranker settings (disabled for now)  
    RERANK_TOP_K = 5 
    MIN_RELEVANCE_SCORE = 0.5
    RERANK_ENABLE = False  # Enable/disable reranking functionality
    
    # Search method weights for balanced multi-search
    SEARCH_METHOD_WEIGHTS = {
        'semantic': 1.0,    # Primary score - semantic similarity is the main ranking factor
        'lexical': 0.02,    # Small boost for keyword matches
        'fuzzy': 0.01       # Small boost for fuzzy keyword matches
    }

    # Logging configuration
    LOGGING_ITERATION_INTERVAL = 5  # Log iterative steps every N iterations
    LOG_FILE = BASE_DIR / "reasoning_chain.txt"  # Path to the log file
    
    # JSON reporting configuration
    REPORTS_DIR = BASE_DIR / "reports"
    JSON_REPORT_ENABLE = True  # Enable detailed JSON reporting
    
    @classmethod
    def get_api_key_for_component(cls, component: str) -> str:
        """Get the appropriate API key for a specific component"""
        api_key_map = {
            'query_generator': cls.GOOGLE_API_KEY_1,
            'reasoner': cls.GOOGLE_API_KEY_2,
            'reranker': cls.GOOGLE_API_KEY_3,
            'indexer': cls.GOOGLE_API_KEY_3,
            'final_generator': cls.GOOGLE_API_KEY_3
        }
        
        # Get component-specific key or fallback to general key
        api_key = api_key_map.get(component) or cls.GOOGLE_API_KEY_1
        
        if not api_key:
            raise ValueError(f"No API key found for component '{component}'. Check your .env file.")
        
        return api_key
    
    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary"""
        settings = {}
        for key in dir(cls):
            if not key.startswith('_') and not callable(getattr(cls, key)):
                value = getattr(cls, key)
                # Convert Path objects to strings for serialization
                if isinstance(value, Path):
                    value = str(value)
                settings[key] = value
        return settings
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configurations are set"""
        # Check for at least one API key
        if not any([cls.GOOGLE_API_KEY_1, cls.GOOGLE_API_KEY_2, cls.GOOGLE_API_KEY_3]):
            raise ValueError("At least one GOOGLE_API_KEY environment variable is required")
        
        # Create directories if they don't exist
        cls.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check for data files and warn if missing
        data_files = {
            "CORPUS_FILE": cls.CORPUS_FILE,
            "QUERIES_FILE": cls.QUERIES_FILE, 
            "MULTIHOP_DATA_FILE": cls.MULTIHOP_DATA_FILE
        }
        
        missing_files = []
        for name, file_path in data_files.items():
            if not file_path.exists():
                missing_files.append(f"{name}: {file_path}")
        
        if missing_files:
            print("Warning: The following data files were not found:")
            for missing in missing_files:
                print(f"  - {missing}")
            print("Some functionality may be limited.")
        
        return True 