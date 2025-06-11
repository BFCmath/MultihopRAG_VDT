#!/usr/bin/env python3
"""
Main entry point for the simplified Multi-hop RAG system with IR-COT structure.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List
import logging

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from config import Config
from prompts import PromptTemplates
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
# Note: Reranker and SearchEngine imports excluded for now
# from utils.reranker import Reranker
# from utils.search_engine import SearchEngine
from ir_cot_structure import IRCOTStructure

# Langchain imports for initialization
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

def initialize_system() -> Dict[str, Any]:
    """Initialize all system components"""
    # This print is user-facing, keep it.
    print("Initializing Multi-hop RAG System...") 
    try:
        # Validate configuration
        Config.validate_config()
        
        # Initialize Langchain Embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            google_api_key=Config.GOOGLE_API_KEY_3
        )
        
        # Initialize Langchain LLM for Reranker and IRCOT (if different models are needed, instantiate separately)
        # For simplicity, using the same LLM model defined in Config for both.
        llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL, # This will be used by Reranker and IRCOT
            google_api_key=Config.GOOGLE_API_KEY_3
        )
        
        # IRCOT will use its own configured model name for its primary LLM
        # but if Reranker needs a specific one, it's now decoupled.
        # The IRCOTStructure itself now uses gemini-1.5-flash-latest by default or what's passed to it.

        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        vector_store = VectorStore(
            embeddings=embeddings, # Pass the Langchain embeddings object
            index_path=Config.FAISS_INDEX_PATH,
            metadata_path=Config.FAISS_METADATA_PATH,
            index_type=Config.FAISS_INDEX_TYPE
        )
        
        # Note: Reranker and search engine excluded for now
        # reranker = Reranker(
        #     llm=llm, # Pass the Langchain LLM object
        #     min_relevance_score=Config.MIN_RELEVANCE_SCORE
        # )
        # 
        # search_engine = SearchEngine(
        #     max_results=Config.MAX_WEB_RESULTS,
        #     rate_limit_delay=1.0
        # )
        
        ir_cot = IRCOTStructure(
            vector_store=vector_store,
            reranker=None,  # Excluded for now
            search_engine=None,  # Excluded for now
            config=Config,
            llm=llm # Pass the shared LLM instance
        )
        
        ir_cot.initialize()
        
        # This print is user-facing, keep it.
        print("✅ System initialized successfully!") 
        
        return {
            'doc_processor': doc_processor,
            'vector_store': vector_store,
            'reranker': None,
            'search_engine': None,
            'ir_cot': ir_cot,
            'llm': llm, # also return the llm if needed elsewhere
            'embeddings': embeddings # also return embeddings if needed
        }
        
    except Exception as e:
        logging.error(f"❌ Error initializing system: {e}", exc_info=True)
        # This print is user-facing for critical failure, keep it.
        print(f"❌ Error initializing system: {e}") 
        # import traceback # Handled by logger with exc_info=True
        # traceback.print_exc()
        return {}

def setup_corpus(components: Dict[str, Any]) -> bool:
    """Setup the corpus in the vector store"""
    # User-facing print, keep it.
    print("\nSetting up corpus...") 
    
    try:
        doc_processor = components['doc_processor']
        vector_store = components['vector_store']
        
        # Check if index already has documents
        collection_info = vector_store.get_collection_info()
        if collection_info.get('document_count', 0) > 0:
            # User-facing print, keep it.
            print(f"✅ Corpus already loaded: {collection_info['document_count']} documents") 
            return True
        
        # Load and process corpus
        if not Config.CORPUS_FILE.exists():
            logging.error(f"❌ Corpus file not found: {Config.CORPUS_FILE}")
            # User-facing print, keep it.
            print(f"❌ Corpus file not found: {Config.CORPUS_FILE}") 
            return False
        
        # User-facing print, keep it.
        print(f"📄 Loading corpus from: {Config.CORPUS_FILE}") 
        documents = doc_processor.parse_corpus_file(str(Config.CORPUS_FILE))
        
        if not documents:
            logging.error("❌ No documents found in corpus file")
            # User-facing print, keep it.
            print("❌ No documents found in corpus file") 
            return False
        
        # User-facing print, keep it.
        print(f"📝 Processing {len(documents)} documents...") 
        chunked_documents = doc_processor.chunk_documents(documents)
        
        # User-facing print, keep it.
        print(f"🔍 Adding {len(chunked_documents)} chunks to vector store...") 
        success = vector_store.add_documents(chunked_documents)
        
        if success:
            # User-facing print, keep it.
            print("✅ Corpus setup completed!") 
            return True
        else:
            logging.error("❌ Failed to add documents to vector store")
            # User-facing print, keep it.
            print("❌ Failed to add documents to vector store") 
            return False
            
    except Exception as e:
        logging.error(f"❌ Error setting up corpus: {e}", exc_info=True)
        # User-facing print, keep it.
        print(f"❌ Error setting up corpus: {e}") 
        return False

def run_query(ir_cot: IRCOTStructure, question: str) -> Dict[str, Any]:
    """Run a single query through the IR-COT system"""
    # This print is user-facing, keep it.
    print(f"\n{'='*60}")
    print(f"🤔 Question: {question}")
    print(f"{'='*60}")
    logging.info(f"Starting query: {question}")
    
    start_time = time.time()
    
    try:
        result = ir_cot.query(question)
        
        # Display results
        # This print is user-facing, keep it.
        print(f"\n💡 Answer: {result['answer']}")
        
        if result.get('reasoning_chain'):
            # This print is user-facing, keep it.
            print(f"\n🧠 Reasoning Chain:")
            for i, step in enumerate(result['reasoning_chain'], 1):
                print(f"  {i}. {step}")
        
        metadata = result.get('metadata', {})
        # This print is user-facing, keep it.
        print(f"\n📊 Statistics:")
        print(f"  • Structure: {metadata.get('structure', 'Unknown')}")
        print(f"  • Iterations: {metadata.get('iterations', 0)}")
        print(f"  • Documents: {metadata.get('total_documents', 0)}")
        print(f"  • Time: {metadata.get('query_time_seconds', 0):.2f}s")
        print(f"  • Termination: {metadata.get('termination_reason', 'Unknown')}")
        
        return result
        
    except Exception as e:
        logging.error(f"❌ Error processing query '{question[:50]}...': {e}", exc_info=True)
        # This print is user-facing for direct user feedback on error for this query
        print(f"❌ Error processing query: {e}") 
        return {
            'question': question,
            'answer': f"Error: {str(e)}",
            'reasoning_chain': [],
            'retrieved_documents': [],
            'metadata': {'error': str(e)}
        }

def load_queries_from_csv(limit: int = None) -> List[str]:
    """Load queries from the CSV file"""
    queries = []
    
    if not Config.QUERIES_FILE.exists():
        logging.error(f"❌ Query CSV file not found: {Config.QUERIES_FILE}")
        # User-facing print if in relevant mode, or handled by caller
        # For now, let the function just log and return empty.
        return queries
    
    try:
        import pandas as pd
        
        # Read CSV file
        df = pd.read_csv(Config.QUERIES_FILE)
        
        # Extract queries from the 'query' column
        if 'query' in df.columns:
            query_list = df['query'].dropna().tolist()
            
            # Apply limit if specified
            if limit:
                query_list = query_list[:limit]
            
            queries = [str(query).strip() for query in query_list if str(query).strip()]
            # This is a user-facing positive feedback, keep as print for relevant modes.
            print(f"✅ Loaded {len(queries)} queries from CSV") 
            
        else:
            logging.error(f"❌ 'query' column not found in CSV. Available: {list(df.columns)}")
            # User-facing print for error, keep it.
            print(f"❌ 'query' column not found in CSV file. Available columns: {list(df.columns)}")             
            
    except ImportError:
        logging.warning("⚠️ pandas not installed. Falling back to basic CSV parsing...")
        # User-facing print for warning, keep it.
        print("❌ pandas not installed. Falling back to basic CSV parsing...")
        # Fallback to basic CSV parsing without pandas
        try:
            doc_processor = DocumentProcessor()
            csv_data = doc_processor.parse_csv_file(str(Config.QUERIES_FILE))
            
            for row in csv_data:
                query = row.get('query')
                if query:
                    queries.append(str(query).strip())
                    if limit and len(queries) >= limit:
                        break
                        
            # User-facing positive feedback, keep as print.
            print(f"✅ Loaded {len(queries)} queries from CSV (fallback method)") 
            
        except Exception as e:
            logging.error(f"❌ Error loading queries from CSV (fallback): {e}", exc_info=True)
            # User-facing print for error, keep it.
            print(f"❌ Error loading queries from CSV: {e}") 
            
    except Exception as e:
        logging.error(f"❌ Error loading queries from CSV (pandas): {e}", exc_info=True)
        # User-facing print for error, keep it.
        print(f"❌ Error loading queries from CSV: {e}") 
    
    return queries

def load_test_queries() -> List[str]:
    """Load test queries from various sources"""
    queries = []
    
    # Try to load from CSV file first
    queries = load_queries_from_csv(limit=10)
    
    # Try to load from JSON file if CSV failed
    if not queries and Config.MULTIHOP_DATA_FILE.exists():
        try:
            doc_processor = DocumentProcessor()
            json_data = doc_processor.parse_json_file(str(Config.MULTIHOP_DATA_FILE))
            
            for item in json_data[:5]:  # Limit to first 5
                question = item.get('question') or item.get('query')
                if question:
                    queries.append(question.strip())
                    
            # User-facing positive feedback, keep as print.
            print(f"✅ Loaded {len(queries)} queries from JSON for testing") 
        except Exception as e:
            logging.error(f"Error loading test queries from JSON: {e}", exc_info=True)
            # User-facing print for error, keep it.
            print(f"Error loading queries from JSON: {e}") 
    
    # Default queries if none found
    if not queries:
        logging.warning("⚠️ No queries found in files for testing, using default test queries")
        # User-facing print for warning, keep it.
        print("⚠️ No queries found in files, using default queries") 
        queries = [
            "What is artificial intelligence and how does it work?",
            "How do solar panels generate electricity?",
            "What are the main causes of climate change?",
            "How does machine learning differ from traditional programming?",
            "What is the process of photosynthesis in plants?"
        ]
    
    return queries

def run_sanity_check(components: Dict[str, Any]):
    """Run first 5 queries from query.csv for sanity check"""
    ir_cot = components['ir_cot']
    
    # This print is user-facing, keep it.
    print(f"\n{'='*60}")
    print("🧪 SANITY CHECK MODE - Running first 5 queries")
    print(f"{'='*60}")
    
    queries = load_queries_from_csv(limit=5)
    if not queries:
        logging.error("❌ No queries found for sanity check")
        # User-facing print, keep it.
        print("❌ No queries found for sanity check") 
        return
    
    results = []
    
    for i, question in enumerate(queries, 1):
        # This print is user-facing, keep it.
        print(f"\n📝 Sanity Check Query {i}/5") 
        result = run_query(ir_cot, question)
        results.append(result)
        
        # Small delay between queries
        time.sleep(1)
    
    print_test_summary(results, "SANITY CHECK")

def run_all_queries(components: Dict[str, Any]):
    """Run all queries from input_test.json and output results to reports/output.json"""
    ir_cot = components['ir_cot']
    
    # This print is user-facing, keep it.
    print(f"\n{'='*60}")
    print("🚀 RUN_ALL MODE - Processing queries from input_test.json")
    print(f"{'='*60}")
    
    # Load queries from input_test.json
    input_file = Config.BASE_DIR / "data" / "input_test.json"
    
    if not input_file.exists():
        logging.error(f"❌ Input file not found: {input_file}")
        print(f"❌ Input file not found: {input_file}")
        return
    
    try:
        # Load JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logging.error("❌ Input JSON should be a list of objects")
            print("❌ Input JSON should be a list of objects")
            return
        
        # Extract queries with ground truth answers
        queries_data = []
        for i, item in enumerate(data):
            if isinstance(item, dict) and 'query' in item:
                queries_data.append({
                    'id': i + 1,
                    'query': item['query'],
                    'ground_truth': item.get('answer', 'N/A')
                })
        
        if not queries_data:
            logging.error("❌ No valid queries found in input_test.json")
            print("❌ No valid queries found in input_test.json")
            return
        
        print(f"📊 Total queries to process: {len(queries_data)}")
        
    except Exception as e:
        logging.error(f"❌ Error loading input_test.json: {e}", exc_info=True)
        print(f"❌ Error loading input_test.json: {e}")
        return
    
    # Process all queries
    results = []
    successful_queries = 0
    failed_queries = 0
    total_processing_time = 0
    
    for i, query_data in enumerate(queries_data, 1):
        query_id = query_data['id']
        question = query_data['query']
        ground_truth = query_data['ground_truth']
        
        print(f"\n📝 Processing Query {i}/{len(queries_data)} (ID: {query_id})")
        print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
        
        try:
            # Process query through IR-COT (disable individual reports for RUN_ALL mode)
            start_time = time.time()
            result = ir_cot.query(question, generate_individual_report=False)
            processing_time = time.time() - start_time
            
            # Extract information from IR-COT result
            answer = result.get('answer', 'No answer generated')
            answer_reasoning = result.get('answer_reasoning', 'No reasoning available')
            fact_list = result.get('reasoning_chain', [])
            metadata = result.get('metadata', {})
            
            # Add processing time to metadata
            metadata['processing_time_seconds'] = processing_time
            
            # Format result according to specification
            formatted_result = {
                'question': question,
                'ground_truth': ground_truth,
                'answer': answer,
                'answer_reasoning': answer_reasoning,
                'fact_list': fact_list,
                'metadata': metadata
            }
            
            results.append(formatted_result)
            successful_queries += 1
            total_processing_time += processing_time
            
            print(f"✅ Query {i} completed in {processing_time:.2f}s")
            print(f"📝 Answer: {answer[:150]}..." if len(answer) > 150 else f"📝 Answer: {answer}")
            print(f"🧠 Facts extracted: {len(fact_list)}")
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"❌ Error processing query {query_id}: {e}", exc_info=True)
            print(f"❌ Query {i} failed: {error_msg}")
            
            # Add error result
            formatted_result = {
                'question': question,
                'ground_truth': ground_truth,
                'answer': f"Error: {error_msg}",
                'answer_reasoning': "Error in processing",
                'fact_list': [],
                'metadata': {
                    'error': error_msg,
                    'processing_time_seconds': 0,
                    'status': 'failed'
                }
            }
            
            results.append(formatted_result)
            failed_queries += 1
        
        # Small delay between queries to prevent overwhelming the system
        time.sleep(0.5)
        
        # Progress update every 10 queries
        if i % 10 == 0:
            print(f"✅ Progress: {i}/{len(queries_data)} queries completed...")
    
    # Save results to reports/output.json
    output_file = Config.BASE_DIR / "reports" / "output.json"
    
    try:
        # Ensure reports directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare final output with summary
        output_data = {
            'run_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'mode': 'RUN_ALL',
                'input_file': str(input_file),
                'output_file': str(output_file),
                'total_queries': len(queries_data),
                'successful_queries': successful_queries,
                'failed_queries': failed_queries,
                'success_rate': (successful_queries / len(queries_data) * 100) if queries_data else 0,
                'total_processing_time_seconds': total_processing_time,
                'average_processing_time_seconds': (total_processing_time / successful_queries) if successful_queries > 0 else 0
            },
            'configuration': {
                'remove_sources_enable': getattr(Config, 'REMOVE_SOURCES_ENABLE', True),
                'expansion_factor': getattr(Config, 'EXPANSION_FACTOR', 5),
                'no_repeat_retrieve_enable': getattr(Config, 'NO_REPEAT_RETRIEVE_ENABLE', False),
                'rerank_enable': getattr(Config, 'RERANK_ENABLE', True),
                'max_iterations': getattr(Config, 'MAX_ITERATIONS', 4),
                'top_k_documents': getattr(Config, 'TOP_K_DOCUMENTS', 5)
            },
            'results': results
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        logging.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logging.error(f"❌ Error saving results to {output_file}: {e}", exc_info=True)
        print(f"❌ Error saving results: {e}")
    
    # Print summary
    print_run_all_summary(successful_queries, failed_queries, len(queries_data), total_processing_time)

def print_run_all_summary(successful_queries: int, failed_queries: int, total_queries: int, total_time: float):
    """Print summary of RUN_ALL results"""
    print(f"\n{'='*60}")
    print(f"📊 RUN_ALL Summary")
    print(f"{'='*60}")
    
    print(f"• Total queries processed: {total_queries}")
    print(f"• Successful queries: {successful_queries}")
    print(f"• Failed queries: {failed_queries}")
    print(f"• Success rate: {(successful_queries/total_queries*100):.1f}%" if total_queries > 0 else "• Success rate: 0%")
    print(f"• Total processing time: {total_time:.2f}s")
    print(f"• Average time per query: {(total_time/successful_queries):.2f}s" if successful_queries > 0 else "• Average time per query: 0s")
    print(f"• Output saved to: reports/output.json")

def print_test_summary(results: List[Dict], mode_name: str):
    """Print summary of test results"""
    # This print is user-facing, keep it.
    print(f"\n{'='*60}")
    print(f"📊 {mode_name} Summary")
    print(f"{'='*60}")
    
    if not results:
        logging.error("❌ No results to summarize")
        # User-facing print for error, keep it.
        print("❌ No results to summarize") 
        return
    
    total_docs = sum(len(r.get('retrieved_documents', [])) for r in results)
    total_time = sum(r.get('metadata', {}).get('query_time_seconds', 0) for r in results)
    
    # This print is user-facing, keep it.
    print(f"• Total queries: {len(results)}")
    print(f"• Total documents retrieved: {total_docs}")
    print(f"• Average documents per query: {total_docs/len(results):.1f if results else 0:.1f}")
    print(f"• Total time: {total_time:.2f}s")
    print(f"• Average time per query: {total_time/len(results):.2f if results else 0:.2f}s")
    
    # Count successful vs error queries
    errors = sum(1 for r in results if 'error' in r.get('metadata', {}))
    successful = len(results) - errors
    
    # This print is user-facing, keep it.
    print(f"• Successful queries: {successful}")
    print(f"• Failed queries: {errors}")
    
    if errors > 0 and results:
        # This print is user-facing, keep it.
        print(f"• Success rate: {(successful/len(results)*100):.1f}%")

def save_results_to_file(results: List[Dict]):
    """Save results to a JSON file"""
    try:
        output_file = Config.BASE_DIR / "results" / f"ir_cot_results_{int(time.time())}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"💾 Results saved to: {output_file}")
        # User-facing print for success, keep it.
        print(f"💾 Results saved to: {output_file}") 
        
    except Exception as e:
        logging.error(f"❌ Error saving results to file: {e}", exc_info=True)
        # User-facing print for error, keep it.
        print(f"❌ Error saving results: {e}") 

def interactive_mode(components: Dict[str, Any]):
    """Run interactive mode for user queries"""
    ir_cot = components['ir_cot']
    
    # This print is user-facing, keep it.
    print(f"\n{'='*60}")
    print("🚀 Interactive Multi-hop RAG System (IR-COT)")
    print("Type 'quit' to exit, 'test' to run test queries, 'stats' for performance summary")
    print(f"{'='*60}")
    
    while True:
        try:
            question = input("\n❓ Enter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'test':
                run_test_queries(components)
                continue
            elif question.lower() == 'stats':
                print(f"\n📊 System Statistics:")
                performance = ir_cot.get_performance_summary()
                for key, value in performance.items():
                    print(f"  • {key}: {value}")
                continue
            elif not question:
                continue
            logging.info(f"User Query: {question}")
            result = run_query(ir_cot, question)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logging.error(f"❌ Error in interactive loop: {e}", exc_info=True)
            print(f"❌ An unexpected error occurred: {e}") # Keep user-facing print

def run_test_queries(components: Dict[str, Any]):
    """Run a set of test queries"""
    ir_cot = components['ir_cot']
    
    # This print is user-facing, keep it.
    print(f"\n{'='*60}")
    print("🧪 Running Test Queries")
    print(f"{'='*60}")
    
    test_queries = load_test_queries()
    results = []
    
    for i, question in enumerate(test_queries, 1):
        # This print is user-facing, keep it.
        print(f"\n📝 Test Query {i}/{len(test_queries)}") 
        result = run_query(ir_cot, question)
        results.append(result)
        
        # Small delay between queries
        time.sleep(1)
    
    print_test_summary(results, "TEST")

def main():
    """Main entry point"""
    # --- Centralized Logging Setup from previous step ---
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Custom filter to exclude ERROR and CRITICAL from file log
    class NoErrorFilter(logging.Filter):
        def filter(self, record):
            return record.levelno < logging.ERROR # Allows levels below ERROR (e.g., INFO, WARNING)

    file_handler = logging.FileHandler(Config.LOG_FILE, mode='a')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO) # File handler still processes INFO and above initially
    file_handler.addFilter(NoErrorFilter()) # Apply the custom filter

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.handlers = [] 
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)
    
    logging.info("🔥 Multi-hop RAG System with IR-COT Structure. Logging to console (INFO+) and %s (DEBUG+).", Config.LOG_FILE)
    logging.info("=" * 50)
    
    # Display current mode
    logging.info(f"🎯 Current MODE: {Config.MODE}")
    
    components = initialize_system()
    if not components:
        # logging.error already called in initialize_system if it fails. 
        # The print there is also user-facing for this critical failure.
        return 
    
    # Setup corpus
    if not setup_corpus(components):
        # logging.error already called in setup_corpus if it fails.
        # The print there is also user-facing.
        logging.critical("❌ System cannot proceed without a valid corpus. Exiting.") # Add critical log.
        print("❌ System cannot proceed without a valid corpus. Exiting.") # Keep critical user-facing print.
        return
    
    # Execute based on mode
    if Config.MODE == "SANITY_CHECK":
        run_sanity_check(components)
    elif Config.MODE == "RUN_ALL":
        run_all_queries(components)
    elif Config.MODE == "INTERACTIVE":
        # Check command line arguments for backward compatibility
        if len(sys.argv) > 1:
            if sys.argv[1] == 'test':
                run_test_queries(components)
            elif sys.argv[1] == 'query' and len(sys.argv) > 2:
                question = ' '.join(sys.argv[2:])
                logging.info(f"User Query: {question}")
                run_query(components['ir_cot'], question)
            else:
                print("Usage: python main.py [test|query 'your question']") 
        else:
            # Interactive mode
            interactive_mode(components)
    else:
        logging.error(f"❌ Unknown MODE: {Config.MODE}. Valid modes: SANITY_CHECK, RUN_ALL, INTERACTIVE")
        print(f"❌ Unknown MODE: {Config.MODE}") 
        print("Valid modes: SANITY_CHECK, RUN_ALL, INTERACTIVE")
        return

if __name__ == "__main__":
    main() 