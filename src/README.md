# IR-COT Multi-hop RAG System

A simplified, clean implementation of **IR-COT (Interleaved Retrieval Chain-of-Thought)** multi-hop RAG system. This system follows the IR-COT specification for alternating between reasoning and retrieval steps to answer complex questions.

## 🏗️ Architecture

The system follows a clean, modular architecture with:

```
src/
├── main.py                    # Simple entry point
├── config.py                  # Central configuration
├── prompts.py                 # All prompt templates
├── multihop_structure.py      # Abstract base class
├── ir_cot_structure.py        # IR-COT implementation
└── utils/                     # General API modules
    ├── document_processor.py  # Document handling
    ├── vector_store.py        # Vector storage & retrieval
    ├── reranker.py            # Document relevance scoring
    └── search_engine.py       # Web search capabilities
```

## 🎯 Key Features

- **Pure IR-COT Implementation**: Follows the exact IR-COT specification
- **Clean Architecture**: Single multi-hop structure with modular utilities
- **General APIs**: All utils serve as reusable APIs for any multi-hop RAG
- **Simple Configuration**: Centralized config and prompts
- **No Complexity**: Removed all unnecessary architecture layers

## 📋 Requirements

- Python 3.8+
- Google API key for Gemini models
- Internet connection for web search (optional)

## 🛠️ Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment**:
```bash
export GOOGLE_API_KEY="your_google_api_key_here"
```

3. **Prepare data files** (place in `../data/`):
   - `multihoprag_corpus_exp.txt` - Your document corpus
   - `query.csv` - Test queries (optional)
   - `MultiHopRAG_exp.json` - Additional query data (optional)

## 🚀 Usage

### Interactive Mode
```bash
python main.py
```

### Run Test Queries
```bash
python main.py test
```

### Single Query
```bash
python main.py query "Your question here"
```

## 🔄 IR-COT Process Flow

The system implements the exact IR-COT specification:

1. **Initial Retrieval**: Use the original question to retrieve initial documents
2. **Iterative Loop**: Alternate between reasoning and retrieval:
   - **Reason**: Generate next CoT sentence based on current documents
   - **Retrieve**: Use CoT sentence as query to get more documents
   - **Accumulate**: Add new documents while avoiding duplicates
3. **Termination**: Stop when answer is found or max iterations reached
4. **Final Answer**: Generate answer using complete CoT and all documents

### Example Flow

```
Question: "In what country was Lost Gravity manufactured?"

Initial Retrieval → 10 documents about Lost Gravity

Iteration 1:
├─ Reason: "Lost Gravity was manufactured by Mack Rides."
└─ Retrieve: Find documents about "Mack Rides" → 8 new documents

Iteration 2:
├─ Reason: "Mack Rides is a company from Germany."
└─ Retrieve: Find documents about "Mack Rides Germany" → 5 new documents

Iteration 3:
├─ Reason: "The answer is Germany."
└─ Termination: Answer phrase detected

Final Answer: "Lost Gravity was manufactured in Germany..."
```

## ⚙️ Configuration

### Core Settings (`config.py`)

```python
# File paths
CORPUS_FILE = "data/multihoprag_corpus_exp.txt"
QUERIES_FILE = "data/query.csv"
MULTIHOP_DATA_FILE = "data/MultiHopRAG_exp.json"

# Model settings
LLM_MODEL = "models/gemini-2.5-flash-preview-05-20"
EMBEDDING_MODEL = "models/embedding-001"

# IR-COT parameters
MAX_ITERATIONS = 5
TOP_K_DOCUMENTS = 10
SIMILARITY_THRESHOLD = 0.7

# Optional features
WEB_SEARCH_ENABLED = True
MAX_WEB_RESULTS = 5
```

### Prompt Templates (`prompts.py`)

All prompts are centralized and easily customizable:

- `IR_COT_REASONING_PROMPT` - Generate next reasoning step
- `COT_GUIDED_RETRIEVAL_PROMPT` - Create retrieval queries
- `FINAL_ANSWER_PROMPT` - Generate final answer
- General prompts for document scoring, evaluation, etc.

## 🧩 Module APIs

### VectorStore API
```python
vector_store = VectorStore(
    embeddings=embeddings,
    index_path="./faiss_db/index.faiss",
    metadata_path="./faiss_db/metadata.pkl",
    index_type="IndexFlatIP"
)
results = vector_store.similarity_search(query="question", top_k=10)
```

### Reranker API
```python
reranker = Reranker(model_name="gemini-pro")
ranked_docs = reranker.rerank_documents(query, documents, top_k=5)
```

### SearchEngine API
```python
search_engine = SearchEngine(max_results=5)
web_results = search_engine.search(query="question", search_type="general")
```

### DocumentProcessor API
```python
processor = DocumentProcessor(chunk_size=512)
documents = processor.parse_corpus_file("corpus.txt")
chunks = processor.chunk_documents(documents)
```

## 📊 Input/Output Format

### Input Files

1. **Corpus File** (`multihoprag_corpus_exp.txt`):
   ```
   Document 1 content here...
   
   Document 2 content here...
   ```

2. **Query CSV** (`query.csv`):
   ```csv
   question,expected_answer
   "What is...?","The answer is..."
   ```

3. **MultiHop JSON** (`MultiHopRAG_exp.json`):
   ```json
   [{"question": "What is...?", "answer": "..."}]
   ```

### Output Format

```python
{
    "question": "Original question",
    "answer": "Generated answer",
    "reasoning_chain": ["Step 1", "Step 2", "..."],
    "retrieved_documents": [{"content": "...", "metadata": {...}}],
    "metadata": {
        "structure": "IR-COT",
        "iterations": 3,
        "total_documents": 25,
        "query_time_seconds": 12.5,
        "termination_reason": "Answer phrase detected"
    }
}
```

## 🔧 Extending the System

### Adding New Multi-hop Structures

1. Create new class inheriting from `MultiHopRagStructure`
2. Implement the `query()` method
3. Use existing util APIs (vector_store, reranker, etc.)

```python
class NewStructure(MultiHopRagStructure):
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        # Your implementation using self.vector_store, self.reranker, etc.
        pass
```

### Customizing Prompts

Simply modify the templates in `prompts.py`:

```python
PromptTemplates.IR_COT_REASONING_PROMPT = """
Your custom reasoning prompt here...
{question}
{documents}
{previous_cot}
"""
```

## 🔍 System Components

### Abstract Base Class
- `MultiHopRagStructure` - Defines interface for all structures
- Provides common utilities and performance tracking
- Ensures consistent API across implementations

### IR-COT Implementation
- `IRCOTStructure` - Complete IR-COT specification implementation
- Alternates between reasoning and retrieval steps
- Accumulates documents with duplicate detection
- Terminates on answer detection or max iterations

### General Utilities
- **DocumentProcessor** - Parse, chunk, and format documents
- **VectorStore** - FAISS integration with Google embeddings
- **Reranker** - LLM-powered document relevance scoring
- **SearchEngine** - Web search with rate limiting and fallbacks

## 🎮 Interactive Commands

When running interactively:

- Type any question to query the system
- `test` - Run test queries from data files
- `stats` - Show system performance statistics
- `quit` - Exit the system

## 🐛 Troubleshooting

### Common Issues

1. **Google API Key Error**:
   ```
   export GOOGLE_API_KEY="your_key_here"
   ```

2. **Data Files Not Found**:
   Ensure files are in `../data/` relative to `src/`:
   ```
   project/
   ├── data/
   │   ├── multihoprag_corpus_exp.txt
   │   ├── query.csv
   │   └── MultiHopRAG_exp.json
   └── src/
       └── main.py
   ```

3. **FAISS Index Issues**:
   Delete `./faiss_db` directory and restart

### Performance Tips

- Adjust `TOP_K_DOCUMENTS` for speed vs. accuracy trade-off
- Set `WEB_SEARCH_ENABLED = False` for corpus-only queries
- Reduce `MAX_ITERATIONS` for faster responses
- Increase `SIMILARITY_THRESHOLD` to filter low-quality documents
- Consider advanced FAISS index types (IVF, HNSW) for large datasets

## 📝 License

This project is open source and available under the MIT License.

## 🎯 Design Philosophy

This implementation prioritizes:

- **Simplicity**: Single multi-hop structure, clean APIs
- **Modularity**: Reusable components for any RAG system
- **Clarity**: Easy to understand and extend
- **Specification Compliance**: True to IR-COT research paper
- **Practical Usage**: Ready-to-run with real data
