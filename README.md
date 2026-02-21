# Mahabharat RAG Chatbot

A production-ready Retrieval Augmented Generation (RAG) chatbot built with Python that answers questions about the Mahabharat/Bhagavad Gita using semantic search with AI-powered insights via Grok LLM.

## Features

- 🔍 **Semantic Search**: Uses sentence-transformers for accurate semantic understanding
- 🤖 **Grok LLM Integration**: AI-powered analysis and insights on retrieved verses
- 💾 **FAISS Vector Store**: Efficient similarity search with persistent storage
- 🎯 **Hallucination Prevention**: Threshold-based filtering ensures only relevant answers
- 🏗️ **Clean Architecture**: Modular design with separation of concerns
- 🛡️ **Production Ready**: Error handling, logging, and input validation
- 💬 **Dual Interface**: Beautiful Streamlit web UI + CLI interface
- 📊 **Structured Responses**: JSON-formatted outputs with confidence scores
- 📜 **Chat History**: Track previous queries and responses
- 📥 **Export Results**: Download responses as JSON files
- 📍 **Exact Verse Lookup**: Search directly by chapter and verse numbers (e.g., "chapter 2 verse 47")

## Project Structure

```
mahabharat-rag/
│
├── data/
│   ├── Bhagvad_gita_rag.json    # RAG dataset (701 verses)
│   └── vector_store/            # Cached FAISS embeddings (auto-generated)
│
├── src/
│   ├── config.py                # Configuration settings
│   ├── load_data.py             # Data loading and document preparation
│   ├── embeddings.py            # Vector store creation and management
│   ├── rag_pipeline.py          # Core RAG logic and retrieval
│   ├── llm_integration.py       # Grok LLM integration
│   └── main.py                  # CLI interface
│
├── app.py                       # Streamlit web interface
├── GROK_SETUP.md                # LLM setup instructions
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the project**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

For GPU support (optional, for faster embeddings):
```bash
# Replace faiss-cpu with faiss-gpu in requirements.txt
pip install faiss-gpu
```

3. **Prepare your data**

**Option A: If you already have a custom JSON format (like `output.json`):**

Run the transformation script to convert it to the expected format:

```bash
python src/transform_data.py
```

This will automatically convert `data/output.json` to `data/mahabharat.json`.

**Option B: Create your own `mahabharat.json` manually:**

Create a `mahabharat.json` file in the `data/` directory with the following format:

```json
[
  {
    "verse_number": "1.1.1",
    "shlok": "धृतराष्ट्र उवाच...",
    "meaning": "Explanation in Hindi or English",
    "translation": "English translation"
  },
  {
    "verse_number": "1.1.2",
    "shlok": "...",
    "meaning": "...",
    "translation": "..."
  }
]
```

> **Note:** Your `output.json` has already been transformed successfully, containing 325 verses ready to use!

## LLM Integration with Grok

This chatbot includes optional integration with **Grok LLM** for AI-powered analysis and insights.

### Enabling Grok LLM

1. **Get your API key**:
   - Visit [https://console.x.ai/](https://console.x.ai/)
   - Create an account and generate an API key
   - Copy your API key

2. **Set up environment variable**:

   **Windows (PowerShell)**:
   ```powershell
   $env:GROK_API_KEY = "your-api-key-here"
   ```

   **macOS/Linux**:
   ```bash
   export GROK_API_KEY="your-api-key-here"
   ```

3. **Run the chatbot**:
   ```bash
   streamlit run app.py
   ```

   When the API key is set, you'll see detailed insights under "💡 Analysis & Insights" section with each query response.

### LLM Features

- **Intelligent Analysis**: Get AI-powered explanations of verses
- **Context Integration**: LLM synthesizes multiple verses for complex questions
- **Spiritual Insights**: Deep explanations of Bhagavad Gita teachings
- **Natural Language**: Responses are naturally written and easy to understand

### Disabling LLM

To run without LLM (just semantic search):
- Don't set the `GROK_API_KEY` environment variable
- Or set `USE_LLM_GENERATION = False` in `src/config.py`

For detailed setup instructions, see [GROK_SETUP.md](GROK_SETUP.md)

## Usage

### Option 1: Web Interface (Recommended)

Run the beautiful Streamlit web interface:

```bash
streamlit run app.py
```

This will open a browser window with a user-friendly interface featuring:
- 🎨 Beautiful, intuitive UI
- 💬 Chat-like interface with history
- 📊 Real-time statistics
- 🔍 Quick query buttons
- 📥 Download responses as JSON
- 📱 Responsive design

### Option 2: Command Line Interface

Run the traditional CLI:

```bash
python src/main.py
```

On first run, the system will:
1. Load the JSON data
2. Generate embeddings (may take a few minutes)
3. Create and save the FAISS vector store
4. Start the interactive CLI

Subsequent runs will load the cached vector store instantly.

### CLI Commands

- **Ask a question**: Simply type your query and press Enter
- **View statistics**: Type `stats`
- **Exit**: Type `quit`, `exit`, or `q`

### Example Interaction

```
You: What does Krishna say about dharma?

================================================================================
RESPONSE:
================================================================================
{
  "verse_number": "2.31",
  "shlok": "स्वधर्ममपि चावेक्ष्य न विकम्पितुमर्हसि...",
  "meaning": "Considering your specific duty as a warrior...",
  "translation": "Considering your duty, you should not waver...",
  "confidence_score": 0.8542
}
================================================================================
```

If no relevant answer is found:

```json
{
  "answer": "Answer not found"
}
```

## Configuration

Edit `src/config.py` to customize:

- **SIMILARITY_THRESHOLD**: Controls relevance filtering (default: 1.5)
  - Lower values = stricter matching
  - Higher values = more lenient matching
  
- **TOP_K_RESULTS**: Number of results to retrieve (default: 3)

- **EMBEDDING_MODEL_NAME**: HuggingFace model to use (default: `sentence-transformers/all-MiniLM-L6-v2`)

- **EMBEDDING_DEVICE**: Set to `"cuda"` for GPU acceleration (default: `"cpu"`)

## How It Works

### Architecture

1. **Data Loading** (`load_data.py`)
   - Loads JSON data
   - Validates required fields
   - Converts to LangChain Document objects

2. **Embeddings** (`embeddings.py`)
   - Creates sentence embeddings using HuggingFace models
   - Builds FAISS vector store for efficient similarity search
   - Caches embeddings to disk for fast loading

3. **RAG Pipeline** (`rag_pipeline.py`)
   - Performs semantic similarity search
   - Applies threshold filtering to prevent hallucinations
   - Generates structured responses with confidence scores

4. **Main Application** (`main.py`)
   - Orchestrates all components
   - Provides interactive CLI
   - Handles errors and logging

### Hallucination Prevention

The system prevents hallucinations through:

- **Similarity Threshold**: Only returns results with similarity scores below the threshold
- **No Generative Model**: Uses pure retrieval without LLM generation
- **Structured Responses**: Returns exact data from the knowledge base
- **Confidence Scores**: Provides transparency about match quality

## API Usage (Programmatic)

You can also use the chatbot programmatically:

```python
from main import MahabharatChatbot

# Initialize chatbot
chatbot = MahabharatChatbot()

# Query
response = chatbot.query("What is dharma?")
print(response)

# Get statistics
stats = chatbot.get_stats()
print(stats)
```

## Troubleshooting

### Issue: "JSON file not found"
**Solution**: Ensure `mahabharat.json` exists in the `data/` directory

### Issue: Slow embeddings creation
**Solution**: 
- Use GPU by setting `EMBEDDING_DEVICE = "cuda"` in config.py
- Use a smaller embedding model
- Reduce the size of your dataset

### Issue: Too many "Answer not found" responses
**Solution**: Increase `SIMILARITY_THRESHOLD` in config.py (e.g., from 1.5 to 2.0)

### Issue: Irrelevant results
**Solution**: Decrease `SIMILARITY_THRESHOLD` in config.py (e.g., from 1.5 to 1.0)

## Performance Optimization

- **Cached Vector Store**: After first run, loads instantly from disk
- **Efficient Search**: FAISS provides sub-linear search complexity
- **Normalized Embeddings**: Faster cosine similarity computation
- **Batch Processing**: Support for batch queries

## Dependencies

- **langchain**: LLM application framework
- **faiss-cpu**: Efficient similarity search
- **sentence-transformers**: Semantic embeddings
- **torch**: PyTorch for model inference
- **transformers**: HuggingFace model integration

## License

This project is open-source and available for educational and commercial use.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Future Enhancements

- [ ] Web UI using Streamlit or Gradio
- [ ] REST API with FastAPI
- [ ] Multiple language support
- [ ] Advanced filtering (by chapter, book, character)
- [ ] Query expansion for better retrieval
- [ ] Hybrid search (keyword + semantic)

## Support

For questions or issues, please open an issue on the project repository.
