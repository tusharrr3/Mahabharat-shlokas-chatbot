"""
Configuration file for Mahabharat RAG Chatbot
Contains all configurable parameters for the system
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

# Data paths  
MAHABHARAT_JSON_PATH = os.path.join(DATA_DIR, "Bhagvad_gita_rag.json")  # Using Bhagavad Gita RAG JSON

# Embedding model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # Change to "cuda" if GPU is available

# Vector store configuration
VECTOR_STORE_TYPE = "FAISS"
TOP_K_RESULTS = 8  # Increased for multi-chapter retrieval

# Chunking configuration (group consecutive verses)
CHUNK_SIZE = 3  # Number of consecutive verses per chunk
CHUNK_OVERLAP = 1  # Overlap between chunks for context continuity

# Similarity threshold configuration
# Documents with similarity score BELOW this threshold will be considered relevant
# FAISS returns L2 distance, lower is better
# Threshold of 1.2 provides optimal balance: accepts BG questions, rejects irrelevant ones
SIMILARITY_THRESHOLD = 1.2  # Optimal threshold: rejects out-of-context, accepts BG questions

# Debug configuration
DEBUG_MODE = True  # Print retrieval details (documents, scores, chapters)

# Response configuration
ANSWER_NOT_FOUND_MESSAGE = "Answer not found"

# LLM Configuration
USE_LLM_GENERATION = True  # Set to False to disable LLM responses
LLM_TYPE = "gemini"  # Options: "gemini", "openai", "other"
LLM_MODEL = "gemini-2.5-flash-lite-preview-09-2025"  # Google Gemini model
LLM_TEMPERATURE = 0.3  # Lower temperature for more factual responses
LLM_MAX_TOKENS = 2000  # Increased for detailed comparisons
STRICT_GROUNDING = True  # Prevent hallucination - only use retrieved context

# Logging configuration
LOG_LEVEL = "INFO"
