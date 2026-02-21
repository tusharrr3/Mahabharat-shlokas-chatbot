

import os
import sys
import json
import logging
from typing import Optional

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    MAHABHARAT_JSON_PATH,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    ANSWER_NOT_FOUND_MESSAGE,
    USE_LLM_GENERATION,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEBUG_MODE,
    STRICT_GROUNDING
)
from load_data import load_and_prepare_documents
from embeddings import (
    create_embeddings_model,
    create_vector_store,
    save_vector_store,
    load_vector_store
)
from rag_pipeline import MahabharatRAGPipeline
from gemini_integration import GeminiLLMIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MahabharatChatbot:
    """
    Main chatbot class that orchestrates the entire RAG system.
    """
    
    def __init__(self):
        """Initialize the chatbot by setting up vector store and pipeline."""
        self.vector_store = None
        self.embeddings = None
        self.pipeline = None
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        """
        Initialize or load the RAG system.
        
        Checks if vector store exists, loads it if available,
        otherwise creates a new one from JSON data.
        """
        try:
            # Check if vector store already exists
            if os.path.exists(VECTOR_STORE_DIR):
                logger.info("Found existing vector store, loading...")
                self._load_existing_vector_store()
            else:
                logger.info("No existing vector store found, creating new one...")
                self._create_new_vector_store()
            
            # Initialize LLM if enabled
            if USE_LLM_GENERATION:
                logger.info("Initializing Gemini LLM...")
                self.llm = GeminiLLMIntegration(
                    model=LLM_MODEL
                )
                if not self.llm.is_available():
                    logger.warning("LLM not available, proceeding without LLM responses")
                    self.llm = None
                else:
                    logger.info("Gemini LLM integration enabled")
            
            # Initialize RAG pipeline
            self.pipeline = MahabharatRAGPipeline(
                vector_store=self.vector_store,
                similarity_threshold=SIMILARITY_THRESHOLD,
                top_k=TOP_K_RESULTS,
                answer_not_found_message=ANSWER_NOT_FOUND_MESSAGE,
                llm_integration=self.llm,
                llm_temperature=LLM_TEMPERATURE,
                llm_max_tokens=LLM_MAX_TOKENS,
                debug_mode=DEBUG_MODE,
                strict_grounding=STRICT_GROUNDING
            )
            
            logger.info("Chatbot initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            raise
    
    def _load_existing_vector_store(self):
        """Load existing vector store from disk."""
        try:
            self.embeddings = create_embeddings_model(EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE)
            self.vector_store = load_vector_store(VECTOR_STORE_DIR, self.embeddings)
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            logger.info("Falling back to creating new vector store...")
            self._create_new_vector_store()
    
    def _create_new_vector_store(self):
        """Create new vector store from JSON data."""
        # Check if JSON file exists
        if not os.path.exists(MAHABHARAT_JSON_PATH):
            raise FileNotFoundError(
                f"Mahabharat JSON file not found at {MAHABHARAT_JSON_PATH}\n"
                f"Please place your mahabharat.json file in the data/ directory."
            )
        
        # Load and prepare documents
        logger.info(f"Loading documents from JSON (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
        documents = load_and_prepare_documents(
            MAHABHARAT_JSON_PATH,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Create embeddings model
        logger.info("Creating embeddings model...")
        self.embeddings = create_embeddings_model(EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE)
        
        # Create vector store
        logger.info("Creating vector store (this may take a few moments)...")
        self.vector_store = create_vector_store(documents, self.embeddings)
        
        # Save vector store for future use
        logger.info("Saving vector store...")
        save_vector_store(self.vector_store, VECTOR_STORE_DIR)
        
        logger.info("Vector store created and saved successfully!")
    
    def query(self, user_query: str) -> dict:
        """
        Query the chatbot.
        
        Args:
            user_query: User's question
            
        Returns:
            Structured response dictionary
        """
        if not self.pipeline:
            raise RuntimeError("Chatbot not initialized properly")
        
        return self.pipeline.query(user_query)
    
    def get_stats(self) -> dict:
        """Get chatbot statistics."""
        if not self.pipeline:
            return {}
        return self.pipeline.get_stats()


def print_response(response: dict):
    """
    Print the response in a formatted JSON structure.
    
    Args:
        response: Response dictionary from the chatbot
    """
    print("\n" + "="*80)
    print("RESPONSE:")
    print("="*80)
    print(json.dumps(response, indent=2, ensure_ascii=False))
    print("="*80 + "\n")


def interactive_mode(chatbot: MahabharatChatbot):
    """
    Run the chatbot in interactive CLI mode.
    
    Args:
        chatbot: Initialized MahabharatChatbot instance
    """
    print("\n" + "="*80)
    print("MAHABHARAT RAG CHATBOT")
    print("="*80)
    print("\nWelcome! Ask me anything about the Mahabharat.")
    print("Type 'quit', 'exit', or 'q' to exit.")
    print("Type 'stats' to see system statistics.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using Mahabharat RAG Chatbot. Goodbye!\n")
                break
            
            # Check for stats command
            if user_input.lower() == 'stats':
                stats = chatbot.get_stats()
                print_response(stats)
                continue
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Query the chatbot
            response = chatbot.query(user_input)
            
            # Print response
            print_response(response)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!\n")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\nError: {e}\n")


def main():
    """Main function to run the chatbot."""
    try:
        # Initialize chatbot
        print("\nInitializing Mahabharat RAG Chatbot...")
        print("This may take a moment on first run...\n")
        
        chatbot = MahabharatChatbot()
        
        # Run interactive mode
        interactive_mode(chatbot)
        
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\nError: {e}\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nFatal error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
