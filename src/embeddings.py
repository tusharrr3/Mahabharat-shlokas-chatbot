"""
Embeddings module for Mahabharat RAG Chatbot
Handles vector store creation and similarity search using FAISS
"""

import os
import logging
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_embeddings_model(model_name: str, device: str = "cpu") -> HuggingFaceEmbeddings:
    """
    Create HuggingFace embeddings model.
    
    Args:
        model_name: Name of the sentence-transformers model
        device: Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    try:
        logger.info(f"Loading embedding model: {model_name}")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity scores
        )
        logger.info("Embedding model loaded successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise


def create_vector_store(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings
) -> FAISS:
    """
    Create FAISS vector store from documents.
    
    Args:
        documents: List of LangChain Document objects
        embeddings: HuggingFace embeddings model
        
    Returns:
        FAISS vector store instance
    """
    try:
        logger.info(f"Creating vector store from {len(documents)} documents")
        vector_store = FAISS.from_documents(documents, embeddings)
        logger.info("Vector store created successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise


def save_vector_store(vector_store: FAISS, save_path: str) -> None:
    """
    Save FAISS vector store to disk.
    
    Args:
        vector_store: FAISS vector store instance
        save_path: Directory path to save the vector store
    """
    try:
        os.makedirs(save_path, exist_ok=True)
        vector_store.save_local(save_path)
        logger.info(f"Vector store saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving vector store: {e}")
        raise


def load_vector_store(
    load_path: str,
    embeddings: HuggingFaceEmbeddings
) -> FAISS:
    """
    Load FAISS vector store from disk.
    
    Args:
        load_path: Directory path to load the vector store from
        embeddings: HuggingFace embeddings model
        
    Returns:
        FAISS vector store instance
    """
    try:
        logger.info(f"Loading vector store from {load_path}")
        vector_store = FAISS.load_local(
            load_path,
            embeddings,
            allow_dangerous_deserialization=True  # Required for loading FAISS
        )
        logger.info("Vector store loaded successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise


def similarity_search_with_scores(
    vector_store: FAISS,
    query: str,
    k: int = 3
) -> List[Tuple[Document, float]]:
    """
    Perform similarity search and return documents with scores.
    
    Args:
        vector_store: FAISS vector store instance
        query: User query string
        k: Number of results to retrieve
        
    Returns:
        List of tuples containing (Document, similarity_score)
        Note: FAISS returns L2 distance, lower scores indicate better matches
    """
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        logger.info(f"Retrieved {len(results)} results for query: '{query[:50]}...'")
        return results
    except Exception as e:
        logger.error(f"Error performing similarity search: {e}")
        raise


def initialize_vector_store(
    documents: List[Document],
    model_name: str,
    device: str = "cpu",
    save_path: str = None
) -> Tuple[FAISS, HuggingFaceEmbeddings]:
    """
    Initialize vector store with documents and embeddings.
    
    This is a convenience function that combines model creation,
    vector store creation, and optional saving.
    
    Args:
        documents: List of LangChain Document objects
        model_name: Name of the sentence-transformers model
        device: Device to run the model on ('cpu' or 'cuda')
        save_path: Optional path to save the vector store
        
    Returns:
        Tuple of (FAISS vector store, embeddings model)
    """
    embeddings = create_embeddings_model(model_name, device)
    vector_store = create_vector_store(documents, embeddings)
    
    if save_path:
        save_vector_store(vector_store, save_path)
    
    return vector_store, embeddings
