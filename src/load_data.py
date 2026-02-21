"""
Data loading module for Mahabharat RAG Chatbot
Handles loading JSON data and converting to LangChain Documents
Supports chunking: groups consecutive verses for better multi-verse retrieval
"""

import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json_data(json_path: str) -> List[Dict[str, Any]]:
    """
    Load Mahabharat data from JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing verse data
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} verses from {json_path}")
        return data
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading JSON data: {e}")
        raise


def convert_to_documents(data: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert JSON data to LangChain Document objects.
    
    Each document combines all fields into page_content for better retrieval,
    and stores structured data in metadata for response generation.
    
    Args:
        data: List of dictionaries containing verse data (from Bhagvad_gita_rag.json)
        
    Returns:
        List of LangChain Document objects
    """
    documents = []
    
    for entry in data:
        try:
            # Extract text field (contains Shloka, Transliteration, Meanings, Commentary)
            page_content = entry.get("text", "")
            metadata_dict = entry.get("metadata", {})
            
            if not page_content:
                logger.warning("Skipping entry with missing text field")
                continue
            
            # Extract metadata
            verse_id = metadata_dict.get("id", "Unknown")
            chapter = metadata_dict.get("chapter", "Unknown")
            verse = metadata_dict.get("verse", "Unknown")
            
            # Create verse number string
            if chapter != "Unknown" and verse != "Unknown":
                verse_number = f"{chapter}.{verse}"
            else:
                verse_number = verse_id
            
            # Validate that we have essential data
            if not page_content:
                logger.warning("Skipping entry with missing page content")
            
            # Store structured data in metadata for easy access
            metadata = {
                "verse_number": verse_number,
                "verse_id": verse_id,
                "chapter": chapter,
                "verse": verse
            }
            
            # Create LangChain Document
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)
            
        except Exception as e:
            logger.warning(f"Error processing entry: {e}")
            continue
    
    logger.info(f"Successfully converted {len(documents)} entries to documents")
    return documents


def load_and_prepare_documents(json_path: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[Document]:
    """
    Load JSON data and prepare LangChain documents.
    
    This is the main function to use for loading data.
    Supports optional chunking for better multi-verse retrieval.
    
    Args:
        json_path: Path to the JSON file
        chunk_size: Number of consecutive verses per chunk (None = no chunking)
        chunk_overlap: Number of verses to overlap between chunks
        
    Returns:
        List of LangChain Document objects ready for embedding
    """
    data = load_json_data(json_path)
    
    if chunk_size and chunk_size > 1:
        documents = create_chunked_documents(data, chunk_size, chunk_overlap or 0)
    else:
        documents = convert_to_documents(data)
    
    if not documents:
        raise ValueError("No valid documents created from JSON data")
    
    return documents


def create_chunked_documents(data: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Create chunked documents by grouping consecutive verses.
    
    This improves retrieval for comparison questions that span multiple verses/chapters.
    
    Args:
        data: List of dictionaries containing verse data
        chunk_size: Number of consecutive verses per chunk
        chunk_overlap: Number of verses to overlap between chunks
        
    Returns:
        List of chunked LangChain Document objects
    """
    # Sort data by chapter and verse to ensure consecutive order
    sorted_data = sorted(data, key=lambda x: (
        int(x.get("metadata", {}).get("chapter", 0)),
        int(x.get("metadata", {}).get("verse", 0))
    ))
    
    documents = []
    step = max(1, chunk_size - chunk_overlap)
    
    for i in range(0, len(sorted_data), step):
        chunk_entries = sorted_data[i:i + chunk_size]
        
        if not chunk_entries:
            continue
        
        # Combine page content from all verses in chunk
        combined_content = "\n\n---\n\n".join([
            entry.get("text", "") for entry in chunk_entries
        ])
        
        # Extract metadata from chunk
        first_entry = chunk_entries[0]
        last_entry = chunk_entries[-1]
        
        first_metadata = first_entry.get("metadata", {})
        last_metadata = last_entry.get("metadata", {})
        
        first_chapter = first_metadata.get("chapter", "Unknown")
        first_verse = first_metadata.get("verse", "Unknown")
        last_chapter = last_metadata.get("chapter", "Unknown")
        last_verse = last_metadata.get("verse", "Unknown")
        
        # Create verse range
        if first_chapter == last_chapter:
            verse_number = f"{first_chapter}.{first_verse}-{last_verse}"
        else:
            verse_number = f"{first_chapter}.{first_verse} to {last_chapter}.{last_verse}"
        
        # Collect all chapters in this chunk
        chapters_in_chunk = list(set([
            entry.get("metadata", {}).get("chapter", "Unknown") 
            for entry in chunk_entries
        ]))
        
        # Collect all verse numbers
        verses_in_chunk = [
            f"{entry.get('metadata', {}).get('chapter', '?')}.{entry.get('metadata', {}).get('verse', '?')}"
            for entry in chunk_entries
        ]
        
        # Store individual verse contents for exact lookups
        verse_texts = {}
        for entry in chunk_entries:
            entry_metadata = entry.get("metadata", {})
            ch = entry_metadata.get("chapter", "?")
            v = entry_metadata.get("verse", "?")
            verse_key = f"{ch}.{v}"
            verse_texts[verse_key] = entry.get("text", "")
        
        metadata = {
            "verse_number": verse_number,
            "verse_id": f"BG{first_chapter}.{first_verse}-{last_verse}",
            "chapter": first_chapter,
            "verse": first_verse,
            "chunk_size": len(chunk_entries),
            "chapters_in_chunk": chapters_in_chunk,
            "verses_in_chunk": verses_in_chunk,
            "verse_texts": verse_texts  # Individual verse texts for exact lookups
        }
        
        doc = Document(page_content=combined_content, metadata=metadata)
        documents.append(doc)
    
    logger.info(f"Created {len(documents)} chunked documents (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return documents

