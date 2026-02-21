"""
RAG Pipeline module for Mahabharat RAG Chatbot
Handles query processing, retrieval, and response generation
"""

import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MahabharatRAGPipeline:
    """
    RAG Pipeline for Mahabharat chatbot.
    
    This class handles the complete RAG workflow:
    1. Query processing
    2. Similarity search
    3. Threshold checking
    4. LLM response generation (optional)
    5. Response formatting
    """
    
    def __init__(
        self,
        vector_store: FAISS,
        similarity_threshold: float,
        top_k: int = 8,
        answer_not_found_message: str = "Answer not found",
        llm_integration: Optional[Any] = None,
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 2000,
        debug_mode: bool = False,
        strict_grounding: bool = True
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: FAISS vector store instance
            similarity_threshold: Threshold for similarity score (lower is better for L2 distance)
            top_k: Number of results to retrieve (higher for multi-chapter comparisons)
            answer_not_found_message: Message to return when no relevant results found
            llm_integration: Optional LLM integration instance for response generation
            llm_temperature: Temperature for LLM generation
            llm_max_tokens: Max tokens for LLM generation
            debug_mode: Print retrieval details (documents, scores, chapters)
            strict_grounding: Prevent hallucination - only use retrieved context
        """
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.answer_not_found_message = answer_not_found_message
        self.llm_integration = llm_integration
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.debug_mode = debug_mode
        self.strict_grounding = strict_grounding
        logger.info(f"RAG Pipeline initialized (top_k={top_k}, debug={debug_mode}, strict_grounding={strict_grounding})")
        if llm_integration and llm_integration.is_available():
            logger.info("LLM integration enabled")
    
    def _retrieve_documents(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents using similarity search.
        
        Args:
            query: User query string
            
        Returns:
            List of tuples containing (Document, similarity_score)
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
            logger.info(f"Retrieved {len(results)} documents for query")
            
            # Debug mode: print detailed retrieval information
            if self.debug_mode and results:
                print("\n" + "="*80)
                print("DEBUG: RETRIEVAL DETAILS")
                print("="*80)
                print(f"Query: {query}")
                print(f"Top K: {self.top_k}")
                print(f"Retrieved Documents: {len(results)}")
                print("-"*80)
                
                for idx, (doc, score) in enumerate(results, 1):
                    metadata = doc.metadata
                    verse_num = metadata.get("verse_number", "Unknown")
                    chapter = metadata.get("chapter", "Unknown")
                    confidence = round(1 / (1 + score), 4)
                    
                    print(f"\n[{idx}] Verse: {verse_num}")
                    print(f"    Chapter: {chapter}")
                    print(f"    L2 Distance: {score:.4f}")
                    print(f"    Confidence: {confidence*100:.2f}%")
                    print(f"    Passes Threshold ({self.similarity_threshold}): {score <= self.similarity_threshold}")
                    
                    # Show chunk info if available
                    if "chapters_in_chunk" in metadata:
                        chapters_in_chunk = metadata.get("chapters_in_chunk", [])
                        verses_in_chunk = metadata.get("verses_in_chunk", [])
                        print(f"    Chunked Document: {len(verses_in_chunk)} verses")
                        print(f"    Chapters in Chunk: {chapters_in_chunk}")
                        print(f"    Verses: {', '.join(map(str, verses_in_chunk))}")
                
                # Summary of chapters retrieved
                all_chapters = set()
                for doc, score in results:
                    if score <= self.similarity_threshold:
                        chapter = doc.metadata.get("chapter", "Unknown")
                        if chapter != "Unknown":
                            all_chapters.add(int(chapter) if isinstance(chapter, (int, str)) and str(chapter).isdigit() else chapter)
                        
                        # Also add chapters from chunks
                        if "chapters_in_chunk" in doc.metadata:
                            for ch in doc.metadata["chapters_in_chunk"]:
                                if ch != "Unknown":
                                    all_chapters.add(int(ch) if isinstance(ch, (int, str)) and str(ch).isdigit() else ch)
                
                print("\n" + "-"*80)
                print(f"Relevant Chapters Retrieved: {sorted(all_chapters) if all_chapters else 'None'}")
                print("="*80 + "\n")
            
            return results
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            return []
    
    def _check_relevance(self, similarity_score: float) -> bool:
        """
        Check if the document is relevant based on similarity threshold.
        
        For FAISS L2 distance: lower scores are better matches.
        Documents with score <= threshold are considered relevant.
        
        Args:
            similarity_score: L2 distance score from FAISS
            
        Returns:
            True if document is relevant, False otherwise
        """
        is_relevant = similarity_score <= self.similarity_threshold
        logger.debug(f"Similarity score: {similarity_score:.4f}, Threshold: {self.similarity_threshold}, Relevant: {is_relevant}")
        return is_relevant
    
    def _format_response(self, document: Document, similarity_score: float) -> Dict[str, Any]:
        """
        Format the response from retrieved document.
        
        Args:
            document: Retrieved LangChain Document
            similarity_score: Similarity score of the document
            
        Returns:
            Structured response dictionary
        """
        metadata = document.metadata
        page_content = document.page_content
        
        response = {
            "verse_number": metadata.get("verse_number", ""),
            "verse_id": metadata.get("verse_id", ""),
            "chapter": metadata.get("chapter", ""),
            "verse": metadata.get("verse", ""),
            "content": page_content,  # Full text with all details
            "confidence_score": float(round(1 / (1 + float(similarity_score)), 4))  # Convert L2 distance to confidence (0-1)
        }
        
        return response
    
    def _create_not_found_response(self) -> Dict[str, str]:
        """
        Create response when no relevant answer is found.
        
        Returns:
            Dictionary with "answer not found" message
        """
        return {"answer": self.answer_not_found_message}
    
    def _extract_verse_number(self, query: str) -> Optional[str]:
        """
        Extract verse number from query like "chapter 2 verse 47" or "BG 2.47".
        
        Args:
            query: User query string
            
        Returns:
            Verse number string like "2.47" if found, None otherwise
        """
        # Match patterns like "chapter 2 verse 47" or "BG 2.47" or "2.47"
        patterns = [
            r'chapter\s+(\d+)\s+verse\s+(\d+)',  # chapter X verse Y
            r'BG\s+(\d+)\.(\d+)',                 # BG X.Y
            r'(\d+)\.(\d+)',                      # X.Y
        ]
        
        query_lower = query.lower()
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                if len(match.groups()) == 2:
                    chapter, verse = match.groups()
                    return f"{chapter}.{verse}"
        
        return None
    
    def _exact_verse_lookup(self, verse_number: str) -> Optional[Dict[str, Any]]:
        """
        Look up a specific verse by exact verse number.
        
        This bypasses the similarity threshold check since we're doing exact matching.
        
        Args:
            verse_number: Verse number string like "2.47"
            
        Returns:
            Response dictionary if found, None otherwise
        """
        try:
            # Extract chapter and verse from verse_number
            parts = verse_number.split('.')
            if len(parts) != 2:
                logger.warning(f"Invalid verse number format: {verse_number}")
                return None
            
            try:
                chapter = int(parts[0])
                verse = int(parts[1])
            except ValueError:
                logger.warning(f"Could not parse chapter/verse from {verse_number}")
                return None
            
            # Search with very large k to get all possible matches
            # then filter by exact chapter and verse
            results = self.vector_store.similarity_search_with_score(
                f"chapter {chapter} verse {verse}", k=700  # Get many results to ensure coverage
            )
            
            if not results:
                logger.info(f"No results found for verse {verse_number}")
                return None
            
            # Look through ALL results for exact verse match
            for document, score in results:
                metadata = document.metadata
                
                # Check if this is a chunked document (has verses_in_chunk)
                verses_in_chunk = metadata.get("verses_in_chunk", [])
                
                if verses_in_chunk:
                    # For chunked documents, check if verse is in the chunk
                    if verse_number in verses_in_chunk:
                        logger.info(f"Found exact match for verse {verse_number} in chunk: {metadata.get('verse_number')} with score: {score}")
                        
                        # Extract only the requested verse from the chunk
                        verse_texts = metadata.get("verse_texts", {})
                        if verse_number in verse_texts:
                            # Create a single-verse document
                            single_verse_doc = Document(
                                page_content=verse_texts[verse_number],
                                metadata={
                                    "verse_number": verse_number,
                                    "verse_id": f"BG{verse_number}",
                                    "chapter": str(chapter),
                                    "verse": str(verse),
                                    "is_exact_match": True
                                }
                            )
                            return self._format_response(single_verse_doc, score)
                        else:
                            # Fallback: return the chunk if individual verse not found
                            return self._format_response(document, score)
                else:
                    # For non-chunked documents, use old logic
                    doc_chapter = metadata.get("chapter")
                    doc_verse = metadata.get("verse")
                    
                    try:
                        if (int(doc_chapter) == chapter and int(doc_verse) == verse):
                            logger.info(f"Found exact match for verse: {verse_number} with score: {score}")
                            return self._format_response(document, score)
                    except (TypeError, ValueError):
                        continue
            
            logger.info(f"Exact verse {verse_number} not found in {len(results)} search results")
            return None
        except Exception as e:
            logger.error(f"Error during exact verse lookup: {e}")
            return None
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Process user query and return structured response.
        
        This is the main method to use for querying the RAG system.
        
        Workflow:
        1. Check if query contains explicit verse reference (e.g., "chapter 2 verse 47")
        2. If yes, try exact verse lookup first
        3. Fall back to semantic search if exact lookup fails
        4. Check if best match exceeds similarity threshold
        5. Return structured response or "answer not found"
        
        Args:
            user_query: User's question or query
            
        Returns:
            Structured response dictionary containing either:
            - Verse information with confidence score (if relevant match found)
            - "Answer not found" message (if no relevant match found)
        """
        logger.info(f"Processing query: '{user_query}'")
        
        # Input validation
        if not user_query or not user_query.strip():
            logger.warning("Empty query received")
            return self._create_not_found_response()
        
        # Try exact verse lookup first
        verse_number = self._extract_verse_number(user_query)
        if verse_number:
            logger.info(f"Detected verse reference: {verse_number}, attempting exact lookup")
            exact_result = self._exact_verse_lookup(verse_number)
            if exact_result:
                return exact_result
            logger.info(f"Exact lookup for {verse_number} failed, falling back to semantic search")
        
        # Retrieve documents using semantic search
        results = self._retrieve_documents(user_query)
        
        if not results:
            logger.info("No results retrieved")
            return self._create_not_found_response()
        
        # Get the best match (first result has lowest distance/best match)
        best_document, best_score = results[0]
        
        logger.info(f"Best match score: {best_score:.4f}")
        
        # Check relevance
        if not self._check_relevance(best_score):
            logger.info(f"Best match score {best_score:.4f} exceeds threshold {self.similarity_threshold}")
            return self._create_not_found_response()
        
        # Format base response
        response = self._format_response(best_document, best_score)
        
        # Generate LLM response if available
        if self.llm_integration and self.llm_integration.is_available():
            try:
                # Prepare relevant verses for LLM context
                relevant_verses = []
                chapters_retrieved = set()
                
                for doc, score in results:
                    if self._check_relevance(score):
                        verse_dict = {
                            "verse_number": doc.metadata.get("verse_number", "Unknown"),
                            "content": doc.page_content,
                            "chapter": doc.metadata.get("chapter", "Unknown")
                        }
                        relevant_verses.append(verse_dict)
                        
                        # Track chapters
                        chapter = doc.metadata.get("chapter")
                        if chapter and chapter != "Unknown":
                            chapters_retrieved.add(int(chapter) if str(chapter).isdigit() else chapter)
                        
                        # Also track chapters from chunked documents
                        if "chapters_in_chunk" in doc.metadata:
                            for ch in doc.metadata["chapters_in_chunk"]:
                                if ch and ch != "Unknown":
                                    chapters_retrieved.add(int(ch) if str(ch).isdigit() else ch)
                
                if relevant_verses:
                    logger.info(f"Generating LLM response with {len(relevant_verses)} relevant verses from chapters: {sorted(chapters_retrieved)}")
                    llm_response = self.llm_integration.generate_response(
                        user_query=user_query,
                        retrieved_verses=relevant_verses,
                        temperature=self.llm_temperature,
                        max_tokens=self.llm_max_tokens,
                        strict_grounding=self.strict_grounding
                    )
                    
                    if llm_response:
                        response["llm_response"] = llm_response
                        response["chapters_retrieved"] = sorted(list(chapters_retrieved))
                        logger.info("LLM response generated successfully")
            except Exception as e:
                logger.warning(f"Error generating LLM response: {e}")
                # Continue without LLM response if there's an error
        
        logger.info(f"Returning verse: {response['verse_number']}")
        return response
    
    def batch_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of user queries
            
        Returns:
            List of structured response dictionaries
        """
        logger.info(f"Processing batch of {len(queries)} queries")
        return [self.query(query) for query in queries]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG pipeline.
        
        Returns:
            Dictionary containing pipeline statistics
        """
        try:
            # Get number of documents in vector store
            # FAISS doesn't have a direct method, but we can get it from index
            num_docs = self.vector_store.index.ntotal
        except:
            num_docs = "unknown"
        
        return {
            "num_documents": num_docs,
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
            "vector_store_type": type(self.vector_store).__name__
        }
