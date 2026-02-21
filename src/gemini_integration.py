"""
Google Gemini LLM Integration for Mahabharat RAG Chatbot
Provides AI-powered response generation using Google's Generative AI API
"""

import logging
import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiLLMIntegration:
    """
    Integration with Google Gemini LLM for response generation.
    
    Uses the google-generativeai library to communicate with Gemini API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash-lite-preview-09-2025"):
        """
        Initialize Gemini LLM integration.
        
        Args:
            api_key: Google Gemini API key (if None, reads from GEMINI_API_KEY env var)
            model: Model name to use (default: gemini-1.5-flash)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini LLM initialized successfully with model: {self.model_name}")
        else:
            self.model = None
            logger.warning("GEMINI_API_KEY environment variable not set")
    
    def is_available(self) -> bool:
        """
        Check if Gemini LLM is available.
        
        Returns:
            True if API key is set and model is initialized, False otherwise
        """
        return self.api_key is not None and self.model is not None
    
    def generate_response(
        self,
        user_query: str,
        retrieved_verses: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        strict_grounding: bool = True
    ) -> Optional[str]:
        """
        Generate an LLM response based on retrieved verses and user query.
        
        Args:
            user_query: The original user question
            retrieved_verses: List of relevant verses with metadata
            temperature: Generation temperature (0-2, higher = more creative)
            max_tokens: Maximum tokens to generate
            strict_grounding: If True, prevent hallucination - only use retrieved context
            
        Returns:
            Generated response text, or None if generation fails
        """
        if not self.is_available():
            logger.warning("Gemini LLM not available")
            return None
        
        try:
            # Build context from retrieved verses
            verse_context = self._build_verse_context(retrieved_verses)
            
            # Create the prompt (strict or normal)
            prompt = self._create_prompt(user_query, verse_context, strict_grounding)
            
            logger.info("Generating response with Gemini LLM...")
            
            # Generate response using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            
            if response.text:
                logger.info("Gemini response generated successfully")
                return response.text.strip()
            else:
                logger.warning("Gemini returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            return None
    
    def _build_verse_context(self, retrieved_verses: List[Dict[str, str]]) -> str:
        """
        Build context string from retrieved verses.
        
        Args:
            retrieved_verses: List of verses with verse_number and content
            
        Returns:
            Formatted verse context with chapter information
        """
        context = "Retrieved Context from Bhagavad Gita:\n\n"
        context+= "="*80 + "\n\n"
        
        for i, verse in enumerate(retrieved_verses, 1):
            verse_num = verse.get("verse_number", "Unknown")
            chapter = verse.get("chapter", "Unknown")
            content = verse.get("content", "")
            
            context += f"[Document {i}] Verse {verse_num} (Chapter {chapter}):\n"
            context += f"{content}\n"
            context += "\n" + "-"*80 + "\n\n"
        
        return context
    
    def _create_prompt(self, user_query: str, verse_context: str, strict_grounding: bool = True) -> str:
        """
        Create a structured prompt for Gemini.
        
        Args:
            user_query: User's original question
            verse_context: Context from retrieved verses
            strict_grounding: If True, use strict prompt to prevent hallucination
            
        Returns:
            Formatted prompt for the LLM
        """
        if strict_grounding:
            # Strict grounding mode: prevent hallucination
            system_prompt = """You are a Bhagavad Gita scripture assistant with STRICT GROUNDING rules:

CRITICAL INSTRUCTIONS:
1. ONLY use information from the provided context below
2. If the context does not contain sufficient information to answer the question, reply EXACTLY: "Answer not found"
3. Do NOT add any external knowledge or information not present in the context
4. Do NOT make assumptions or inferences beyond what is explicitly stated
5. Always cite specific verse numbers when answering
6. If comparing chapters, ensure BOTH chapters are present in the context

You must follow these rules absolutely. Any violation will be considered an error."""

            prompt = f"""{system_prompt}

{verse_context}

User Question: {user_query}

Instructions:
- Answer ONLY using the context above
- If the context is insufficient, reply exactly: "Answer not found"
- Cite specific verse numbers
- For comparisons, verify all required chapters are present in context"""

        else:
            # Normal mode: helpful but still grounded
            system_prompt = """You are an expert in the Bhagavad Gita and Hindu philosophy. 
Your role is to provide insightful, accurate, and helpful answers based on the verses provided.
Give clear, concise answers that relate the verse content directly to the user's question.
Cite relevant verses when explaining concepts. Be respectful and educational."""
        
            prompt = f"""{system_prompt}

{verse_context}

User Question: {user_query}

Please provide a thoughtful answer based on the verses above. Include:
1. Direct reference to the verses
2. Clear explanation of the concept
3. How it applies to the user's question"""
        
        return prompt
