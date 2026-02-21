"""
LLM Integration module for Mahabharat RAG Chatbot
Handles integration with Grok LLM via xAI API
"""

import os
import logging
import requests
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrokLLMIntegration:
    """
    Integration with Grok LLM via xAI API using direct HTTP requests.
    
    Uses the xAI API endpoint for Grok.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-beta",
        temperature: float = 0.7,
        max_tokens: int = 1500
    ):
        """
        Initialize Grok LLM integration.
        
        Args:
            api_key: xAI API key (defaults to XAI_API_KEY or GROK_API_KEY env variable)
            model: Model name (default: grok-beta)
            temperature: Temperature for generation (0-2)
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://api.x.ai/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("GROK_API_KEY or XAI_API_KEY environment variable not set")
        else:
            logger.info(f"Grok LLM initialized successfully with model: {model}")
    
    def _create_context_prompt(
        self,
        user_query: str,
        retrieved_verses: List[Dict[str, Any]],
        system_role: str = "You are an expert on the Bhagavad Gita."
    ) -> str:
        """
        Create a prompt with retrieved verses as context.
        
        Args:
            user_query: Original user query
            retrieved_verses: List of relevant verses from RAG system
            system_role: System role/instructions for the LLM
            
        Returns:
            Formatted prompt string
        """
        verses_context = "\n\n".join(
            [f"Verse {v.get('verse_number', 'Unknown')}:\n{v.get('content', '')}"
             for v in retrieved_verses]
        )
        
        prompt = f"""{system_role}

You are helping someone understand the Bhagavad Gita. Based on the following verses from the scripture, answer the user's question comprehensively and thoughtfully.

RELEVANT VERSES FROM BHAGAVAD GITA:
{verses_context}

USER'S QUESTION:
{user_query}

Please provide a clear, insightful answer based on these verses. Explain how the verses relate to the question and provide spiritual wisdom from the teachings."""
        
        return prompt
    
    def generate_response(
        self,
        user_query: str,
        retrieved_verses: List[Dict[str, Any]],
        system_role: str = "You are an expert on the Bhagavad Gita."
    ) -> Optional[str]:
        """
        Generate an LLM response based on user query and retrieved verses.
        
        Args:
            user_query: Original user query
            retrieved_verses: List of relevant verses from RAG system
            system_role: System role/instructions for the LLM
            
        Returns:
            LLM-generated response string, or None if LLM not initialized or error occurs
        """
        if not self.api_key:
            logger.warning("LLM not initialized (no API key), cannot generate response")
            return None
        
        try:
            prompt = self._create_context_prompt(user_query, retrieved_verses, system_role)
            
            logger.info("Generating response with Grok LLM...")
            
            # Prepare request headers and payload
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_role
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            # Make API request
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                content = message.get("content", "")
                
                if content:
                    logger.info("Response generated successfully")
                    return content
            
            logger.warning("No valid response content from Grok API")
            return None
            
        except requests.exceptions.Timeout:
            logger.error("Timeout connecting to Grok API")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from Grok API: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if LLM is available and initialized."""
        return self.api_key is not None
    
    def generate_verse_explanation(
        self,
        verse_content: str,
        verse_number: str
    ) -> Optional[str]:
        """
        Generate a detailed explanation for a specific verse.
        
        Args:
            verse_content: Full verse content/text
            verse_number: Verse number (e.g., "2.47")
            
        Returns:
            LLM-generated explanation, or None if LLM not initialized
        """
        if not self.api_key:
            return None
        
        try:
            prompt = f"""Provide a comprehensive explanation of the following verse from the Bhagavad Gita:

Verse {verse_number}:
{verse_content}

Include:
1. The meaning and context of the verse
2. Key teachings and wisdom
3. Practical applications in modern life
4. Connection to broader spiritual themes"""
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0].get("message", {}).get("content", "")
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating verse explanation: {e}")
            return None

