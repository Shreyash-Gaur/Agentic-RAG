"""
Ollama LLM client wrapper.
"""

import requests
from typing import Optional, Dict, Any
from core.config import settings
from core.logger import setup_logger
from core.exceptions import LLMError

logger = setup_logger(__name__)


class OllamaClient:
    """
    Client for interacting with Ollama LLM API.
    """
    
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        temperature: float = None
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use
            temperature: Sampling temperature
        """
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self.temperature = temperature or settings.TEMPERATURE
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            system: Optional system message
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    **kwargs
                }
            }
            
            if system:
                payload["system"] = system
            
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise LLMError(f"Failed to generate response: {e}")
    
    def chat(
        self,
        messages: list,
        **kwargs
    ) -> str:
        """
        Chat completion using messages format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        try:
            url = f"{self.base_url}/api/chat"
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    **kwargs
                }
            }
            
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise LLMError(f"Failed to chat: {e}")

