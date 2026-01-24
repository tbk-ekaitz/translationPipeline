"""
LLM client wrapper for Ollama and llama.cpp
Supports multiple models running simultaneously
"""
import requests
from typing import Optional
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Base class for LLM clients"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text from prompt"""
        pass


class OllamaClient(BaseLLMClient):
    """Client for Ollama API"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client

        Args:
            model_name: Name of the Ollama model
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate text using Ollama API

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)

        Returns:
            Generated text
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")


class LlamaCppClient(BaseLLMClient):
    """Client for llama.cpp server"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize llama.cpp client

        Args:
            base_url: Base URL for llama.cpp server
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/completion"

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate text using llama.cpp API

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)

        Returns:
            Generated text
        """
        # Combine system and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "prompt": full_prompt,
            "n_predict": 512,
            "temperature": 0.7,
            "stop": ["\n\n", "###"],
            "stream": False
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result.get("content", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"llama.cpp API error: {e}")


class TranslationLLMManager:
    """Manages multiple LLM clients for translation"""

    def __init__(self, russian_client: BaseLLMClient, kazakh_client: BaseLLMClient):
        """
        Initialize translation manager

        Args:
            russian_client: LLM client for Russian translation
            kazakh_client: LLM client for Kazakh translation
        """
        self.russian_client = russian_client
        self.kazakh_client = kazakh_client

    def translate_to_russian(self, text: str, prompt_template: str) -> str:
        """
        Translate text to Russian

        Args:
            text: Text to translate
            prompt_template: Prompt template with {text} placeholder

        Returns:
            Russian translation
        """
        prompt = prompt_template.format(text=text)
        return self.russian_client.generate(prompt)

    def translate_to_kazakh(self, text: str, prompt_template: str) -> str:
        """
        Translate text to Kazakh

        Args:
            text: Text to translate
            prompt_template: Prompt template with {text} placeholder

        Returns:
            Kazakh translation
        """
        prompt = prompt_template.format(text=text)
        return self.kazakh_client.generate(prompt)
