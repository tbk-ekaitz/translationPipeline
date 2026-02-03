"""
LLM client wrapper for Ollama and llama.cpp
Supports multiple models running simultaneously with batch processing
"""
import requests
from typing import List, Optional
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed


class BaseLLMClient(ABC):
    """Base class for LLM clients"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str], system_prompt: str = "") -> List[str]:
        """Generate text from multiple prompts in batch"""
        pass


class OllamaClient(BaseLLMClient):
    """Client for Ollama API"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", max_workers: int = 16):
        """
        Initialize Ollama client

        Args:
            model_name: Name of the Ollama model
            base_url: Base URL for Ollama API
            max_workers: Max parallel requests for batch processing (default 16).
                         Should match OLLAMA_NUM_PARALLEL env var in Ollama server.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.max_workers = max_workers

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

    def generate_batch(self, prompts: List[str], system_prompt: str = "") -> List[str]:
        """
        Generate text from multiple prompts in parallel using ThreadPoolExecutor.
        Requires Ollama server started with OLLAMA_NUM_PARALLEL env var set.

        Args:
            prompts: List of user prompts
            system_prompt: System prompt (optional)

        Returns:
            List of generated texts in same order as input prompts
        """
        if not prompts:
            return []

        # Pre-allocate results list to maintain order
        results = [None] * len(prompts)

        def process_prompt(index_prompt):
            idx, prompt = index_prompt
            try:
                return idx, self.generate(prompt, system_prompt)
            except Exception as e:
                return idx, f"[ERROR: {str(e)}]"

        # Use ThreadPoolExecutor for parallel HTTP requests
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = executor.map(process_prompt, enumerate(prompts))
            for idx, result in futures:
                results[idx] = result

        return results


class LlamaCppClient(BaseLLMClient):
    """Client for llama.cpp server"""

    def __init__(self, base_url: str = "http://localhost:8080", max_workers: int = 16):
        """
        Initialize llama.cpp client

        Args:
            base_url: Base URL for llama.cpp server
            max_workers: Max parallel requests for batch processing (default 16).
                         Should match --parallel flag in llama-server.
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/completion"
        self.max_workers = max_workers

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

    def generate_batch(self, prompts: List[str], system_prompt: str = "") -> List[str]:
        """
        Generate text from multiple prompts in parallel using ThreadPoolExecutor.
        Requires llama-server started with --parallel flag.

        Args:
            prompts: List of user prompts
            system_prompt: System prompt (optional)

        Returns:
            List of generated texts in same order as input prompts
        """
        if not prompts:
            return []

        # Pre-allocate results list to maintain order
        results = [None] * len(prompts)

        def process_prompt(index_prompt):
            idx, prompt = index_prompt
            try:
                return idx, self.generate(prompt, system_prompt)
            except Exception as e:
                return idx, f"[ERROR: {str(e)}]"

        # Use ThreadPoolExecutor for parallel HTTP requests
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = executor.map(process_prompt, enumerate(prompts))
            for idx, result in futures:
                results[idx] = result

        return results


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

    def translate_batch_to_russian(self, texts: List[str], prompt_template: str) -> List[str]:
        """
        Translate multiple texts to Russian in batch
        Preserves exact order of input texts

        Args:
            texts: List of texts to translate
            prompt_template: Prompt template with {text} placeholder

        Returns:
            List of Russian translations in same order as input
        """
        prompts = [prompt_template.format(text=text) for text in texts]
        return self.russian_client.generate_batch(prompts)

    def translate_batch_to_kazakh(self, texts: List[str], prompt_template: str) -> List[str]:
        """
        Translate multiple texts to Kazakh in batch
        Preserves exact order of input texts

        Args:
            texts: List of texts to translate
            prompt_template: Prompt template with {text} placeholder

        Returns:
            List of Kazakh translations in same order as input
        """
        prompts = [prompt_template.format(text=text) for text in texts]
        return self.kazakh_client.generate_batch(prompts)
