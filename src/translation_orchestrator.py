"""
Translation orchestrator
Coordinates the entire translation pipeline
"""
from typing import List
from .sentence_segmenter import Sentence
from .llm_client import TranslationLLMManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationOrchestrator:
    """Orchestrates the translation process"""

    def __init__(
        self,
        llm_manager: TranslationLLMManager,
        russian_prompt: str,
        kazakh_prompt: str
    ):
        """
        Initialize orchestrator

        Args:
            llm_manager: LLM manager with Russian and Kazakh clients
            russian_prompt: Prompt template for Russian translation
            kazakh_prompt: Prompt template for Kazakh translation
        """
        self.llm_manager = llm_manager
        self.russian_prompt = russian_prompt
        self.kazakh_prompt = kazakh_prompt

    def translate_sentences(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        Translate all sentences to Russian and Kazakh

        Args:
            sentences: List of Sentence objects

        Returns:
            List of Sentence objects with translations added
        """
        total = len(sentences)
        logger.info(f"Starting translation of {total} sentences...")

        for idx, sentence in enumerate(sentences, start=1):
            logger.info(f"Translating sentence {idx}/{total}: {sentence.text[:50]}...")

            try:
                # Translate to Russian
                russian_translation = self.llm_manager.translate_to_russian(
                    sentence.text,
                    self.russian_prompt
                )
                sentence.add_translation('russian', russian_translation)
                logger.info(f"  ✓ Russian: {russian_translation[:50]}...")

            except Exception as e:
                logger.error(f"  ✗ Russian translation failed: {e}")
                sentence.add_translation('russian', f"[ERROR: {str(e)}]")

            try:
                # Translate to Kazakh
                kazakh_translation = self.llm_manager.translate_to_kazakh(
                    sentence.text,
                    self.kazakh_prompt
                )
                sentence.add_translation('kazakh', kazakh_translation)
                logger.info(f"  ✓ Kazakh: {kazakh_translation[:50]}...")

            except Exception as e:
                logger.error(f"  ✗ Kazakh translation failed: {e}")
                sentence.add_translation('kazakh', f"[ERROR: {str(e)}]")

        logger.info("Translation completed!")
        return sentences
