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
    """Orchestrates the translation process with batch processing"""

    def __init__(
        self,
        llm_manager: TranslationLLMManager,
        russian_prompt: str,
        kazakh_prompt: str,
        russian_batch_size: int = 32,
        kazakh_batch_size: int = 48
    ):
        """
        Initialize orchestrator

        Args:
            llm_manager: LLM manager with Russian and Kazakh clients
            russian_prompt: Prompt template for Russian translation
            kazakh_prompt: Prompt template for Kazakh translation
            russian_batch_size: Batch size for Russian translation
            kazakh_batch_size: Batch size for Kazakh translation
        """
        self.llm_manager = llm_manager
        self.russian_prompt = russian_prompt
        self.kazakh_prompt = kazakh_prompt
        self.russian_batch_size = russian_batch_size
        self.kazakh_batch_size = kazakh_batch_size

    def translate_sentences(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        Translate all sentences to Russian and Kazakh using batch processing
        Phase 1: All Russian translations in batches
        Phase 2: All Kazakh translations in batches

        Args:
            sentences: List of Sentence objects

        Returns:
            List of Sentence objects with translations added (order preserved)
        """
        total = len(sentences)
        logger.info(f"Starting batch translation of {total} sentences...")
        logger.info(f"Batch sizes: Russian={self.russian_batch_size}, Kazakh={self.kazakh_batch_size}")

        # PHASE 1: Russian translation in batches
        logger.info("\n=== PHASE 1: Russian Translation ===")
        self._translate_language_batched(
            sentences,
            'russian',
            self.russian_batch_size,
            self.llm_manager.translate_batch_to_russian,
            self.russian_prompt
        )

        # PHASE 2: Kazakh translation in batches
        logger.info("\n=== PHASE 2: Kazakh Translation ===")
        self._translate_language_batched(
            sentences,
            'kazakh',
            self.kazakh_batch_size,
            self.llm_manager.translate_batch_to_kazakh,
            self.kazakh_prompt
        )

        logger.info("\n✓ Translation completed!")
        return sentences

    def _translate_language_batched(
        self,
        sentences: List[Sentence],
        language: str,
        batch_size: int,
        translate_func,
        prompt_template: str
    ):
        """
        Translate sentences to a specific language in batches
        CRITICAL: Preserves exact order using indices

        Args:
            sentences: List of Sentence objects
            language: Target language name ('russian' or 'kazakh')
            batch_size: Size of each batch
            translate_func: Translation function to use
            prompt_template: Prompt template for translation
        """
        total = len(sentences)
        num_batches = (total + batch_size - 1) // batch_size  # Ceiling division

        for batch_idx in range(num_batches):
            # Calculate batch boundaries
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_sentences = sentences[start_idx:end_idx]

            logger.info(f"Batch {batch_idx + 1}/{num_batches}: Translating sentences {start_idx + 1}-{end_idx} to {language}...")

            # Extract texts from batch (preserves order)
            batch_texts = [sent.text for sent in batch_sentences]

            # Translate entire batch
            try:
                translations = translate_func(batch_texts, prompt_template)

                # CRITICAL: Assign translations back to correct sentences using index
                for i, translation in enumerate(translations):
                    sentence_idx = start_idx + i  # Global index in sentences list
                    sentences[sentence_idx].add_translation(language, translation)
                    logger.info(f"  [{sentence_idx + 1}] ✓ {translation[:60]}...")

            except Exception as e:
                logger.error(f"  ✗ Batch translation failed: {e}")
                # On batch failure, mark all sentences in batch with error
                for i in range(len(batch_sentences)):
                    sentence_idx = start_idx + i
                    sentences[sentence_idx].add_translation(language, f"[ERROR: {str(e)}]")
