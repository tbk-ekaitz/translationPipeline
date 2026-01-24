"""
Translation orchestrator
Coordinates the entire translation pipeline with chunk support
"""
from typing import List
from .sentence_segmenter import Sentence
from .llm_client import TranslationLLMManager
from tqdm import tqdm
import re
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
        Handles chunked sentences by expanding and merging

        Args:
            sentences: List of Sentence objects
            language: Target language name ('russian' or 'kazakh')
            batch_size: Size of each batch
            translate_func: Translation function to use
            prompt_template: Prompt template for translation
        """
        total = len(sentences)
        num_batches = (total + batch_size - 1) // batch_size  # Ceiling division

        # Progress bar for batches
        progress_bar = tqdm(
            range(num_batches),
            desc=f"Translating to {language.capitalize()}",
            unit="batch",
            ncols=100
        )

        for batch_idx in progress_bar:
            # Calculate batch boundaries
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_sentences = sentences[start_idx:end_idx]

            # Update progress bar description with current batch info
            progress_bar.set_postfix({
                'sentences': f'{start_idx + 1}-{end_idx}/{total}'
            })

            # Expand chunks for batch
            batch_items = []
            chunk_metadata = []

            for local_idx, sent in enumerate(batch_sentences):
                global_idx = start_idx + local_idx  # CRITICAL: Global index

                if sent.chunks:
                    # Sentence was split, add chunks
                    batch_items.extend(sent.chunks)
                    chunk_metadata.append((global_idx, True, len(sent.chunks)))
                else:
                    # Normal sentence
                    batch_items.append(sent.text)
                    chunk_metadata.append((global_idx, False, 1))

            # Translate entire batch (chunks expanded)
            try:
                translations = translate_func(batch_items, prompt_template)

                # Merge chunks and assign back to sentences
                trans_idx = 0
                for global_idx, is_chunked, num_chunks in chunk_metadata:
                    if is_chunked:
                        # Get chunk translations and merge
                        chunk_trans = translations[trans_idx:trans_idx + num_chunks]
                        merged = self._merge_chunk_translations(chunk_trans)
                        sentences[global_idx].add_translation(language, merged)
                        trans_idx += num_chunks
                    else:
                        # Single translation
                        sentences[global_idx].add_translation(language, translations[trans_idx])
                        trans_idx += 1

            except Exception as e:
                logger.error(f"  ✗ Batch translation failed: {e}")
                # On batch failure, mark all sentences in batch with error
                for global_idx, _, _ in chunk_metadata:
                    sentences[global_idx].add_translation(language, f"[ERROR: {str(e)}]")

    @staticmethod
    def _merge_chunk_translations(chunk_trans: List[str]) -> str:
        """
        Merge chunk translations intelligently
        Cleans up spacing and punctuation

        Args:
            chunk_trans: List of translated chunks

        Returns:
            Merged translation
        """
        # Filter empty chunks
        chunks = [t.strip() for t in chunk_trans if t.strip()]

        if not chunks:
            return "[ERROR: Empty translation]"

        # Join with space
        merged = " ".join(chunks)

        # Clean space before punctuation: " ." → "."
        merged = re.sub(r'\s+([.,;!?:])', r'\1', merged)

        # Clean multiple spaces: "  " → " "
        merged = re.sub(r'\s+', ' ', merged)

        return merged.strip()
