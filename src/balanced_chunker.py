"""
Balanced Chunker Module
Creates balanced chunks across a batch for optimal parallel processing.
Uses greedy algorithm with weighted cut points (. > ; > , > space).
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import re


@dataclass
class CutPoint:
    """Represents a potential cut point in text"""
    position: int       # Character position
    char: str           # The separator character
    weight: float       # Priority weight (higher = better cut point)
    token_count: int    # Estimated tokens up to this point


@dataclass
class ChunkInfo:
    """Metadata for a chunk to enable reassembly"""
    cell_idx: int       # Original cell index in batch
    chunk_idx: int      # Chunk number within cell (0, 1, 2...)
    total_chunks: int   # Total chunks for this cell
    text: str           # The chunk text
    start_pos: int      # Start position in original text
    end_pos: int        # End position in original text


@dataclass
class BalancedBatch:
    """Result of balanced chunking for a batch"""
    chunks: List[ChunkInfo]           # All chunks ready for translation
    cell_count: int                   # Original number of cells
    total_tokens: int                 # Total estimated tokens
    avg_chunk_tokens: int             # Average tokens per chunk
    max_chunk_tokens: int             # Max tokens in any chunk
    min_chunk_tokens: int             # Min tokens in any chunk (excluding empty)


class BalancedChunker:
    """Creates balanced chunks across a batch for parallel processing"""

    # Default weights for cut points (higher = better place to cut)
    DEFAULT_WEIGHTS = {
        '.': 1.0,    # Period - best cut point
        '!': 1.0,    # Exclamation
        '?': 1.0,    # Question mark
        ';': 0.7,    # Semicolon
        ':': 0.6,    # Colon
        ',': 0.4,    # Comma
        ' ': 0.1,    # Space - last resort
    }

    def __init__(
        self,
        max_tokens: int = 3000,
        target_ratio: float = 0.8,
        cut_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize balanced chunker.

        Args:
            max_tokens: Maximum tokens per chunk (hard limit)
            target_ratio: Target chunk size as ratio of max_tokens (0.8 = 80%)
            cut_weights: Custom weights for cut point characters
        """
        self.max_tokens = max_tokens
        self.target_tokens = int(max_tokens * target_ratio)
        self.cut_weights = cut_weights or self.DEFAULT_WEIGHTS

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate number of tokens in text.
        Approximation: ~0.75 words per token (or ~4 chars per token)

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        word_count = len(text.split())
        return max(1, int(word_count / 0.75))

    def find_cut_points(self, text: str) -> List[CutPoint]:
        """
        Find all potential cut points in text with their weights.

        Args:
            text: Text to analyze

        Returns:
            List of CutPoint objects sorted by position
        """
        cut_points = []
        running_tokens = 0
        last_pos = 0

        for i, char in enumerate(text):
            if char in self.cut_weights:
                # Check if followed by space (for punctuation) or is a space
                is_valid_cut = (
                    char == ' ' or
                    (i + 1 < len(text) and text[i + 1] == ' ') or
                    i + 1 == len(text)
                )

                if is_valid_cut:
                    # Estimate tokens up to this point
                    segment = text[last_pos:i + 1]
                    segment_tokens = self.estimate_tokens(segment)
                    running_tokens += segment_tokens
                    last_pos = i + 1

                    cut_points.append(CutPoint(
                        position=i + 1,  # Position after the character
                        char=char,
                        weight=self.cut_weights[char],
                        token_count=running_tokens
                    ))

        # Add end of text as final cut point
        if text and (not cut_points or cut_points[-1].position < len(text)):
            remaining_tokens = self.estimate_tokens(text[last_pos:])
            cut_points.append(CutPoint(
                position=len(text),
                char='END',
                weight=1.0,
                token_count=running_tokens + remaining_tokens
            ))

        return cut_points

    def chunk_text_balanced(self, text: str, target_tokens: Optional[int] = None) -> List[Tuple[str, int, int]]:
        """
        Split text into balanced chunks using greedy algorithm.

        Args:
            text: Text to chunk
            target_tokens: Target tokens per chunk (uses self.target_tokens if None)

        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        if not text or not text.strip():
            return []

        target = target_tokens or self.target_tokens
        total_tokens = self.estimate_tokens(text)

        # If text fits in target, return as single chunk
        if total_tokens <= target:
            return [(text.strip(), 0, len(text))]

        cut_points = self.find_cut_points(text)

        if not cut_points:
            return [(text.strip(), 0, len(text))]

        chunks = []
        chunk_start = 0
        current_tokens = 0

        for i, cp in enumerate(cut_points):
            segment_tokens = cp.token_count - current_tokens

            # Check if adding this segment would exceed max
            if cp.token_count - (chunks[-1][1] if chunks else 0) > self.max_tokens:
                # Find best cut point before this one
                best_cut = self._find_best_cut_before(
                    cut_points[:i],
                    chunk_start,
                    target
                )

                if best_cut:
                    chunk_text = text[chunk_start:best_cut.position].strip()
                    if chunk_text:
                        chunks.append((chunk_text, chunk_start, best_cut.position))
                    chunk_start = best_cut.position
                    current_tokens = best_cut.token_count

            # Check if we've reached target and this is a good cut point
            tokens_since_chunk_start = cp.token_count - current_tokens
            if tokens_since_chunk_start >= target and cp.weight >= 0.4:
                chunk_text = text[chunk_start:cp.position].strip()
                if chunk_text:
                    chunks.append((chunk_text, chunk_start, cp.position))
                chunk_start = cp.position
                current_tokens = cp.token_count

        # Add remaining text as final chunk
        if chunk_start < len(text):
            remaining = text[chunk_start:].strip()
            if remaining:
                chunks.append((remaining, chunk_start, len(text)))

        return chunks if chunks else [(text.strip(), 0, len(text))]

    def _find_best_cut_before(
        self,
        cut_points: List[CutPoint],
        chunk_start_tokens: int,
        target: int
    ) -> Optional[CutPoint]:
        """
        Find the best cut point that keeps chunk under target.
        Prioritizes high-weight cut points close to target.

        Args:
            cut_points: Available cut points
            chunk_start_tokens: Token count at chunk start
            target: Target tokens for chunk

        Returns:
            Best CutPoint or None
        """
        if not cut_points:
            return None

        best_cut = None
        best_score = -1

        for cp in cut_points:
            tokens_for_chunk = cp.token_count - chunk_start_tokens
            if tokens_for_chunk <= 0:
                continue

            if tokens_for_chunk <= self.max_tokens:
                # Score based on: closeness to target + weight
                distance_ratio = 1 - abs(tokens_for_chunk - target) / target
                score = (distance_ratio * 0.6) + (cp.weight * 0.4)

                if score > best_score:
                    best_score = score
                    best_cut = cp

        return best_cut

    def process_batch(self, texts: List[str]) -> BalancedBatch:
        """
        Process a batch of texts into balanced chunks.

        Args:
            texts: List of texts to process

        Returns:
            BalancedBatch with all chunks and metadata
        """
        all_chunks: List[ChunkInfo] = []
        total_tokens = 0
        chunk_token_counts = []

        for cell_idx, text in enumerate(texts):
            if not text or not str(text).strip() or str(text).lower() == 'nan':
                # Empty cell - add placeholder
                all_chunks.append(ChunkInfo(
                    cell_idx=cell_idx,
                    chunk_idx=0,
                    total_chunks=1,
                    text="",
                    start_pos=0,
                    end_pos=0
                ))
                continue

            text = str(text).strip()
            text_chunks = self.chunk_text_balanced(text)

            for chunk_idx, (chunk_text, start_pos, end_pos) in enumerate(text_chunks):
                chunk_tokens = self.estimate_tokens(chunk_text)
                total_tokens += chunk_tokens
                if chunk_tokens > 0:
                    chunk_token_counts.append(chunk_tokens)

                all_chunks.append(ChunkInfo(
                    cell_idx=cell_idx,
                    chunk_idx=chunk_idx,
                    total_chunks=len(text_chunks),
                    text=chunk_text,
                    start_pos=start_pos,
                    end_pos=end_pos
                ))

        return BalancedBatch(
            chunks=all_chunks,
            cell_count=len(texts),
            total_tokens=total_tokens,
            avg_chunk_tokens=int(total_tokens / len(chunk_token_counts)) if chunk_token_counts else 0,
            max_chunk_tokens=max(chunk_token_counts) if chunk_token_counts else 0,
            min_chunk_tokens=min(chunk_token_counts) if chunk_token_counts else 0
        )

    def reassemble_translations(
        self,
        batch: BalancedBatch,
        translations: List[str]
    ) -> List[str]:
        """
        Reassemble translated chunks back into original cell structure.

        Args:
            batch: The BalancedBatch that was translated
            translations: List of translations matching batch.chunks order

        Returns:
            List of translations matching original cell order
        """
        if len(translations) != len(batch.chunks):
            raise ValueError(
                f"Translation count ({len(translations)}) doesn't match "
                f"chunk count ({len(batch.chunks)})"
            )

        # Group translations by cell_idx
        cell_translations: Dict[int, List[Tuple[int, str]]] = {}

        for chunk_info, translation in zip(batch.chunks, translations):
            cell_idx = chunk_info.cell_idx
            if cell_idx not in cell_translations:
                cell_translations[cell_idx] = []
            cell_translations[cell_idx].append((chunk_info.chunk_idx, translation))

        # Reassemble in order
        result = []
        for cell_idx in range(batch.cell_count):
            if cell_idx not in cell_translations:
                result.append("")
                continue

            # Sort by chunk_idx and merge
            chunks = sorted(cell_translations[cell_idx], key=lambda x: x[0])
            merged = self._merge_translations([t for _, t in chunks])
            result.append(merged)

        return result

    def _merge_translations(self, translations: List[str]) -> str:
        """
        Merge translated chunks intelligently.

        Args:
            translations: List of translated chunk texts

        Returns:
            Merged translation
        """
        # Filter empty and clean
        chunks = [t.strip() for t in translations if t and t.strip()]

        if not chunks:
            return ""

        if len(chunks) == 1:
            return chunks[0]

        # Join with space, avoiding double spaces
        merged = " ".join(chunks)
        merged = re.sub(r'\s+', ' ', merged)

        return merged.strip()


def create_chunker_from_config(config: dict) -> BalancedChunker:
    """
    Create a BalancedChunker from config dictionary.

    Args:
        config: Configuration dictionary with 'segmentation' section

    Returns:
        Configured BalancedChunker instance
    """
    seg_config = config.get('segmentation', {})

    max_tokens = seg_config.get('max_tokens', 3000)
    target_ratio = seg_config.get('target_ratio', 0.8)

    cut_weights = None
    if 'cut_weights' in seg_config:
        cut_weights = {
            '.': seg_config['cut_weights'].get('period', 1.0),
            '!': seg_config['cut_weights'].get('period', 1.0),
            '?': seg_config['cut_weights'].get('period', 1.0),
            ';': seg_config['cut_weights'].get('semicolon', 0.7),
            ':': seg_config['cut_weights'].get('colon', 0.6),
            ',': seg_config['cut_weights'].get('comma', 0.4),
            ' ': seg_config['cut_weights'].get('space', 0.1),
        }

    return BalancedChunker(
        max_tokens=max_tokens,
        target_ratio=target_ratio,
        cut_weights=cut_weights
    )
