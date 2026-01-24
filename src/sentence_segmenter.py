"""
Sentence segmenter module
Splits markdown documents into individual sentences
Handles long sentences by chunking them intelligently
"""
import re
from typing import List, Optional
from dataclasses import dataclass, field
import nltk

# Download required NLTK data on first run
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


@dataclass
class Sentence:
    """Represents a sentence with translations and optional chunks"""
    id: int
    text: str
    translations: dict = field(default_factory=dict)
    chunks: Optional[List[str]] = None  # If sentence was split, contains chunks

    def add_translation(self, lang_code: str, translation: str):
        """Add a translation for a specific language"""
        self.translations[lang_code] = translation

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'original': self.text,
            **self.translations
        }


class SentenceSegmenter:
    """Segments markdown text into sentences with intelligent chunking"""

    def __init__(self, max_tokens: int = 3000):
        """
        Initialize segmenter

        Args:
            max_tokens: Maximum tokens per sentence chunk (default: 3000)
        """
        # Compatible with both old and new NLTK versions
        try:
            self.tokenizer = nltk.data.load('tokenizers/punkt_tab/english/sent_tokenizer.pickle')
        except LookupError:
            try:
                self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            except LookupError:
                # Fallback to basic sentence tokenizer
                from nltk.tokenize import sent_tokenize
                self.tokenizer = type('obj', (object,), {'tokenize': lambda self, text: sent_tokenize(text)})()
        self.max_tokens = max_tokens

    def segment(self, markdown_path: str) -> List[Sentence]:
        """
        Segment a markdown file into sentences

        Args:
            markdown_path: Path to markdown file

        Returns:
            List of Sentence objects in order of appearance
        """
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Clean markdown formatting while preserving text
        text = self._clean_markdown(content)

        # Tokenize into sentences
        raw_sentences = self.tokenizer.tokenize(text)

        # Create Sentence objects and split if necessary
        sentences = []
        for idx, sent in enumerate(raw_sentences, start=1):
            sent = sent.strip()
            if sent:  # Skip empty sentences
                sentence = Sentence(id=idx, text=sent)
                # Check if sentence needs splitting
                self._split_if_too_long(sentence)
                sentences.append(sentence)

        return sentences

    def _clean_markdown(self, content: str) -> str:
        """
        Clean markdown formatting while preserving text content

        Args:
            content: Raw markdown content

        Returns:
            Cleaned text
        """
        # Remove markdown headers but keep the text
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)

        # Remove markdown links but keep the text [text](url) -> text
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)

        # Remove bold/italic markers
        content = re.sub(r'\*\*([^\*]+)\*\*', r'\1', content)
        content = re.sub(r'\*([^\*]+)\*', r'\1', content)
        content = re.sub(r'__([^_]+)__', r'\1', content)
        content = re.sub(r'_([^_]+)_', r'\1', content)

        # Remove code blocks and inline code
        content = re.sub(r'```[^`]*```', '', content, flags=re.DOTALL)
        content = re.sub(r'`([^`]+)`', r'\1', content)

        # Remove horizontal rules
        content = re.sub(r'^-{3,}$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\*{3,}$', '', content, flags=re.MULTILINE)

        # Remove image tags ![alt](url)
        content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', content)

        # Clean up multiple newlines
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content.strip()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Estimate number of tokens in text
        Approximation: 1 token ≈ 0.75 words (4 characters)

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        word_count = len(text.split())
        return int(word_count / 0.75)

    def _split_sentence_smart(self, text: str) -> List[str]:
        """
        Split sentence intelligently at natural breakpoints
        Priority: . > , > ; > space
        Uses iterative approach (no recursion)

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        remaining = text

        while remaining:
            tokens = self._estimate_tokens(remaining)

            if tokens <= self.max_tokens:
                chunks.append(remaining.strip())
                break

            # Calculate safe character limit (approx 4 chars per token)
            safe_chars = int(self.max_tokens * 4)
            safe_text = remaining[:safe_chars]

            # Find best cut point (priority: . > , > ; > space)
            cut_points = [
                safe_text.rfind('. '),
                safe_text.rfind(', '),
                safe_text.rfind('; '),
                safe_text.rfind(' ')
            ]

            # Get the rightmost valid cut point
            cut_idx = max(cut_points)

            if cut_idx == -1:
                # No separator found, force cut at safe_chars
                cut_idx = safe_chars

            # Include the separator character (., ,, ;)
            if cut_idx < len(safe_text) - 1 and safe_text[cut_idx] in '.,;':
                cut_idx += 1

            # Add chunk and continue with remainder
            chunk = remaining[:cut_idx].strip()
            if chunk:
                chunks.append(chunk)

            remaining = remaining[cut_idx:].strip()

        return chunks

    def _split_if_too_long(self, sentence: Sentence):
        """
        Split sentence into chunks if it exceeds max_tokens
        Modifies sentence.chunks in-place

        Args:
            sentence: Sentence object to check and potentially split
        """
        tokens = self._estimate_tokens(sentence.text)

        if tokens > self.max_tokens:
            # Sentence is too long, split it
            sentence.chunks = self._split_sentence_smart(sentence.text)
        else:
            # Sentence is fine as-is
            sentence.chunks = None
