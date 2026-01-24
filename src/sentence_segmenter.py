"""
Sentence segmenter module
Splits markdown documents into individual sentences
"""
import re
from typing import List
from dataclasses import dataclass, field
import nltk

# Download required NLTK data on first run
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


@dataclass
class Sentence:
    """Represents a sentence with translations"""
    id: int
    text: str
    translations: dict = field(default_factory=dict)

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
    """Segments markdown text into sentences"""

    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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

        # Create Sentence objects
        sentences = []
        for idx, sent in enumerate(raw_sentences, start=1):
            sent = sent.strip()
            if sent:  # Skip empty sentences
                sentences.append(Sentence(id=idx, text=sent))

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
