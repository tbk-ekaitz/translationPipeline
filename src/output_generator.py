"""
Output generator
Creates pretty-formatted JSON output
"""
import json
from typing import List
from pathlib import Path
from .sentence_segmenter import Sentence


class OutputGenerator:
    """Generates formatted output files"""

    @staticmethod
    def generate_json(sentences: List[Sentence], output_path: str):
        """
        Generate pretty-formatted JSON output

        Args:
            sentences: List of translated Sentence objects
            output_path: Path for output JSON file
        """
        # Convert sentences to dictionaries
        data = [sentence.to_dict() for sentence in sentences]

        # Create output directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write pretty-formatted JSON with proper indentation
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,  # Preserve Cyrillic and other Unicode characters
                indent=2,  # Pretty indentation with 2 spaces
                sort_keys=False  # Keep original order (id, original, russian, kazakh)
            )

        return output_path

    @staticmethod
    def generate_readable_text(sentences: List[Sentence], output_path: str):
        """
        Generate human-readable text format for easy copy-paste

        Args:
            sentences: List of translated Sentence objects
            output_path: Path for output text file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(f"[{sentence.id}]\n")
                f.write(f"Original:  {sentence.text}\n")
                if 'russian' in sentence.translations:
                    f.write(f"Russian:   {sentence.translations['russian']}\n")
                if 'kazakh' in sentence.translations:
                    f.write(f"Kazakh:    {sentence.translations['kazakh']}\n")
                f.write("\n" + "="*80 + "\n\n")

        return output_path
