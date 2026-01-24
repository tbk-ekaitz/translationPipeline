"""
Document converter module using markitdown
Converts DOCX/PDF to Markdown format
"""
import os
from pathlib import Path
from markitdown import MarkItDown


class DocumentConverter:
    """Converts documents to Markdown using markitdown"""

    def __init__(self):
        self.converter = MarkItDown()

    def convert_to_markdown(self, input_path: str, output_path: str = None) -> str:
        """
        Convert a document to markdown

        Args:
            input_path: Path to input document (DOCX/PDF)
            output_path: Optional path for output markdown file

        Returns:
            Path to the generated markdown file
        """
        input_file = Path(input_path)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Convert document
        result = self.converter.convert(str(input_file))
        markdown_content = result.text_content

        # Determine output path
        if output_path is None:
            output_path = input_file.with_suffix('.md')

        # Write markdown file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(markdown_content, encoding='utf-8')

        return str(output_file)
