#!/usr/bin/env python3
"""
Main translation pipeline script
Orchestrates document conversion, segmentation, translation, and output generation
"""
import argparse
import logging
import yaml
from pathlib import Path

from src import (
    DocumentConverter,
    SentenceSegmenter,
    OllamaClient,
    TranslationLLMManager,
    TranslationOrchestrator,
    OutputGenerator
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded"""
    import nltk

    resources = ['punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}...")
            nltk.download(resource, quiet=True)
            logger.info(f"✓ NLTK resource downloaded: {resource}")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Translation Pipeline: DOCX/PDF → Markdown → Sentences → Translations → JSON'
    )
    parser.add_argument(
        'input_file',
        help='Path to input document (DOCX or PDF)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file name (without extension)',
        default=None
    )
    parser.add_argument(
        '-c', '--config',
        help='Path to configuration file',
        default='config.yaml'
    )
    parser.add_argument(
        '--skip-conversion',
        action='store_true',
        help='Skip document conversion (input must be .md file)'
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Ensure NLTK resources are available
    ensure_nltk_resources()

    # Initialize components
    logger.info("Initializing pipeline components...")

    # Step 1: Document Conversion
    if args.skip_conversion:
        markdown_path = args.input_file
        logger.info(f"Skipping conversion, using markdown file: {markdown_path}")
    else:
        logger.info(f"Converting document: {args.input_file}")
        converter = DocumentConverter()
        markdown_path = converter.convert_to_markdown(args.input_file)
        logger.info(f"✓ Converted to markdown: {markdown_path}")

    # Step 2: Sentence Segmentation
    logger.info("Segmenting sentences...")
    max_tokens = config.get('segmentation', {}).get('max_tokens', 3000)
    segmenter = SentenceSegmenter(max_tokens=max_tokens)
    sentences = segmenter.segment(markdown_path)

    # Count chunked sentences
    chunked_count = sum(1 for s in sentences if s.chunks)
    logger.info(f"✓ Segmented into {len(sentences)} sentences ({chunked_count} split into chunks)")

    # Step 3: Initialize LLM clients
    logger.info("Initializing LLM clients...")

    # Russian model (Ollama)
    russian_config = config['llm']['russian']
    russian_client = OllamaClient(
        model_name=russian_config['model_name'],
        base_url=russian_config['base_url']
    )
    logger.info(f"✓ Russian model ready: {russian_config['model_name']}")

    # Kazakh model (Ollama)
    kazakh_config = config['llm']['kazakh']
    kazakh_client = OllamaClient(
        model_name=kazakh_config['model_name'],
        base_url=kazakh_config['base_url']
    )
    logger.info(f"✓ Kazakh model ready: {kazakh_config['model_name']}")

    # Create LLM manager
    llm_manager = TranslationLLMManager(russian_client, kazakh_client)

    # Step 4: Translation
    logger.info("Starting translation process...")
    orchestrator = TranslationOrchestrator(
        llm_manager,
        russian_prompt=config['prompts']['russian'],
        kazakh_prompt=config['prompts']['kazakh'],
        russian_batch_size=russian_config.get('batch_size', 32),
        kazakh_batch_size=kazakh_config.get('batch_size', 48)
    )
    translated_sentences = orchestrator.translate_sentences(sentences)

    # Step 5: Generate output
    logger.info("Generating output files...")

    # Determine output filename
    if args.output:
        output_base = args.output
    else:
        input_path = Path(args.input_file)
        output_base = input_path.stem + "_translated"

    output_dir = config['output']['directory']

    # Generate requested formats
    formats = config['output']['formats']

    if 'json' in formats:
        json_path = f"{output_dir}/{output_base}.json"
        OutputGenerator.generate_json(translated_sentences, json_path)
        logger.info(f"✓ JSON output: {json_path}")

    if 'text' in formats:
        text_path = f"{output_dir}/{output_base}.txt"
        OutputGenerator.generate_readable_text(translated_sentences, text_path)
        logger.info(f"✓ Text output: {text_path}")

    logger.info("🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main()
