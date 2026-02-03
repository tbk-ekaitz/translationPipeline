#!/usr/bin/env python3
"""
Excel/CSV Translation Pipeline
Translates specific columns from xlsx/csv files to Russian and Kazakh
Supports multi-sheet xlsx files and custom column selection
"""
import argparse
import logging
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from dataclasses import dataclass, field

import pandas as pd

from src.llm_client import OllamaClient, TranslationLLMManager
from src.sentence_segmenter import SentenceSegmenter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hardcoded column tuples to search for (case-sensitive, both must exist)
COLUMN_TUPLES = [
    ("en-title", "en-content"),
    ("EN_title", "EN_content"),
]


@dataclass
class CellItem:
    """Represents a cell with its translation and optional chunks"""
    row_idx: int
    text: str
    translations: dict = field(default_factory=dict)
    chunks: Optional[List[str]] = None

    def add_translation(self, lang_code: str, translation: str):
        """Add a translation for a specific language"""
        self.translations[lang_code] = translation


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
            logger.info(f"NLTK resource downloaded: {resource}")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def read_excel_or_csv(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Read xlsx or csv file into DataFrames (one per sheet)

    Args:
        file_path: Path to the input file

    Returns:
        Dict mapping sheet names to DataFrames
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == '.xlsx':
        # Read all sheets
        xlsx = pd.ExcelFile(file_path)
        sheets = {}
        for sheet_name in xlsx.sheet_names:
            sheets[sheet_name] = pd.read_excel(xlsx, sheet_name=sheet_name)
        return sheets
    elif ext == '.csv':
        # CSV has only one "sheet"
        return {"Sheet1": pd.read_csv(file_path)}
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .xlsx or .csv")


def find_matching_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """
    Find a matching column tuple from the hardcoded list

    Args:
        df: DataFrame to search in

    Returns:
        Tuple of (title_col, content_col) if found, None otherwise
    """
    columns = set(df.columns)

    for title_col, content_col in COLUMN_TUPLES:
        if title_col in columns and content_col in columns:
            return (title_col, content_col)

    return None


def validate_custom_columns(df: pd.DataFrame, custom_columns: List[str]) -> List[str]:
    """
    Validate that custom columns exist in the DataFrame

    Args:
        df: DataFrame to check
        custom_columns: List of column names to validate

    Returns:
        List of valid column names that exist in the DataFrame
    """
    df_columns = set(df.columns)
    valid_columns = [col for col in custom_columns if col in df_columns]
    return valid_columns


def get_output_column_names_for_tuple(source_tuple: Tuple[str, str]) -> dict:
    """
    Generate output column names based on the source tuple style

    Args:
        source_tuple: The matched source column tuple

    Returns:
        Dict with column name mappings
    """
    title_col, content_col = source_tuple

    # Detect naming style based on the title column
    if title_col == "en-title":
        # Lowercase with hyphen style
        return {
            "columns": [
                {"source": title_col, "ru": "ru-title", "kz": "kz-title"},
                {"source": content_col, "ru": "ru-content", "kz": "kz-content"},
            ]
        }
    else:
        # Uppercase with underscore style (EN_title)
        return {
            "columns": [
                {"source": title_col, "ru": "RU_title", "kz": "KZ_title"},
                {"source": content_col, "ru": "RU_content", "kz": "KZ_content"},
            ]
        }


def get_output_column_names_for_custom(custom_columns: List[str]) -> dict:
    """
    Generate output column names for custom columns

    Args:
        custom_columns: List of custom column names

    Returns:
        Dict with column name mappings
    """
    columns = []
    for col in custom_columns:
        columns.append({
            "source": col,
            "ru": f"{col}_RU",
            "kz": f"{col}_KZ",
        })
    return {"columns": columns}


def extract_cells_from_column(df: pd.DataFrame, column_name: str, max_tokens: int) -> List[CellItem]:
    """
    Extract cells from a column and prepare them for translation

    Args:
        df: Source DataFrame
        column_name: Name of the column to extract
        max_tokens: Maximum tokens per chunk

    Returns:
        List of CellItem objects
    """
    segmenter = SentenceSegmenter(max_tokens=max_tokens)
    cells = []

    for idx, value in enumerate(df[column_name]):
        # Convert to string and handle NaN/None
        text = str(value) if pd.notna(value) else ""
        text = text.strip()

        cell = CellItem(row_idx=idx, text=text)

        # Check if cell needs chunking (same logic as sentence segmenter)
        if text:
            tokens = segmenter._estimate_tokens(text)
            if tokens > max_tokens:
                cell.chunks = segmenter._split_sentence_smart(text)

        cells.append(cell)

    return cells


def translate_cells_batched(
    cells: List[CellItem],
    language: str,
    batch_size: int,
    translate_func,
    prompt_template: str,
    column_name: str = ""
):
    """
    Translate cells in batches (modifies cells in-place)

    Args:
        cells: List of CellItem objects
        language: Target language name ('russian' or 'kazakh')
        batch_size: Size of each batch
        translate_func: Translation function to use
        prompt_template: Prompt template for translation
        column_name: Name of the column being translated (for progress display)
    """
    from tqdm import tqdm
    import re

    total = len(cells)
    num_batches = (total + batch_size - 1) // batch_size

    desc = f"[{column_name}] -> {language.capitalize()}" if column_name else f"Translating to {language.capitalize()}"
    progress_bar = tqdm(
        range(num_batches),
        desc=desc,
        unit="batch",
        ncols=100
    )

    for batch_idx in progress_bar:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_cells = cells[start_idx:end_idx]

        progress_bar.set_postfix({
            'cells': f'{start_idx + 1}-{end_idx}/{total}'
        })

        # Expand chunks for batch
        batch_items = []
        chunk_metadata = []

        for local_idx, cell in enumerate(batch_cells):
            global_idx = start_idx + local_idx

            if not cell.text:
                # Empty cell, skip translation
                chunk_metadata.append((global_idx, False, 0, True))  # is_empty=True
            elif cell.chunks:
                batch_items.extend(cell.chunks)
                chunk_metadata.append((global_idx, True, len(cell.chunks), False))
            else:
                batch_items.append(cell.text)
                chunk_metadata.append((global_idx, False, 1, False))

        # Translate batch if there are items
        if batch_items:
            try:
                translations = translate_func(batch_items, prompt_template)

                # Merge chunks and assign back
                trans_idx = 0
                for global_idx, is_chunked, num_chunks, is_empty in chunk_metadata:
                    if is_empty:
                        cells[global_idx].add_translation(language, "")
                    elif is_chunked:
                        chunk_trans = translations[trans_idx:trans_idx + num_chunks]
                        merged = merge_chunk_translations(chunk_trans)
                        cells[global_idx].add_translation(language, merged)
                        trans_idx += num_chunks
                    else:
                        cells[global_idx].add_translation(language, translations[trans_idx])
                        trans_idx += 1

            except Exception as e:
                logger.error(f"Batch translation failed: {e}")
                for global_idx, _, _, is_empty in chunk_metadata:
                    if not is_empty:
                        cells[global_idx].add_translation(language, f"[ERROR: {str(e)}]")
        else:
            # All cells in batch were empty
            for global_idx, _, _, _ in chunk_metadata:
                cells[global_idx].add_translation(language, "")


def merge_chunk_translations(chunk_trans: List[str]) -> str:
    """
    Merge chunk translations intelligently

    Args:
        chunk_trans: List of translated chunks

    Returns:
        Merged translation
    """
    import re

    chunks = [t.strip() for t in chunk_trans if t.strip()]

    if not chunks:
        return "[ERROR: Empty translation]"

    merged = " ".join(chunks)
    merged = re.sub(r'\s+([.,;!?:])', r'\1', merged)
    merged = re.sub(r'\s+', ' ', merged)

    return merged.strip()


def process_sheet(
    df: pd.DataFrame,
    sheet_name: str,
    column_mappings: dict,
    max_tokens: int,
    llm_manager: TranslationLLMManager,
    russian_config: dict,
    kazakh_config: dict,
    russian_prompt: str,
    kazakh_prompt: str
) -> pd.DataFrame:
    """
    Process a single sheet through the translation pipeline

    Args:
        df: Source DataFrame
        sheet_name: Name of the sheet (for logging)
        column_mappings: Dict with column name mappings
        max_tokens: Maximum tokens per chunk
        llm_manager: Translation LLM manager
        russian_config: Russian model config (with batch_size_title/content)
        kazakh_config: Kazakh model config (with batch_size_title/content)
        russian_prompt: Prompt for Russian translation
        kazakh_prompt: Prompt for Kazakh translation

    Returns:
        DataFrame with translated columns
    """
    result_data = {}

    for col_info in column_mappings["columns"]:
        source_col = col_info["source"]
        ru_col = col_info["ru"]
        kz_col = col_info["kz"]

        # Determine batch size based on column type (title vs content)
        is_title = "title" in source_col.lower()
        if is_title:
            ru_batch = russian_config.get('batch_size_title', russian_config.get('batch_size', 128))
            kz_batch = kazakh_config.get('batch_size_title', kazakh_config.get('batch_size', 64))
            col_type = "title"
        else:
            ru_batch = russian_config.get('batch_size_content', russian_config.get('batch_size', 64))
            kz_batch = kazakh_config.get('batch_size_content', kazakh_config.get('batch_size', 20))
            col_type = "content"

        logger.info(f"\n=== [{sheet_name}] Processing column: {source_col} ({col_type}) ===")
        logger.info(f"Batch sizes: Russian={ru_batch}, Kazakh={kz_batch}")

        # Extract cells
        cells = extract_cells_from_column(df, source_col, max_tokens)
        chunked_count = sum(1 for c in cells if c.chunks)
        logger.info(f"Extracted {len(cells)} cells ({chunked_count} chunked)")

        # Translate to Russian
        translate_cells_batched(
            cells,
            'russian',
            ru_batch,
            llm_manager.translate_batch_to_russian,
            russian_prompt,
            column_name=source_col
        )

        # Translate to Kazakh
        translate_cells_batched(
            cells,
            'kazakh',
            kz_batch,
            llm_manager.translate_batch_to_kazakh,
            kazakh_prompt,
            column_name=source_col
        )

        # Store results
        result_data[ru_col] = [c.translations.get("russian", "") for c in cells]
        result_data[kz_col] = [c.translations.get("kazakh", "") for c in cells]

    return pd.DataFrame(result_data)


def save_multi_sheet_excel(
    sheets_data: Dict[str, pd.DataFrame],
    output_path: str
):
    """
    Save multiple sheets to an Excel file

    Args:
        sheets_data: Dict mapping sheet names to DataFrames
        output_path: Path for output file
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in sheets_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Excel/CSV Translation Pipeline: XLSX/CSV -> Translations -> XLSX'
    )
    parser.add_argument(
        'input_file',
        help='Path to input file (XLSX or CSV)'
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
        '--columns',
        nargs='+',
        help='Custom columns to translate (overrides default tuple search). '
             'Example: --columns "Title" "Description" "Notes"',
        default=None
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Ensure NLTK resources are available
    ensure_nltk_resources()

    # Step 1: Read input file (all sheets)
    logger.info(f"Reading input file: {args.input_file}")
    sheets = read_excel_or_csv(args.input_file)
    logger.info(f"Found {len(sheets)} sheet(s): {list(sheets.keys())}")

    # Step 2: Initialize LLM clients
    logger.info("Initializing LLM clients...")

    # Global max_workers (should match OLLAMA_NUM_PARALLEL)
    max_workers = config['llm'].get('max_workers', 16)
    logger.info(f"Using max_workers={max_workers} (ensure OLLAMA_NUM_PARALLEL={max_workers})")

    russian_config = config['llm']['russian']
    russian_client = OllamaClient(
        model_name=russian_config['model_name'],
        base_url=russian_config['base_url'],
        max_workers=max_workers
    )
    logger.info(f"Russian model ready: {russian_config['model_name']}")

    kazakh_config = config['llm']['kazakh']
    kazakh_client = OllamaClient(
        model_name=kazakh_config['model_name'],
        base_url=kazakh_config['base_url'],
        max_workers=max_workers
    )
    logger.info(f"Kazakh model ready: {kazakh_config['model_name']}")

    llm_manager = TranslationLLMManager(russian_client, kazakh_client)

    russian_prompt = config['prompts']['russian']
    kazakh_prompt = config['prompts']['kazakh']
    max_tokens = config.get('segmentation', {}).get('max_tokens', 3000)

    # Step 3: Process each sheet
    output_sheets = {}
    use_custom_columns = args.columns is not None

    for sheet_name, df in sheets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing sheet: {sheet_name} ({len(df)} rows)")
        logger.info(f"{'='*60}")

        # Determine columns to translate
        if use_custom_columns:
            # Custom column mode
            valid_columns = validate_custom_columns(df, args.columns)
            if not valid_columns:
                logger.warning(f"[{sheet_name}] No matching columns found. Skipping sheet.")
                logger.warning(f"  Requested: {args.columns}")
                logger.warning(f"  Available: {list(df.columns)}")
                continue

            if len(valid_columns) != len(args.columns):
                missing = set(args.columns) - set(valid_columns)
                logger.warning(f"[{sheet_name}] Some columns not found: {missing}")

            column_mappings = get_output_column_names_for_custom(valid_columns)
            logger.info(f"Using custom columns: {valid_columns}")
        else:
            # Default tuple mode
            matched_tuple = find_matching_columns(df)

            if matched_tuple is None:
                logger.warning(f"[{sheet_name}] No matching column tuple found. Skipping sheet.")
                logger.warning(f"  Expected one of: {COLUMN_TUPLES}")
                logger.warning(f"  Found: {list(df.columns)}")
                continue

            column_mappings = get_output_column_names_for_tuple(matched_tuple)
            logger.info(f"Found matching columns: {matched_tuple}")

        # Process the sheet
        result_df = process_sheet(
            df,
            sheet_name,
            column_mappings,
            max_tokens,
            llm_manager,
            russian_config,
            kazakh_config,
            russian_prompt,
            kazakh_prompt
        )

        output_sheets[sheet_name] = result_df

    # Step 4: Save output
    if not output_sheets:
        logger.error("No sheets were processed! Check that your input file has the required columns.")
        print("\n[ERROR] No sheets were processed.")
        if use_custom_columns:
            print(f"Requested columns: {args.columns}")
        else:
            print(f"Required column pairs: {COLUMN_TUPLES}")
        return

    logger.info("\nGenerating output file...")

    if args.output:
        output_base = args.output
    else:
        input_path = Path(args.input_file)
        output_base = input_path.stem + "_translated"

    output_dir = config['output']['directory']
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_path = f"{output_dir}/{output_base}.xlsx"
    save_multi_sheet_excel(output_sheets, output_path)

    logger.info(f"\nPipeline completed successfully!")
    logger.info(f"Processed {len(output_sheets)} sheet(s)")


if __name__ == "__main__":
    main()
