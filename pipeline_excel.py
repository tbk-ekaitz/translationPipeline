#!/usr/bin/env python3
"""
Excel/CSV Translation Pipeline using Yandex Translate API
Translates specific columns from xlsx/csv files to Russian and Kazakh
Supports multi-sheet xlsx files and custom column selection
"""
import argparse
import logging
import time
import yaml
import requests
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

import pandas as pd
from tqdm import tqdm

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

# Yandex API limits (official)
MAX_CHARS_PER_STRING = 2000       # Max chars per individual text
MAX_CHARS_PER_REQUEST = 10000     # Max total chars per API request

# Retry configuration
MAX_RETRIES = 5
BASE_RETRY_DELAY = 1.0  # Start with 1 second


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


class YandexTranslateClient:
    """Client for Yandex Translate API with automatic retry on rate limits"""

    API_URL = "https://translate.api.cloud.yandex.net/translate/v2/translate"

    def __init__(self, api_key: str, folder_id: str):
        self.api_key = api_key
        self.folder_id = folder_id

    def _wait_and_retry(self, response: requests.Response, attempt: int) -> float:
        """
        Calculate wait time when rate limited

        Args:
            response: The 429 response from Yandex
            attempt: Current retry attempt number

        Returns:
            Seconds to wait before retry
        """
        # Try to get Retry-After header from Yandex
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass

        # Fallback: exponential backoff
        return BASE_RETRY_DELAY * (2 ** attempt)

    def translate(self, texts: List[str], target_lang: str, source_lang: str = "en") -> List[str]:
        """
        Translate a list of texts to target language with automatic retry

        Args:
            texts: List of texts to translate
            target_lang: Target language code (ru, kk)
            source_lang: Source language code (default: en)

        Returns:
            List of translated texts
        """
        if not texts:
            return []

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }

        payload = {
            "folderId": self.folder_id,
            "texts": texts,
            "targetLanguageCode": target_lang,
            "sourceLanguageCode": source_lang
        }

        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(self.API_URL, headers=headers, json=payload, timeout=60)

                # Handle rate limiting (429)
                if response.status_code == 429:
                    wait_time = self._wait_and_retry(response, attempt)
                    logger.warning(f"Rate limited by Yandex. Waiting {wait_time:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                    continue

                # Handle other errors
                response.raise_for_status()

                result = response.json()
                translations = [t["text"] for t in result.get("translations", [])]

                if len(translations) != len(texts):
                    logger.warning(f"Translation count mismatch: sent {len(texts)}, got {len(translations)}")

                return translations

            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    wait_time = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                continue

        # All retries exhausted
        logger.error(f"Yandex API request failed after {MAX_RETRIES} attempts")
        if last_exception:
            raise last_exception
        raise Exception("Translation failed: rate limit exceeded")


class TranslationManager:
    """Manages translations to multiple languages"""

    def __init__(self, client: YandexTranslateClient):
        self.client = client

    def translate_batch(self, texts: List[str], target_lang: str) -> List[str]:
        """Translate a batch of texts"""
        return self.client.translate(texts, target_lang)


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
        xlsx = pd.ExcelFile(file_path)
        sheets = {}
        for sheet_name in xlsx.sheet_names:
            sheets[sheet_name] = pd.read_excel(xlsx, sheet_name=sheet_name)
        return sheets
    elif ext == '.csv':
        return {"Sheet1": pd.read_csv(file_path)}
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .xlsx or .csv")


def find_matching_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """Find a matching column tuple from the hardcoded list"""
    columns = set(df.columns)
    for title_col, content_col in COLUMN_TUPLES:
        if title_col in columns and content_col in columns:
            return (title_col, content_col)
    return None


def validate_custom_columns(df: pd.DataFrame, custom_columns: List[str]) -> List[str]:
    """Validate that custom columns exist in the DataFrame"""
    df_columns = set(df.columns)
    return [col for col in custom_columns if col in df_columns]


def get_output_column_names_for_tuple(source_tuple: Tuple[str, str]) -> dict:
    """Generate output column names based on the source tuple style"""
    title_col, content_col = source_tuple

    if title_col == "en-title":
        return {
            "columns": [
                {"source": title_col, "ru": "ru-title", "kz": "kz-title"},
                {"source": content_col, "ru": "ru-content", "kz": "kz-content"},
            ]
        }
    else:
        return {
            "columns": [
                {"source": title_col, "ru": "RU_title", "kz": "KZ_title"},
                {"source": content_col, "ru": "RU_content", "kz": "KZ_content"},
            ]
        }


def get_output_column_names_for_custom(custom_columns: List[str]) -> dict:
    """Generate output column names for custom columns"""
    columns = []
    for col in custom_columns:
        columns.append({
            "source": col,
            "ru": f"{col}_RU",
            "kz": f"{col}_KZ",
        })
    return {"columns": columns}


def split_text_by_chars(text: str, max_chars: int = MAX_CHARS_PER_STRING) -> List[str]:
    """
    Split text into chunks that fit within character limit

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text

    # Priority separators for splitting
    separators = ['. ', ', ', '; ', ' ', '']

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        # Find best split point
        chunk = remaining[:max_chars]
        split_pos = max_chars

        for sep in separators:
            if sep:
                pos = chunk.rfind(sep)
                if pos > max_chars // 2:  # Don't split too early
                    split_pos = pos + len(sep)
                    break
            else:
                # Force split at max_chars
                split_pos = max_chars

        chunks.append(remaining[:split_pos].strip())
        remaining = remaining[split_pos:].strip()

    return [c for c in chunks if c]


def extract_cells_from_column(df: pd.DataFrame, column_name: str) -> List[CellItem]:
    """
    Extract cells from a column and prepare them for translation

    Args:
        df: Source DataFrame
        column_name: Name of the column to extract

    Returns:
        List of CellItem objects
    """
    cells = []

    for idx, value in enumerate(df[column_name]):
        text = str(value) if pd.notna(value) else ""
        text = text.strip()

        cell = CellItem(row_idx=idx, text=text)

        # Check if cell needs chunking (char limit)
        if text and len(text) > MAX_CHARS_PER_STRING:
            cell.chunks = split_text_by_chars(text)

        cells.append(cell)

    return cells


def create_batches(items: List[str], max_chars: int = MAX_CHARS_PER_REQUEST) -> List[List[str]]:
    """
    Group items into batches that fit within character limit per request

    Args:
        items: List of text items
        max_chars: Maximum total characters per batch

    Returns:
        List of batches (each batch is a list of strings)
    """
    batches = []
    current_batch = []
    current_chars = 0

    for item in items:
        item_chars = len(item)

        # If single item exceeds limit, it goes alone (already chunked)
        if item_chars > max_chars:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            batches.append([item])
            continue

        # Check if adding this item exceeds limit
        if current_chars + item_chars > max_chars:
            if current_batch:
                batches.append(current_batch)
            current_batch = [item]
            current_chars = item_chars
        else:
            current_batch.append(item)
            current_chars += item_chars

    if current_batch:
        batches.append(current_batch)

    return batches


def merge_chunk_translations(chunk_trans: List[str]) -> str:
    """Merge chunk translations intelligently"""
    import re

    chunks = [t.strip() for t in chunk_trans if t.strip()]
    if not chunks:
        return "[ERROR: Empty translation]"

    merged = " ".join(chunks)
    merged = re.sub(r'\s+([.,;!?:])', r'\1', merged)
    merged = re.sub(r'\s+', ' ', merged)

    return merged.strip()


def translate_cells(
    cells: List[CellItem],
    target_lang: str,
    lang_name: str,
    manager: TranslationManager,
    column_name: str = ""
):
    """
    Translate cells using Yandex API with smart batching

    Args:
        cells: List of CellItem objects
        target_lang: Target language code (ru, kk)
        lang_name: Language name for display (russian, kazakh)
        manager: Translation manager
        column_name: Column name for progress display
    """
    # Build flat list of items to translate with metadata
    items_to_translate = []
    metadata = []  # (cell_idx, is_chunked, chunk_count, is_empty)

    for idx, cell in enumerate(cells):
        if not cell.text:
            metadata.append((idx, False, 0, True))
        elif cell.chunks:
            items_to_translate.extend(cell.chunks)
            metadata.append((idx, True, len(cell.chunks), False))
        else:
            items_to_translate.append(cell.text)
            metadata.append((idx, False, 1, False))

    if not items_to_translate:
        # All cells are empty
        for idx, _, _, _ in metadata:
            cells[idx].add_translation(lang_name, "")
        return

    # Create batches
    batches = create_batches(items_to_translate)

    desc = f"[{column_name}] -> {lang_name.capitalize()}" if column_name else f"-> {lang_name.capitalize()}"

    # Translate batches
    all_translations = []
    total_chars = sum(len(item) for item in items_to_translate)

    progress_bar = tqdm(
        batches,
        desc=desc,
        unit="batch",
        ncols=100
    )

    for batch in progress_bar:
        batch_chars = sum(len(t) for t in batch)
        progress_bar.set_postfix({
            'items': len(batch),
            'chars': f'{batch_chars}'
        })

        try:
            translations = manager.translate_batch(batch, target_lang)
            all_translations.extend(translations)
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            all_translations.extend([f"[ERROR: {str(e)}]"] * len(batch))

    # Assign translations back to cells
    trans_idx = 0
    for cell_idx, is_chunked, chunk_count, is_empty in metadata:
        if is_empty:
            cells[cell_idx].add_translation(lang_name, "")
        elif is_chunked:
            chunk_trans = all_translations[trans_idx:trans_idx + chunk_count]
            merged = merge_chunk_translations(chunk_trans)
            cells[cell_idx].add_translation(lang_name, merged)
            trans_idx += chunk_count
        else:
            cells[cell_idx].add_translation(lang_name, all_translations[trans_idx])
            trans_idx += 1

    logger.info(f"Translated {len(items_to_translate)} items ({total_chars} chars) in {len(batches)} batches")


def process_sheet(
    df: pd.DataFrame,
    sheet_name: str,
    column_mappings: dict,
    manager: TranslationManager
) -> pd.DataFrame:
    """
    Process a single sheet through the translation pipeline

    Args:
        df: Source DataFrame
        sheet_name: Name of the sheet (for logging)
        column_mappings: Dict with column name mappings
        manager: Translation manager

    Returns:
        DataFrame with translated columns
    """
    result_data = {}

    for col_info in column_mappings["columns"]:
        source_col = col_info["source"]
        ru_col = col_info["ru"]
        kz_col = col_info["kz"]

        logger.info(f"\n=== [{sheet_name}] Processing column: {source_col} ===")

        # Extract cells
        cells = extract_cells_from_column(df, source_col)
        non_empty = sum(1 for c in cells if c.text)
        chunked_count = sum(1 for c in cells if c.chunks)
        total_chars = sum(len(c.text) for c in cells)

        logger.info(f"Extracted {len(cells)} cells ({non_empty} non-empty, {chunked_count} chunked)")
        logger.info(f"Total characters: {total_chars:,}")

        # Translate to Russian
        translate_cells(cells, "ru", "russian", manager, column_name=source_col)

        # Translate to Kazakh
        translate_cells(cells, "kk", "kazakh", manager, column_name=source_col)

        # Store results
        result_data[ru_col] = [c.translations.get("russian", "") for c in cells]
        result_data[kz_col] = [c.translations.get("kazakh", "") for c in cells]

    return pd.DataFrame(result_data)


def save_multi_sheet_excel(sheets_data: Dict[str, pd.DataFrame], output_path: str):
    """Save multiple sheets to an Excel file"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in sheets_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    logger.info(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Excel/CSV Translation Pipeline (Yandex Translate API)'
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

    # Initialize Yandex client
    yandex_config = config.get('yandex', {})
    api_key = yandex_config.get('api_key')
    folder_id = yandex_config.get('folder_id')

    if not api_key or not folder_id:
        logger.error("Yandex API credentials not found in config!")
        logger.error("Please add 'yandex.api_key' and 'yandex.folder_id' to config.yaml")
        return

    logger.info("Initializing Yandex Translate API client...")
    client = YandexTranslateClient(api_key, folder_id)
    manager = TranslationManager(client)
    logger.info("Yandex client ready")

    # Read input file
    logger.info(f"Reading input file: {args.input_file}")
    sheets = read_excel_or_csv(args.input_file)
    logger.info(f"Found {len(sheets)} sheet(s): {list(sheets.keys())}")

    # Process each sheet
    output_sheets = {}
    use_custom_columns = args.columns is not None

    for sheet_name, df in sheets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing sheet: {sheet_name} ({len(df)} rows)")
        logger.info(f"{'='*60}")

        # Determine columns to translate
        if use_custom_columns:
            valid_columns = validate_custom_columns(df, args.columns)
            if not valid_columns:
                logger.warning(f"[{sheet_name}] No matching columns found. Skipping.")
                continue

            if len(valid_columns) != len(args.columns):
                missing = set(args.columns) - set(valid_columns)
                logger.warning(f"[{sheet_name}] Columns not found: {missing}")

            column_mappings = get_output_column_names_for_custom(valid_columns)
            logger.info(f"Using custom columns: {valid_columns}")
        else:
            matched_tuple = find_matching_columns(df)

            if matched_tuple is None:
                logger.warning(f"[{sheet_name}] No matching column tuple found. Skipping.")
                logger.warning(f"  Expected: {COLUMN_TUPLES}")
                logger.warning(f"  Found: {list(df.columns)}")
                continue

            column_mappings = get_output_column_names_for_tuple(matched_tuple)
            logger.info(f"Found matching columns: {matched_tuple}")

        # Process the sheet
        result_df = process_sheet(df, sheet_name, column_mappings, manager)
        output_sheets[sheet_name] = result_df

    # Save output
    if not output_sheets:
        logger.error("No sheets were processed!")
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

    output_dir = config.get('output', {}).get('directory', './output')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_path = f"{output_dir}/{output_base}.xlsx"
    save_multi_sheet_excel(output_sheets, output_path)

    logger.info(f"\nPipeline completed successfully!")
    logger.info(f"Processed {len(output_sheets)} sheet(s)")


if __name__ == "__main__":
    main()
