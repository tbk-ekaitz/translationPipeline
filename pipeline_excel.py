#!/usr/bin/env python3
"""
Excel/CSV Translation Pipeline using Yandex Translate API
Translates specific columns from xlsx/csv files to Russian and Kazakh
Supports multi-sheet xlsx files, custom column selection, and partial progress resume
"""
import argparse
import hashlib
import json
import logging
import time
import yaml
import requests
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field, asdict

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

# Yandex API limits (using 90% of official to avoid constant 429s)
# Official: 2000 chars/string, 10K chars/request, 20 req/s
MAX_CHARS_PER_STRING = 2000       # Max chars per individual text
MAX_CHARS_PER_REQUEST = 9000      # 90% of 10K
MAX_REQUESTS_PER_SECOND = 18      # 90% of 20
REQUEST_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND  # ~55ms between requests

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


class PartialProgress:
    """Manages partial progress saving and loading"""

    def __init__(self, output_dir: str, input_file: str):
        self.output_dir = Path(output_dir)
        self.input_hash = self._compute_file_hash(input_file)
        self.input_name = Path(input_file).stem
        self.partial_file = self.output_dir / f".{self.input_name}.partial.json"

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute MD5 hash of input file for identification"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def exists(self) -> bool:
        """Check if partial progress file exists and matches input"""
        if not self.partial_file.exists():
            return False

        try:
            with open(self.partial_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('input_hash') == self.input_hash
        except (json.JSONDecodeError, KeyError):
            return False

    def load(self) -> Optional[Dict[str, Any]]:
        """Load partial progress if it exists and matches"""
        if not self.exists():
            return None

        try:
            with open(self.partial_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"\n[RESUME] Found partial progress file: {self.partial_file}")
            print(f"[RESUME] Resuming from previous session...")
            return data
        except Exception as e:
            logger.warning(f"Failed to load partial progress: {e}")
            return None

    def save(self, progress_data: Dict[str, Any]):
        """Save current progress"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        progress_data['input_hash'] = self.input_hash
        progress_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

        with open(self.partial_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)

        print(f"\n[PARTIAL SAVE] Progress saved to: {self.partial_file}")
        print(f"[PARTIAL SAVE] Run again to resume from this point")

    def delete(self):
        """Delete partial progress file after successful completion"""
        if self.partial_file.exists():
            self.partial_file.unlink()
            logger.info(f"Cleaned up partial progress file")


class YandexTranslateClient:
    """Client for Yandex Translate API with rate limiting and automatic retry"""

    API_URL = "https://translate.api.cloud.yandex.net/translate/v2/translate"

    def __init__(self, api_key: str, folder_id: str):
        self.api_key = api_key
        self.folder_id = folder_id
        self.last_request_time = 0
        self.total_429_hits = 0

    def _rate_limit(self):
        """Enforce rate limiting at 90% of official limit"""
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()

    def _wait_and_retry(self, response: requests.Response, attempt: int) -> float:
        """Calculate wait time when rate limited"""
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return BASE_RETRY_DELAY * (2 ** attempt)

    def translate(self, texts: List[str], target_lang: str, source_lang: str = "en") -> List[str]:
        """Translate a list of texts to target language with automatic retry"""
        if not texts:
            return []

        self._rate_limit()

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
                    self.total_429_hits += 1
                    wait_time = self._wait_and_retry(response, attempt)
                    print(f"\n[429 RATE LIMITED] Yandex says slow down! Waiting {wait_time:.1f}s... (attempt {attempt + 1}/{MAX_RETRIES}, total 429s: {self.total_429_hits})")
                    time.sleep(wait_time)
                    continue

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
    """Read xlsx or csv file into DataFrames (one per sheet)"""
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
    """Split text into chunks that fit within character limit"""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text
    separators = ['. ', ', ', '; ', ' ', '']

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        chunk = remaining[:max_chars]
        split_pos = max_chars

        for sep in separators:
            if sep:
                pos = chunk.rfind(sep)
                if pos > max_chars // 2:
                    split_pos = pos + len(sep)
                    break
            else:
                split_pos = max_chars

        chunks.append(remaining[:split_pos].strip())
        remaining = remaining[split_pos:].strip()

    return [c for c in chunks if c]


def extract_cells_from_column(df: pd.DataFrame, column_name: str) -> List[CellItem]:
    """Extract cells from a column and prepare them for translation"""
    cells = []

    for idx, value in enumerate(df[column_name]):
        text = str(value) if pd.notna(value) else ""
        text = text.strip()

        cell = CellItem(row_idx=idx, text=text)

        if text and len(text) > MAX_CHARS_PER_STRING:
            cell.chunks = split_text_by_chars(text)

        cells.append(cell)

    return cells


def create_batches(items: List[str], max_chars: int = MAX_CHARS_PER_REQUEST) -> List[List[str]]:
    """Group items into batches that fit within character limit per request"""
    batches = []
    current_batch = []
    current_chars = 0

    for item in items:
        item_chars = len(item)

        if item_chars > max_chars:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            batches.append([item])
            continue

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
    column_name: str = "",
    partial_progress: Optional[PartialProgress] = None,
    progress_data: Optional[Dict] = None
) -> bool:
    """
    Translate cells using Yandex API with smart batching

    Returns:
        True if completed successfully, False if failed (partial saved)
    """
    items_to_translate = []
    metadata = []

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
        for idx, _, _, _ in metadata:
            cells[idx].add_translation(lang_name, "")
        return True

    batches = create_batches(items_to_translate)

    desc = f"[{column_name}] -> {lang_name.capitalize()}" if column_name else f"-> {lang_name.capitalize()}"

    all_translations = []
    total_chars = sum(len(item) for item in items_to_translate)

    progress_bar = tqdm(
        batches,
        desc=desc,
        unit="batch",
        ncols=100
    )

    try:
        for batch_idx, batch in enumerate(progress_bar):
            batch_chars = sum(len(t) for t in batch)
            progress_bar.set_postfix({
                'items': len(batch),
                'chars': f'{batch_chars}'
            })

            translations = manager.translate_batch(batch, target_lang)
            all_translations.extend(translations)

    except Exception as e:
        logger.error(f"Translation failed: {e}")

        # Save partial progress
        if partial_progress and progress_data is not None:
            # Save what we have so far
            trans_idx = 0
            for cell_idx, is_chunked, chunk_count, is_empty in metadata:
                if trans_idx >= len(all_translations):
                    break
                if is_empty:
                    cells[cell_idx].add_translation(lang_name, "")
                elif is_chunked and trans_idx + chunk_count <= len(all_translations):
                    chunk_trans = all_translations[trans_idx:trans_idx + chunk_count]
                    merged = merge_chunk_translations(chunk_trans)
                    cells[cell_idx].add_translation(lang_name, merged)
                    trans_idx += chunk_count
                elif not is_chunked:
                    cells[cell_idx].add_translation(lang_name, all_translations[trans_idx])
                    trans_idx += 1

            print(f"\n[FAILED] Translation failed after {len(all_translations)} items")
            partial_progress.save(progress_data)

        return False

    # Assign all translations
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
    return True


def process_sheet(
    df: pd.DataFrame,
    sheet_name: str,
    column_mappings: dict,
    manager: TranslationManager,
    partial_progress: Optional[PartialProgress] = None,
    existing_results: Optional[Dict] = None
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Process a single sheet through the translation pipeline

    Returns:
        Tuple of (result DataFrame or None if failed, progress data dict)
    """
    result_data = existing_results.copy() if existing_results else {}
    progress_data = {'sheets': {}, 'completed_columns': {}}

    for col_info in column_mappings["columns"]:
        source_col = col_info["source"]
        ru_col = col_info["ru"]
        kz_col = col_info["kz"]

        # Check if already completed in partial
        if ru_col in result_data and kz_col in result_data:
            logger.info(f"[{sheet_name}] Column {source_col} already translated, skipping...")
            continue

        logger.info(f"\n=== [{sheet_name}] Processing column: {source_col} ===")

        cells = extract_cells_from_column(df, source_col)
        non_empty = sum(1 for c in cells if c.text)
        chunked_count = sum(1 for c in cells if c.chunks)
        total_chars = sum(len(c.text) for c in cells)

        logger.info(f"Extracted {len(cells)} cells ({non_empty} non-empty, {chunked_count} chunked)")
        logger.info(f"Total characters: {total_chars:,}")

        # Update progress data
        progress_data['sheets'][sheet_name] = result_data

        # Translate to Russian (if not already done)
        if ru_col not in result_data:
            success = translate_cells(
                cells, "ru", "russian", manager,
                column_name=source_col,
                partial_progress=partial_progress,
                progress_data=progress_data
            )
            if not success:
                result_data[ru_col] = [c.translations.get("russian", "") for c in cells]
                progress_data['sheets'][sheet_name] = result_data
                return None, progress_data

            result_data[ru_col] = [c.translations.get("russian", "") for c in cells]

        # Translate to Kazakh (if not already done)
        if kz_col not in result_data:
            success = translate_cells(
                cells, "kk", "kazakh", manager,
                column_name=source_col,
                partial_progress=partial_progress,
                progress_data=progress_data
            )
            if not success:
                result_data[kz_col] = [c.translations.get("kazakh", "") for c in cells]
                progress_data['sheets'][sheet_name] = result_data
                return None, progress_data

            result_data[kz_col] = [c.translations.get("kazakh", "") for c in cells]

    return pd.DataFrame(result_data), progress_data


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
    parser.add_argument(
        '--renew',
        action='store_true',
        help='Ignore partial progress and start fresh'
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    output_dir = config.get('output', {}).get('directory', './output')

    # Initialize partial progress tracker
    partial_progress = PartialProgress(output_dir, args.input_file)

    # Check for existing partial progress
    existing_progress = None
    if not args.renew:
        existing_progress = partial_progress.load()
        if existing_progress:
            logger.info("Continuing from partial progress...")
    else:
        if partial_progress.exists():
            print("[RENEW] Ignoring existing partial progress, starting fresh...")
            partial_progress.delete()

    # Initialize Yandex client
    yandex_config = config.get('yandex', {})
    api_key = yandex_config.get('api_key')
    folder_id = yandex_config.get('folder_id')

    if not api_key or not folder_id:
        logger.error("Yandex API credentials not found in config!")
        logger.error("Please add 'yandex.api_key' and 'yandex.folder_id' to config.yaml")
        return

    logger.info("Initializing Yandex Translate API client...")
    logger.info(f"Rate limiting at 90%: {MAX_REQUESTS_PER_SECOND} req/s, {MAX_CHARS_PER_REQUEST} chars/request")
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
    all_success = True

    for sheet_name, df in sheets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing sheet: {sheet_name} ({len(df)} rows)")
        logger.info(f"{'='*60}")

        # Get existing results for this sheet if resuming
        existing_results = None
        if existing_progress and 'sheets' in existing_progress:
            existing_results = existing_progress['sheets'].get(sheet_name)
            if existing_results:
                logger.info(f"Found {len(existing_results)} existing columns from partial progress")

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
        result_df, progress_data = process_sheet(
            df, sheet_name, column_mappings, manager,
            partial_progress=partial_progress,
            existing_results=existing_results
        )

        if result_df is None:
            all_success = False
            break

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

    if all_success:
        logger.info("\nGenerating output file...")

        if args.output:
            output_base = args.output
        else:
            input_path = Path(args.input_file)
            output_base = input_path.stem + "_translated"

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        output_path = f"{output_dir}/{output_base}.xlsx"
        save_multi_sheet_excel(output_sheets, output_path)

        # Clean up partial progress on success
        partial_progress.delete()

        print(f"\n[SUCCESS] Pipeline completed!")
        print(f"[SUCCESS] Output: {output_path}")
        print(f"[SUCCESS] Processed {len(output_sheets)} sheet(s)")
        if client.total_429_hits > 0:
            print(f"[INFO] Total 429 rate limits hit: {client.total_429_hits}")
    else:
        print(f"\n[INCOMPLETE] Pipeline stopped due to errors")
        print(f"[INCOMPLETE] Run again to resume from partial progress")


if __name__ == "__main__":
    main()
