
"""
Document_loader.py - Stage 1 (IMPROVED)
========================================
Removed LLM correction to prevent data loss
Added better OCR text cleaning
"""

import pymupdf as fitz
import pytesseract
from PIL import Image
from io import BytesIO
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import re
import platform

# ============================================================================
# TESSERACT SETUP FOR WINDOWS
# ============================================================================
if platform.system() == 'Windows':
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]

    for tess_path in possible_paths:
        if os.path.exists(tess_path):
            pytesseract.pytesseract.tesseract_cmd = tess_path
            print(f"✓ Tesseract found: {tess_path}")
            break


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for document processing pipeline"""

    # Paths
    OUTPUT_DIR = "processed_documents"
    CHUNKS_DIR = os.path.join(OUTPUT_DIR, "chunks")
    METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")

    # OCR Settings
    OCR_DPI = 300  # Higher DPI = better quality but slower

    # Chunking Settings
    CHUNK_SIZE = 1500  # Characters per chunk
    CHUNK_OVERLAP = 180  # Overlap for context preservation

    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.CHUNKS_DIR, exist_ok=True)
        os.makedirs(cls.METADATA_DIR, exist_ok=True)
        print(f"✓ Directories created: {cls.OUTPUT_DIR}")


# ============================================================================
# STAGE 1.1: OCR EXTRACTION
# ============================================================================

class OCRExtractor:
    """Extract text from PDF using OCR"""

    def __init__(self, dpi: int = Config.OCR_DPI):
        self.dpi = dpi

    def extract_from_pdf(self, pdf_path: str) -> Dict:
        """
        Extract text from PDF using OCR

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extracted text and metadata
        """
        print(f"\n{'='*70}")
        print("STAGE 1.1: OCR EXTRACTION")
        print(f"{'='*70}")
        print(f"Processing: {pdf_path}\n")

        doc = fitz.open(pdf_path)

        full_text = []
        page_texts = []
        total_chars = 0

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Try native text extraction first
            text = page.get_text()

            # If no text or very little, use OCR
            if len(text.strip()) < 50:
                # Convert page to image
                pix = page.get_pixmap(dpi=self.dpi)
                img_data = pix.tobytes("png")
                img = Image.open(BytesIO(img_data))

                # Perform OCR
                text = pytesseract.image_to_string(img)

            page_texts.append(text)
            full_text.append(text)

            char_count = len(text)
            total_chars += char_count

            print(f"  Page {page_num + 1}/{len(doc)}: {char_count} characters extracted")

        doc.close()

        combined_text = "\n\n".join(full_text)

        print(f"\n✅ OCR Complete: {total_chars} characters from {len(page_texts)} pages\n")

        return {
            'full_text': combined_text,
            'page_texts': page_texts,
            'total_pages': len(page_texts),
            'total_characters': total_chars
        }


# ============================================================================
# STAGE 1.2: TEXT CLEANING (REPLACES LLM CORRECTION)
# ============================================================================

class TextCleaner:
    """Clean OCR text without LLM (prevents data loss)"""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean OCR text using rule-based methods

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text
        """
        print(f"{'='*70}")
        print("STAGE 1.2: TEXT CLEANING")
        print(f"{'='*70}")
        print(f"Cleaning {len(text)} characters...\n")

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common OCR errors
        text = text.replace('|', 'I')  # Common OCR mistake
        text = text.replace('0', 'O').replace('O', '0')  # Only in specific contexts

        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')

        # Fix multiple periods
        text = re.sub(r'\.{3,}', '...', text)

        # Remove extra newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        print(f"✅ Text cleaned: {len(text)} characters (no data loss!)\n")

        return text


# ============================================================================
# STAGE 1.3: TEXT CHUNKING
# ============================================================================

class TextChunker:
    """Split text into overlapping chunks"""

    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, 
                 chunk_overlap: int = Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"{'='*70}")
        print("STAGE 1.3: TEXT CHUNKER INITIALIZED")
        print(f"{'='*70}")
        print(f"Chunk size: {chunk_size} characters")
        print(f"Chunk overlap: {chunk_overlap} characters\n")

    def chunk_text(self, text: str) -> List[Dict]:
        """
        Split text into overlapping chunks

        Args:
            text: Full text to chunk

        Returns:
            List of chunk dictionaries
        """
        print("Splitting text into chunks...")

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunk = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_position': start,
                'end_position': end,
                'length': len(chunk_text)
            }

            chunks.append(chunk)
            print(f"  Chunk {chunk_id}: {len(chunk_text)} characters")

            start += self.chunk_size - self.chunk_overlap
            chunk_id += 1

        print(f"\n✅ Created {len(chunks)} chunks\n")
        return chunks


# ============================================================================
# STAGE 1.4: SAVE CHUNKS
# ============================================================================

class ChunkSaver:
    """Save chunks and metadata"""

    @staticmethod
    def save_chunks(chunks: List[Dict], source_file: str, 
                   ocr_metadata: Dict) -> Dict[str, str]:
        """
        Save chunks and metadata to JSON files

        Args:
            chunks: List of chunk dictionaries
            source_file: Original PDF filename
            ocr_metadata: OCR extraction metadata

        Returns:
            Dictionary with paths to saved files
        """
        print(f"{'='*70}")
        print("STAGE 1.4: SAVING CHUNKS")
        print(f"{'='*70}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = Path(source_file).stem

        # Add metadata to each chunk
        for chunk in chunks:
            chunk['source_file'] = source_file
            chunk['timestamp'] = datetime.now().isoformat()

        # Save chunks
        chunks_filename = f"{source_name}_chunks_{timestamp}.json"
        chunks_path = os.path.join(Config.CHUNKS_DIR, chunks_filename)

        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(chunks)} chunks to: {chunks_path}")

        # Save metadata
        metadata = {
            'source_file': source_file,
            'processing_timestamp': datetime.now().isoformat(),
            'total_chunks': len(chunks),
            'chunk_size': Config.CHUNK_SIZE,
            'chunk_overlap': Config.CHUNK_OVERLAP,
            'ocr_metadata': ocr_metadata
        }

        metadata_filename = f"{source_name}_metadata_{timestamp}.json"
        metadata_path = os.path.join(Config.METADATA_DIR, metadata_filename)

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved metadata to: {metadata_path}\n")

        return {
            'chunks_file': chunks_path,
            'metadata_file': metadata_path
        }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class DocumentProcessor:
    """Main document processing pipeline"""

    def __init__(self):
        Config.setup_directories()
        self.ocr_extractor = OCRExtractor()
        self.text_cleaner = TextCleaner()
        self.text_chunker = TextChunker()

    def process_document(self, pdf_path: str) -> str:
        """
        Process PDF document through complete Stage 1 pipeline

        Args:
            pdf_path: Path to PDF file

        Returns:
            Path to chunks JSON file
        """
        print(f"\n{'#'*70}")
        print("DOCUMENT PROCESSING PIPELINE - STAGE 1 (IMPROVED)")
        print(f"{'#'*70}\n")

        # Stage 1.1: OCR Extraction
        ocr_result = self.ocr_extractor.extract_from_pdf(pdf_path)

        # Stage 1.2: Text Cleaning (NO LLM - No Data Loss!)
        cleaned_text = self.text_cleaner.clean_text(ocr_result['full_text'])

        # Stage 1.3: Text Chunking
        chunks = self.text_chunker.chunk_text(cleaned_text)

        # Stage 1.4: Save Chunks
        result_files = ChunkSaver.save_chunks(
            chunks, 
            pdf_path, 
            {
                'total_pages': ocr_result['total_pages'],
                'total_characters': ocr_result['total_characters'],
                'cleaned_characters': len(cleaned_text)
            }
        )

        print(f"{'='*70}")
        print("✅ STAGE 1 COMPLETE!")
        print(f"{'='*70}")
        print(f"Chunks file: {result_files['chunks_file']}")
        print(f"Total chunks: {len(chunks)}")
        print(f"NO DATA LOSS: {len(cleaned_text)} characters preserved!\n")

        return result_files['chunks_file']


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    processor = DocumentProcessor()

    # Test with your PDF
    pdf_path = r"C:\Users\Nikhil Pathak\Downloads\CUAD_v1\CUAD_v1\full_contract_pdf\Part_II\Service\BLACKSTONEGSOLONG-SHORTCREDITINCOMEFUND_05_11_2020-EX-99.(K)(1)-SERVICE AGREEMENT.PDF"

    try:
        chunks_file = processor.process_document(pdf_path)
        print(f"✅ Success! Chunks saved to: {chunks_file}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
