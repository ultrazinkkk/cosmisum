#!/usr/bin/env python3
"""
cosmisum.py - Manga/Comic/Document Analysis Pipeline and Summarizer
Extracts panels, performs OCR, creates token-limited chunks with uniform distribution,
and sends to LLM for analysis.
"""

import os
import sys
import re
import shutil
import tempfile
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI

# ANSI Color codes for prettier logs
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_info(msg):
    print(f"{Colors.OKBLUE}(?){Colors.ENDC} {msg}")

def log_success(msg):
    print(f"{Colors.OKGREEN}(OK){Colors.ENDC} {msg}")

def log_warning(msg):
    print(f"{Colors.WARNING}(!){Colors.ENDC} {msg}")

def log_error(msg):
    print(f"{Colors.FAIL}[X] {Colors.BOLD}ERROR: {msg}{Colors.ENDC}")

def log_step(msg):
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}(>) {msg}{Colors.ENDC}")


# Default values for CLI flags
DEFAULT_MAX_TOKENS_PER_CHUNK = 2000
DEFAULT_MAX_TOKENS_TOTAL = 200000

@dataclass
class PanelText:
    """Represents OCR text from a single panel"""
    page: int
    panel: int
    text: str
    token_count: int

def setup_environment():
    """Load environment variables"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
    model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4")
    
    if not api_key or api_key == "your_api_key_here":
        log_error("OPENAI_API_KEY not configured in .env file or environment variables.")
        log_info("Please edit .env and add your actual API key.\n")
        sys.exit(1)
    
    return api_key, base_url, model_id


def create_temp_directory() -> Path:
    """Create random temporary directory compatible with any OS"""
    # Create a random subdirectory in system temp
    temp_base = tempfile.gettempdir()
    temp_dir = Path(temp_base) / f"panels_{os.getpid()}_{id(object())}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def prepare_temp_directory(temp_dir: Path):
    """Clean temporary directory if it exists"""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)


def extract_panels_from_page(page_image: np.ndarray, page_num: int) -> List[Tuple[int, np.ndarray]]:
    """
    Extract individual panels from a page using contour detection.
    Returns list of (panel_index, panel_image) tuples.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours by area (largest panels first)
    valid_contours = []
    min_area = page_image.shape[0] * page_image.shape[1] * 0.01  # at least 1% of page
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            valid_contours.append((y, x, w, h))  # Store y-first for top-to-bottom sorting
    
    # Sort by vertical position (top to bottom), then horizontal (left to right)
    valid_contours.sort(key=lambda c: (c[0], c[1]))
    
    # Extract panel images
    panels = []
    for idx, (y, x, w, h) in enumerate(valid_contours, start=1):
        panel_img = page_image[y:y+h, x:x+w]
        panels.append((idx, panel_img))
    
    return panels


def save_panel(panel_image: np.ndarray, page_num: int, panel_num: int, temp_dir: Path) -> Path:
    """Save panel image to temp directory"""
    filename = f"page_{page_num:02d}_panel_{panel_num:03d}.png"
    filepath = temp_dir / filename
    
    # Convert BGR to RGB if needed
    if len(panel_image.shape) == 3:
        panel_image = cv2.cvtColor(panel_image, cv2.COLOR_BGR2RGB)
    
    Image.fromarray(panel_image).save(filepath)
    return filepath


def extract_all_panels(pdf_path: str, temp_dir: Path) -> List[Path]:
    """
    Convert PDF to images and extract all panels.
    Returns list of saved panel paths.
    """
    log_info(f"Converting PDF: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=200)
    
    panel_paths = []
    
    for page_num, page_pil in enumerate(pages, start=1):
        log_info(f"Processing page {page_num}/{len(pages)}")
        
        # Convert PIL to numpy array
        page_array = np.array(page_pil)
        
        # Extract panels from this page
        panels = extract_panels_from_page(page_array, page_num)
        
        # Save each panel
        for panel_num, panel_img in panels:
            panel_path = save_panel(panel_img, page_num, panel_num, temp_dir)
            panel_paths.append(panel_path)
    
    log_success(f"Extracted {len(panel_paths)} panels total")
    return panel_paths


def clean_ocr_text(text: str) -> str:
    """Clean OCR artifacts and noise"""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove isolated single characters (likely noise)
    text = re.sub(r'\b\w\b', '', text)
    
    # Remove common OCR artifacts
    text = re.sub(r'[|_~`]', '', text)
    
    # Strip and remove if too short
    text = text.strip()
    
    return text if len(text) > 3 else ""


def perform_ocr(panel_path: Path) -> str:
    """Extract text from panel image using OCR"""
    try:
        img = Image.open(panel_path)
        text = pytesseract.image_to_string(img, lang='eng+spa')
        return clean_ocr_text(text)
    except Exception as e:
        log_warning(f"OCR error on {panel_path.name}: {e}")
        return ""


def extract_page_panel_from_filename(filename: str) -> Tuple[int, int]:
    """Parse page and panel numbers from filename"""
    match = re.match(r'page_(\d+)_panel_(\d+)\.png', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


def ocr_all_panels(panel_paths: List[Path], encoding) -> List[PanelText]:
    """
    Apply OCR to all panels and count tokens.
    Returns list of PanelText objects.
    """
    panel_texts = []
    
    for panel_path in panel_paths:
        page, panel = extract_page_panel_from_filename(panel_path.name)
        text = perform_ocr(panel_path)
        
        if text:
            token_count = len(encoding.encode(text))
            panel_texts.append(PanelText(page, panel, text, token_count))
    
    log_success(f"Extracted text from {len(panel_texts)} panels")
    return panel_texts


def select_uniform_chunks(panel_texts: List[PanelText], encoding, max_total_tokens: int, max_tokens_per_chunk: int) -> List[str]:
    """
    Select chunks uniformly distributed across all panels.
    Ensures total tokens <= max_total_tokens.
    Uses deterministic spacing algorithm.
    """
    if not panel_texts:
        return []
    
    total_panels = len(panel_texts)
    
    # Calculate how many chunks we can fit
    max_chunks = max_total_tokens // max_tokens_per_chunk
    
    # Determine actual number of chunks (can't exceed available panels)
    num_chunks = min(max_chunks, total_panels)
    
    if num_chunks == 0:
        return []
    
    # Calculate step size for uniform distribution
    step = total_panels / num_chunks
    
    selected_chunks = []
    
    # Select panels at uniform intervals
    for i in range(num_chunks):
        # Determine which panel to start this chunk from
        panel_idx = int(i * step)
        panel_idx = min(panel_idx, total_panels - 1)  # Safety bound
        
        # Build chunk starting from this panel
        chunk_text_parts = []
        chunk_tokens = 0
        
        # Try to fill chunk with consecutive panels
        for j in range(panel_idx, total_panels):
            panel = panel_texts[j]
            potential_tokens = chunk_tokens + panel.token_count
            
            if potential_tokens <= max_tokens_per_chunk:
                chunk_text_parts.append(
                    f"[Page {panel.page}, Panel {panel.panel}] {panel.text}"
                )
                chunk_tokens = potential_tokens
            else:
                break
        
        if chunk_text_parts:
            chunk_text = "\n".join(chunk_text_parts)
            selected_chunks.append(chunk_text)
    
    # Verify total token budget
    total_tokens = sum(len(encoding.encode(chunk)) for chunk in selected_chunks)
    log_info(f"Selected {len(selected_chunks)} chunks, {total_tokens} tokens total")
    
    return selected_chunks


def build_llm_prompt(chunks: List[str], output_format: str = "markdown") -> str:
    """Build prompt for LLM analysis"""
    chunks_text = "\n\n---\n\n".join(
        f"CHUNK {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)
    )
    
    format_instruction = ""
    if output_format == "json":
        format_instruction = """Format your response as a valid JSON object with the following keys:
- "plot_summary": (string)
- "thematic_tags": (list of strings)
- "genre": (string)
Only return the JSON object, do not include any other text or markdown block markers."""
    else:
        format_instruction = """Format your response clearly with these three sections:
1. **Plot Summary**
2. **Thematic Tags**
3. **Genre/Category**"""

    prompt = f"""You are analyzing a manga/comic based on text extracted from panels.

Below are text excerpts from uniformly distributed panels across the work:

{chunks_text}

Based on this content, please provide an analysis.

{format_instruction}
"""
    
    return prompt


def query_llm(prompt: str, api_key: str, base_url: str, model_id: str, max_total_tokens: int) -> str:
    """Send prompt to LLM and return response"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    log_info(f"Querying LLM: {model_id}")
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=int(max_total_tokens / 2)
    )
    
    return response.choices[0].message.content


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="Manga/Comic/Document Analysis Pipeline and Summarizer")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument("--max-total", type=int, default=DEFAULT_MAX_TOKENS_TOTAL, help=f"Maximum total tokens allowed (default: {DEFAULT_MAX_TOKENS_TOTAL})")
    parser.add_argument("--max-per-chunk", type=int, default=DEFAULT_MAX_TOKENS_PER_CHUNK, help=f"Maximum tokens per chunk (default: {DEFAULT_MAX_TOKENS_PER_CHUNK})")
    parser.add_argument("--out", choices=["markdown", "json"], default="markdown", help="Output format (default: markdown)")
    parser.add_argument("--nofile", action="store_true", help="Do not save the result to a file")
    
    args = parser.parse_args()
    pdf_path = args.pdf_path
    
    if not os.path.exists(pdf_path):
        log_error(f"File not found: {pdf_path}")
        sys.exit(1)
    
    # Setup
    api_key, base_url, model_id = setup_environment()
    
    # Create random temp directory (OS-independent)
    temp_dir = create_temp_directory()
    log_info(f"Using temporary directory: {temp_dir}")
    
    prepare_temp_directory(temp_dir)
    
    # Initialize tokenizer
    encoding = tiktoken.encoding_for_model("gpt-4")
    
    try:
        # Step 1: Extract panels
        log_step("STEP 1: Extracting panels from PDF")
        panel_paths = extract_all_panels(pdf_path, temp_dir)
        
        # Step 2: OCR
        log_step("STEP 2: Performing OCR on panels")
        panel_texts = ocr_all_panels(panel_paths, encoding)
        
        # Step 3: Select uniform chunks
        log_step("STEP 3: Selecting representative chunks")
        chunks = select_uniform_chunks(panel_texts, encoding, args.max_total, args.max_per_chunk)
        
        if not chunks:
            log_error("No text extracted or chunks created")
            sys.exit(1)
        
        # Step 4: Query LLM
        log_step("STEP 4: Querying LLM for analysis")
        prompt = build_llm_prompt(chunks, args.out)
        response = query_llm(prompt, api_key, base_url, model_id, args.max_total)
        
        # Step 5: Output and persistence
        log_step("FINAL RESULT")
        line_sep = "=" * 60
        print(f"\n{line_sep}")
        print(response)
        print(f"{line_sep}\n")
        
        if not args.nofile:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            ext = "json" if args.out == "json" else "md"
            output_file = f"result-{timestamp}.{ext}"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(response)
            
            log_success(f"Result saved to: {output_file}")
            
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup log
        log_info(f"Temporary panels saved in: {temp_dir} (for inspection)")


if __name__ == "__main__":
    main()