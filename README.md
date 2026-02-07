# Cosmisum - Comic/Manga/Manhwa/Manhua Summarizer and Analyzer

A Python pipeline for analyzing manga, comics, and documents. Extracts panels from PDFs, performs OCR, and uses LLM to generate summaries, tags, and genre classification.

## Features

- 📄 PDF page extraction and panel detection
- 🔍 OCR text extraction from panels
- 🧠 Smart token-limited chunking with uniform distribution
- 🤖 LLM-powered analysis (OpenAI API compatible)
- 💻 Cross-platform support (Windows, macOS, Linux)

## Prerequisites

### 1. Python 3.10+

Make sure you have Python 3.10 or higher installed.

### 2. Poppler

Poppler is required for PDF processing. Install it based on your OS:

#### Windows

**Option A: Using Chocolatey (Recommended)**
```bash
choco install poppler
```

**Option B: Manual Installation**
1. Download poppler for Windows from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract the zip file (e.g., to `C:\poppler`)
3. Add the `bin` folder to your PATH:
   - Right-click "This PC" → Properties → Advanced system settings
   - Click "Environment Variables"
   - Under "System variables", find and edit "Path"
   - Add new entry: `C:\poppler\Library\bin` (or wherever you extracted it)
   - Click OK and restart your terminal

#### macOS

```bash
brew install poppler
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

#### Linux (Fedora/RHEL)

```bash
sudo dnf install poppler-utils
```

### 3. Tesseract OCR

Tesseract is required for text extraction.

#### Windows

**Option A: Using Chocolatey**
```bash
choco install tesseract
```

**Option B: Manual Installation**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer
3. Add Tesseract to PATH (usually `C:\Program Files\Tesseract-OCR`)
4. Restart your terminal

#### macOS

```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get install tesseract-ocr
```

#### Linux (Fedora/RHEL)

```bash
sudo dnf install tesseract
```

## Installation

### 1. Clone or download this repository

```bash
git clone https://github.com/sammwyy/cosmisum
cd cosmisum
```

### 2. Create a virtual environment (recommended but optional)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### 1. Set up environment variables

On first run, the script will create a default `.env` file. Edit it with your configuration:

```bash
# .env
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_API_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_ID=gpt-4
```

**Note:** You can use any OpenAI-compatible API by changing the `OPENAI_API_BASE_URL` (e.g., for local models, Azure OpenAI, etc.)

### 2. Get an API key

- For OpenAI: https://platform.openai.com/api-keys
- For other providers: Check their documentation

## Usage

### Basic usage

```bash
python cosmisum.py input.pdf
```

### Example

```bash
python cosmisum.py my_manga_chapter.pdf
```

### What it does

1. **Extracts panels** from each PDF page
2. **Performs OCR** on each panel to extract text
3. **Creates uniform chunks** distributed across the document (respecting token limits)
4. **Sends to LLM** for analysis
5. **Outputs**:
   - Plot summary
   - Thematic tags
   - Genre/category classification

### Output

The script will display results in the console:

```
================================================================================
Based on the provided text extracts, here is the analysis of the manga/comic:

1. **Plot Summary**: [Generated summary...]

2. **Thematic Tags**: action, adventure, friendship, ...

3. **Genre/Category**: Shonen manga / Action-Adventure

================================================================================
```

## Troubleshooting

### "Unable to get page count. Is poppler installed and in PATH?"

- Make sure poppler is installed (see Prerequisites)
- Verify it's in your PATH: run `pdfinfo -v` in terminal
- On Windows, restart your terminal after adding to PATH

### "pytesseract.pytesseract.TesseractNotFoundError"

- Make sure Tesseract is installed (see Prerequisites)
- On Windows, you may need to specify the path in the script:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```

### "OPENAI_API_KEY not configured"

- Edit the `.env` file and add your actual API key
- Make sure the `.env` file is in the same directory as `cosmisum.py`

### No text extracted

- Check if your PDF contains actual text (not just images)
- Try adjusting the OCR language settings in the code
- Increase PDF DPI in `extract_all_panels()` function

## Advanced Usage

### Using with local LLM

Edit `.env` to point to your local API:

```bash
OPENAI_API_KEY=not-needed
OPENAI_API_BASE_URL=http://localhost:1234/v1
OPENAI_MODEL_ID=local-model
```

### Multi-language OCR

Modify the `perform_ocr()` function to use different languages:

```python
text = pytesseract.image_to_string(img, lang='jpn+eng')  # Japanese + English
```

Download language data from: https://github.com/tesseract-ocr/tessdata

## License

MIT

## Contributing

Pull requests are welcome. For major changes, please open an issue first.