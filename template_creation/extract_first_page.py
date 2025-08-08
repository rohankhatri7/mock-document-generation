import argparse
from pdf2image import convert_from_path
from pathlib import Path

# Configuration
DPI = 300
POPPLER_PATH = r"C:\Users\573641\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"  # Add your Poppler bin path here

def extract_first_page(pdf_path):
    # extract first page from PDF for multi page documents as PNG
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"File must be a PDF: {pdf_path}")
    
    # Parse filename: prefix_suffix.pdf
    stem = pdf_path.stem
    if '_' not in stem:
        raise ValueError(f"PDF filename must follow 'prefix_suffix.pdf' convention: {pdf_path.name}")
    
    # Split into prefix and suffix
    parts = stem.split('_')
    if len(parts) < 2:
        raise ValueError(f"PDF filename must have at least one underscore (prefix_suffix): {pdf_path.name}")
    
    prefix = parts[0]
    suffix = '_'.join(parts[1:])
    
    # Output directory is same as input PDF
    output_dir = pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename: prefix_page1_suffix.png
    output_stem = f"{prefix}_page1_{suffix}"
    png_path = output_dir / f"{output_stem}.png"
    
    print(f"Extracting first page from: {pdf_path}")
    print(f"Output will be: {png_path}")
    
    # Extract first page as PNG
    convert_from_path(
        pdf_path,
        first_page=1,
        last_page=1,
        dpi=DPI,
        fmt='png',
        output_folder=str(output_dir),
        output_file=output_stem,
        poppler_path=POPPLER_PATH
    )
    
    print(f"Extracted page‑1 → {png_path}")
    return png_path

def main():
    parser = argparse.ArgumentParser(description='Extract first page from PDF following naming convention')
    parser.add_argument('pdf_path', help='Path to the PDF file (must follow prefix_suffix.pdf naming)')
    
    args = parser.parse_args()
    
    try:
        extract_first_page(args.pdf_path)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())