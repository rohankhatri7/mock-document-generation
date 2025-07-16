# extract_first_page.py
#
# One‑shot helper script
# ----------------------
# • Reads the original 4‑page W‑4 PDF
#     template_tools/templates/taxreturn/fw4_taxreturn.pdf
# • Extracts page‑1 at 300 dpi
# • Saves it to the same folder, using the generator’s naming convention:
#     fw4_page1_taxreturn.png
#
# After manual cleanup/deskewing you will rename that file to
#     fw4_page1_taxreturn_clean.png
# and create a matching
#     fw4_page1_taxreturn_spec.json
# in the *same* directory.
#
# Dependencies
# ------------
#   pip install pdf2image pillow
#   # plus Poppler (brew install poppler  OR  apt-get install poppler-utils)

from pdf2image import convert_from_path
from pathlib import Path

# ------------------------------------------------------------------
# Paths – adjust only if your repo layout changes
# ------------------------------------------------------------------
PDF_PATH   = Path('template_tools/templates/taxreturn/fw4_taxreturn.pdf')
OUTPUT_DIR = Path('template_tools/templates/taxreturn')
DPI        = 300
# ------------------------------------------------------------------

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Derive cleaned‑page filename pattern
# fw4_taxreturn.pdf  →  prefix='fw4' , suffix='taxreturn'
prefix, suffix = PDF_PATH.stem.split('_', 1)          # guarantees two parts
output_stem    = f"{prefix}_page1_{suffix}"           # fw4_page1_taxreturn
png_path       = OUTPUT_DIR / f"{output_stem}.png"

# Extract first page as PNG
convert_from_path(
    PDF_PATH,
    first_page=1,
    last_page=1,
    dpi=DPI,
    fmt='png',
    output_folder=str(OUTPUT_DIR),
    output_file=output_stem
)

print(f"Extracted page‑1 → {png_path}")
