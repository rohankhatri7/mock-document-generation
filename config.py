import zipcodes
from faker import Faker
from random import random 
import os, pathlib

FILE = "fakedata.xlsx"   
NUMROWS = 100                
SHEETNAME = "Sheet1"

# Initialize Faker
fake = Faker(['en_US', 'es'])
fake_en = fake['en_US']
fake_es = fake['es']
fake_address = Faker('en_US')  # Addresses follow US format

# Preload NY ZIP codes for Faker use
ny_zips = zipcodes.filter_by(state="NY")

# folder that holds NY address CSVs downloaded from OpenAddresses
NY_ADDR_DIR = pathlib.Path(__file__).parent / "ny"
# statewide CSV for variety
NY_ADDR_CSV = os.getenv("NY_ADDR_CSV", str(NY_ADDR_DIR / "statewide.csv"))

# csv output file
CSV_FILE = os.getenv("CSV_FILE", FILE.replace(".xlsx", ".csv"))

# directory layout for templates & fonts
BASE_DIR = pathlib.Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "pngtemplates"
FONTS_DIR = BASE_DIR / "fonts"

SSN_TEMPLATE_PATH = os.getenv("SSN_TEMPLATE_PATH", str(TEMPLATE_DIR / "ssn.png"))

# handwriting font
HANDWRITING_FONT = os.getenv("HANDWRITING_FONT", str(FONTS_DIR / "handwriting.ttf"))

# OpenSans font (used for SSN number and full name)
OPENSANS_FONT = os.getenv("OPENSANS_FONT", str(FONTS_DIR / "OpenSans_SemiCondensed-Regular.ttf"))

# signature font
SIGNATURE_FONT = os.getenv("SIGNATURE_FONT", str(FONTS_DIR / "signature.ttf"))

# blank page template
BLANK_PAGE_PATH = os.getenv("BLANK_PAGE_PATH", str(TEMPLATE_DIR / "blank.png"))

# page size (standard US-Letter)
PAGE_WIDTH  = int(os.getenv("PAGE_WIDTH", "2550"))
PAGE_HEIGHT = int(os.getenv("PAGE_HEIGHT", "3300"))
