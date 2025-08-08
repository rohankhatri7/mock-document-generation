import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from random import choice, random
import re
from gender_guesser.detector import Detector

from config import (
    FILE, NUMROWS, SHEETNAME,
    fake, fake_address, ny_zips,
    CSV_FILE
)

import zipcodes

def generate_complex_name(kind: str) -> str:
    if kind == "first":
        return fake.first_name()
    elif kind == "last":
        return fake.last_name()
    else:
        return fake.name()

def split_address(address: str):
    # Try to split on common secondary unit indicators
    match = re.match(r"(.+?)(?:,?\s*(Apt|Suite|Unit|#)\s*\w+)?$", address)
    if match:
        street1 = match.group(1).strip()
        street2 = address[len(street1):].strip().lstrip(',').strip()
        return street1, street2
    return address, ""

_detector = Detector(case_sensitive=False)

# helper function
def _gender_code(first_name: str) -> str:
    #Return "M" or "F" using gender_guesser
    g = _detector.get_gender(first_name.split()[0])
    if g in ("male", "mostly_male"):
        return "M"
    elif g in ("female", "mostly_female"):
        return "F"
    else:
        return "M" if random() < 0.5 else "F"

# Generate rows in spreadsheet
def generate_rows(n: int = NUMROWS) -> pd.DataFrame:
    rows = []

    for _ in range(n):
        # Generate fake data only
        rec = choice(ny_zips)
        full_address = fake_address.street_address()
        street1, street2_candidate = split_address(full_address)
        street2 = street2_candidate or (fake_address.secondary_address() if random() < 0.3 else "")
        
        # Generate person data
        first_name = generate_complex_name("first")
        last_name = generate_complex_name("last")
        middle_initial = fake.random_uppercase_letter()
        gender = _gender_code(first_name)
        
        rows.append({
            "Formtype": "",
            "AccountID": fake.bothify("AC##########"),
            "HealthBenefitID": fake.bothify("HX###########"),
            "DOB": fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%m/%d/%Y"),
            "FirstName": first_name,
            "MiddleInitial": middle_initial,
            "LastName": last_name,
            "FullName": f"{first_name} {middle_initial}. {last_name}".strip(),
            "Gender": gender,
            "SSN": fake.ssn(),
            "County": rec["county"].replace("County", "").replace("county", "").strip(),
            "Street1": street1,
            "Street2": street2,
            "Zip": rec["zip_code"],
            "City": rec["city"],
            "State": "NY",
            "Filename": "",
        })

    return pd.DataFrame(rows)

def main():
    try:
        if not os.path.exists(FILE):
            print(f"Creating new file: {FILE}")
        else:
            print(f"Replacing data in existing file: {FILE}")
        
        new_rows = generate_rows()
        # Write Excel
        new_rows.to_excel(FILE,
                          index=False,
                          sheet_name=SHEETNAME,
                          engine="openpyxl")

        # also output to csv format
        new_rows.to_csv(CSV_FILE, index=False)

        print(f"Success: generated {NUMROWS} rows to '{FILE}' and '{CSV_FILE}'.")

    except PermissionError:
        print(f"ERROR: File '{FILE}' is open. Close it and re-run.")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()