# Manual Setup Guide

## Prerequisites

- Python 3.7+
- Install required Python packages:
  ```powershell
  pip install -r requirements.txt
  pip install labelme
  ```
- You must have the LabelMe GUI tool installed to annotate templates. Run `labelme` from the command line to launch the GUI.

## Required Folders

- **fonts/**: Place all font files (e.g., .ttf) here that will be used for rendering text and signatures on documents.
- **ny/**: This folder contains a dataset used for generating fake data (e.g., names, addresses, etc.). You can replace this with your own dataset if desired.
- **forms/**: Contains all document form templates and their corresponding LabelMe annotation files. Each form type should have its own subfolder.

## File Naming

**Template files:** `{prefix}_{suffix}.png`
- **Prefix**: Variant identifier (e.g., `ny`, `ca`, `federal`)
- **Suffix**: Form type, must match directory name (e.g., `drivers_license`)
- **Extension**: Must be `.png`

**Examples:**
```
 ny_drivers_license.png
 ca_drivers_license.png
 ny_license.png (missing form type)
 drivers_license_ny.png (reversed)
```

## Field Names

**Critical:** Field names in LabelMe annotations must exactly match these names:

**From fake data (app.py):**
- `FirstName`, `LastName`, `FullName`, `MiddleInitial`
- `DOB` (MM/DD/YYYY), `SSN`, `Gender`
- `Street1`, `Street2`, `City`, `State`, `Zip`
- `AccountID`, `HealthBenefitID`

**Auto-generated fields:**
- `Address` (combined address)
- `Email` (auto-generated)
- `Date` (current date)
- `TelephoneNumber` (generated phone)
- `Signature` (uses FullName)

## Setup Steps

### 1. Create Directory Structure
All new forms must be organized in folders under `forms\`:
```
forms\
├── {form_type}\           # Your new form type folder
│   ├── {prefix}_{suffix}.png    # Template files
│   └── {prefix}_{suffix}.json   # LabelMe annotations
```

Example:
```powershell
mkdir forms/{form_type}
# Creates: forms\drivers_license\
```

### 2. Add Templates
Place `{prefix}_{suffix}.png` files in the directory.

### 3. Update Code
Add to `DOCUMENT_SUBTYPES` in `document_generation/generate_document.py`:
```python
'drivers_license': ['ny_drivers_license', 'ca_drivers_license']
```

### 4. Annotate with LabelMe
```powershell
# Run labelme from the root directory
python labelme
# Navigate to forms/{form_type}/ and choose template you want to annotate
```
- Draw rectangles/polygons around fields
- Label with exact field names from above
- Save annotations

### 5. Create and Update config.json
You must have a `config.json` file in the root directory to specify what documents to generate.  
See [JSON_INSTRUCTION_SETUP.md](JSON_INSTRUCTION_SETUP.md) for detailed instructions and examples on how to format this file.

**Example:**
```json
{
  "documents": [
    {
      "type": "single",
      "num_docs": 5,
      "quality": "clear", 
      "form_type": "drivers_license",
      "form_directory": "forms/drivers_license"
    }
  ]
}
```

## Run

```powershell
python run.py
```

## Common Issues

- **"Form type not found"**: Add to `DOCUMENT_SUBTYPES` dictionary in `document_generation/generate_document.py`
- **"Template files not found"**: Check file naming `{prefix}_{suffix}.png` in `forms/{form_type}/`
- **Empty fields**: Field names must match exactly (case-sensitive)
- **Missing annotations**: Run LabelMe and save `.json` files in `forms/{form_type}/`
- **Output not found**: Check `output/` directory for generated files