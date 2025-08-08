# Document Generation Workflow

Simple overview of how the system generates synthetic documents from templates.

## Required Folders

- **fonts/**: Place all font files (e.g., .ttf) here that will be used for rendering text and signatures on documents.
- **ny/**: This folder contains a dataset used for generating fake data (e.g., names, addresses, etc.). You can replace this with your own dataset if desired.
- **forms/**: Contains all document form templates and their corresponding LabelMe annotation files. Each form type should have its own subfolder.

## Workflow Process

The system follows this automated process:

### 1. **Generate Fake Data**
- Creates realistic synthetic data (names, addresses, SSNs, etc.)
- Saves to `fakedata.csv` for document population
- Run: `python app.py` (optional - auto-generated if needed)

### 2. **Clean Templates** 
- Takes original document templates from `forms/` directory
- Removes existing text and markings to create blank forms
- Produces clean templates in `output/clean_templates/`

### 3. **Generate Documents**
- Applies fake data to cleaned templates using LabelMe annotations
- Creates synthetic documents with realistic information
- Produces both single-page PNGs and multi-page PDFs in `output/documents/`

### 4. **Track Ground Truth**
- Records all generated documents with their associated data
- Tracks which data was placed in which fields for each document
- Saves comprehensive tracking in `output/complete_ground_truth.json`

## Usage

**Run the complete workflow:**
```powershell
python run.py
```

**Input:** JSON configuration file (`config.json`) specifying:
- Document types to generate
- Number of documents 
- Quality settings (clear/unclear)
- Single-page vs multi-page documents

**Output Locations:** 
- **Mock Documents**: `output/documents/` (PNG files for single-page, PDF files for multi-page)
- **Cleaned Templates**: `output/clean_templates/` (cleaned PNG templates + JSON spec files)
- **Ground Truth**: `output/complete_ground_truth.json` (data tracking for all generated documents)

## JSON Configuration

See [JSON_INSTRUCTION_SETUP.md](JSON_INSTRUCTION_SETUP.md) for formatting JSON instruction input.

## Setup New Document Types

See [MANUAL_SETUP.md](MANUAL_SETUP.md) for adding new document types to the system.