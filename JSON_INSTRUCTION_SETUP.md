## Usage

1. Create a `config.json` file following the format below
2. Place it in the same directory as `run.py` (root)
3. Run: `python run.py`

### Document Configuration Types

#### 1. Single Document Configuration
Generates individual document files.

```json
{
  "type": "single",                           // REQUIRED: must be "single"
  "num_docs": <integer>,                      // REQUIRED: number of copies to generate (positive integer)
  "quality": "<clear|unclear>",               // REQUIRED: generation quality ("clear" or "unclear")
  "form_type": "<form_type>",                 // REQUIRED: document type (see valid types below)
  "form_directory": "forms/<form_type>"  // REQUIRED: path to form templates
}
```

#### 2. Multi-Page Document Configuration
Generates PDF files with multiple form types as separate pages.

```json
{
  "type": "multi",                            // REQUIRED: must be "multi"
  "num_docs": <integer>,                      // REQUIRED: number of multi-page PDFs to generate
  "quality": "<clear|unclear>",               // REQUIRED: generation quality for all pages
  "form_types": [                             // REQUIRED: array of form types (minimum 1)
    {
      "form_type": "<form_type_1>",           // REQUIRED: first form type
      "form_directory": "forms/<form_type_1>"  // REQUIRED: path to templates
    },
    {
      "form_type": "<form_type_2>",           // REQUIRED: second form type
      "form_directory": "forms/<form_type_2>"  // REQUIRED: path to templates
    }
    // Add more form_type objects as needed
  ]
}
```

### Valid Form Types
The following form types are supported as of now:
- `passport`
- `ssn`
- `paystub`
- `empletter`
- `authrep`
- `i766`
- `taxreturn`

*Note: Each form type must have corresponding template files in the specified directory.*

## Complete Example

```json
{
  "documents": [
    {
      "type": "single",
      "num_docs": 2,
      "quality": "clear",
      "form_type": "passport",
      "form_directory": "forms/passport"
    },
    {
      "type": "single",
      "num_docs": 1,
      "quality": "unclear",
      "form_type": "ssn",
      "form_directory": "forms/ssn"
    },
    {
      "type": "multi",
      "num_docs": 1,
      "quality": "clear",
      "form_types": [
        {
          "form_type": "paystub",
          "form_directory": "forms/paystub"
        },
        {
          "form_type": "empletter",
          "form_directory": "forms/empletter"
        },
        {
          "form_type": "authrep",
          "form_directory": "forms/authrep"
        }
      ]
    },
    {
      "type": "multi",
      "num_docs": 2,
      "quality": "clear",
      "form_types": [
        {
          "form_type": "i766",
          "form_directory": "forms/i766"
        },
        {
          "form_type": "ssn",
          "form_directory": "forms/ssn"
        },
        {
          "form_type": "taxreturn",
          "form_directory": "forms/taxreturn"
        }
      ]
    }
  ]
}
```

Based on the example `config.json` above, the following files will be generated:

1. **Single Documents (3 files)**:
   - 2 clear passport documents: `us_passport_clean1.png`, `us_passport_clean2.png`
   - 1 unclear SSN document: `ssn1_ssn_unclean1.png`

2. **Multi-Page Documents (3 PDFs)**:
   - 1 PDF with 3 pages (paystub → empletter → authrep): `multi_config3_document_1.pdf`
   - 2 PDFs with 3 pages each (i766 → ssn → taxreturn): `multi_config4_document_1.pdf`, `multi_config4_document_2.pdf`

3. **Ground Truth File (1 JSON)**:
   - Complete tracking file: `complete_ground_truth.json`

**Total: 7 files generated (6 documents + 1 ground truth)**
