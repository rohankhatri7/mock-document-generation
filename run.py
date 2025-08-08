import sys
import json
import random
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

sys.path.append(str(Path(__file__).parent / 'document_generation'))
sys.path.append(str(Path(__file__).parent / 'template_creation'))

from generate_document import DOCUMENT_SUBTYPES

class DirectDocumentGenerator:
    def __init__(self, config_path: str = "config.json"):
        self.base_dir = Path(__file__).parent
        self.config_path = self.base_dir / config_path

        self.forms_dir = self.base_dir / 'forms'
        self.template_creation_dir = self.base_dir / 'template_creation'
        self.document_generation_dir = self.base_dir / 'document_generation'
        self.output_dir = self.base_dir / 'output'
        
        self.fake_data_path = self.base_dir / 'fakedata.csv'
        
        # Initialize ground truth tracking
        self.ground_truth = {
            "generation_timestamp": datetime.now().isoformat(),
            "config_used": None,
            "documents": []
        }
        
        # Load config
        self.config = self._load_config()
        self.ground_truth["config_used"] = self.config
        
        # Ensure output directories exist
        (self.output_dir / 'clean_templates').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'documents').mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_fake_data(self) -> List[Dict[str, Any]]:
        """Load fake data from CSV"""
        if not self.fake_data_path.exists():
            print("  Fake data not found, generating...")
            self._generate_fake_data()
        
        df = pd.read_csv(self.fake_data_path)
        rows = df.to_dict('records')
        
        if not rows:
            raise ValueError("No data found in CSV file")
        
        return rows
    
    def _generate_fake_data(self):
        # Generate fake data using app.py
        import os
        
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            [sys.executable, 'app.py'],
            cwd=str(self.base_dir),
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=env
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate fake data: {result.stderr}")

    def _clean_template(self, form_type: str) -> tuple:
        # Clean template and return paths to cleaned template and spec
        # Pick a random subtype for the form type
        if form_type in DOCUMENT_SUBTYPES:
            subtype = random.choice(DOCUMENT_SUBTYPES[form_type])
        else:
            subtype = form_type
        
        # Check if cleaned files already exist
        clean_template_path = self.output_dir / 'clean_templates' / f'{subtype}_clean.png'
        spec_path = self.output_dir / 'clean_templates' / f'{subtype}_spec.json'
        
        if clean_template_path.exists() and spec_path.exists():
            print(f"  Using existing cleaned template for {form_type} (subtype: {subtype})")
            return str(clean_template_path), str(spec_path), subtype
        
        # If files don't exist, clean the template
        print(f"  Cleaning template for {form_type} (subtype: {subtype})")
        
        result = subprocess.run(
            [sys.executable, 'clean_template.py', subtype],
            cwd=str(self.template_creation_dir),
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clean template: {result.stderr}")
        
        if not clean_template_path.exists() or not spec_path.exists():
            raise FileNotFoundError(f"Template cleaning failed - output files not found")
        
        return str(clean_template_path), str(spec_path), subtype

    def run(self):
        # Run the complete document generation workflow
        try:
            print("Starting Direct Document Generation Workflow")
            
            # Load fake data
            print("Loading fake data...")
            fake_data = self._load_fake_data()
            
            # Process each document configuration
            for doc_config in self.config['documents']:
                self._process_document_config(doc_config, fake_data)
            
            # Save ground truth
            self._save_ground_truth()
            
            print("Workflow completed successfully!")
            
        except Exception as e:
            print(f"Workflow failed: {e}")
            raise

    def _process_document_config(self, doc_config: Dict[str, Any], fake_data: List[Dict[str, Any]]):
        doc_type = doc_config.get('type', 'single')
        
        if doc_type == 'single':
            self._process_single_documents(doc_config, fake_data)
        elif doc_type == 'multi':
            self._process_multi_documents(doc_config, fake_data)
        else:
            raise ValueError(f"Unknown document type: {doc_type}")

    def _process_single_documents(self, doc_config: Dict[str, Any], fake_data: List[Dict[str, Any]]):
        form_type = doc_config['form_type']
        num_docs = doc_config.get('num_docs', 1)
        quality = doc_config.get('quality', 'unclear')
        
        print(f"Generating {num_docs} {quality} {form_type} documents...")
        
        # Clean template
        clean_template_path, spec_path, subtype = self._clean_template(form_type)
        
        # Generate documents
        for i in range(num_docs):
            data_row = random.choice(fake_data)
            
            output_filename = f"{subtype}_{quality}_{i+1}.png"
            
            self._generate_single_document(
                clean_template_path, 
                spec_path, 
                data_row, 
                quality,
                output_filename
            )
            
            # Track in ground truth
            self._add_to_ground_truth(
                document_type="single",
                filename=output_filename,
                form_type=form_type,
                subtype=subtype,
                quality=quality,
                data_used=data_row,
                pages=[{
                    "page_number": 1,
                    "form_type": form_type,
                    "subtype": subtype,
                    "data": data_row
                }]
            )

    def _process_multi_documents(self, doc_config: Dict[str, Any], fake_data: List[Dict[str, Any]]):
        form_types = doc_config['form_types']
        num_docs = doc_config.get('num_docs', 1)
        quality = doc_config.get('quality', 'unclear')
        
        print(f"Generating {num_docs} multi-page {quality} documents...")
        
        for i in range(num_docs):
            page_images = []
            pages_info = []
            
            # Generate each page with DIFFERENT data
            for page_num, form_config in enumerate(form_types, 1):
                form_type = form_config['form_type']
                
                # Use DIFFERENT data row for each page
                data_row = random.choice(fake_data)
                
                clean_template_path, spec_path, subtype = self._clean_template(form_type)
                
                # Generate page image
                page_img = self._generate_single_document(
                    clean_template_path,
                    spec_path,
                    data_row,  
                    quality,
                    None 
                )
                page_images.append(page_img)
                
                pages_info.append({
                    "page_number": page_num,
                    "form_type": form_type,
                    "subtype": subtype,
                    "data": data_row  # Each page has its own data
                })
            
            # Combine pages into PDF
            output_filename = f"multi_doc_{quality}_{i+1}.pdf"
            self._combine_pages_to_pdf(page_images, output_filename)
            
            # Track in ground truth with different data for each page
            self._add_to_ground_truth(
                document_type="multi",
                filename=output_filename,
                form_type="multi_page",
                subtype="combined",
                quality=quality,
                data_used=None,
                pages=pages_info
            )

    def _generate_single_document(self, clean_template_path: str, spec_path: str, 
                                data_row: Dict[str, Any], quality: str, output_filename: str = None):
        # Import and use DocumentGenerator
        from generate_document import DocumentGenerator
        
        generator = DocumentGenerator(clean_template_path, spec_path, quality=quality)
        
        if output_filename:
            output_path = self.output_dir / 'documents' / output_filename
        else:
            output_path = None
            
        return generator.generate(data_row, output_path)

    def _combine_pages_to_pdf(self, page_images: List, output_filename: str):
        from PIL import Image
        
        output_path = self.output_dir / 'documents' / output_filename
        
        # Convert all images to RGB
        rgb_images = [img.convert('RGB') for img in page_images]
        
        # Save as PDF
        rgb_images[0].save(
            output_path,
            save_all=True,
            append_images=rgb_images[1:],
            resolution=300
        )
        
        print(f"  Saved multi-page document: {output_filename}")

    def _add_to_ground_truth(self, document_type: str, filename: str, form_type: str, 
                           subtype: str, quality: str, data_used: Dict[str, Any], pages: List[Dict]):
        if document_type == "single":
            # Clean the data for single documents
            cleaned_data = {}
            for key, value in data_used.items():
                if key not in ["Filename", "Formtype"]:
                    if pd.isna(value) or value is None or str(value).lower() == 'nan':
                        cleaned_data[key] = "NA"
                    else:
                        cleaned_data[key] = value
            
            # Single page document structure
            document_entry = {
                "document_id": len(self.ground_truth["documents"]) + 1,
                "filename": filename,
                "document_type": document_type,
                "quality": quality,
                "data_used": cleaned_data
            }
        else:
            # Multi-page document structure with data at page level
            document_entry = {
                "document_id": len(self.ground_truth["documents"]) + 1,
                "filename": filename,
                "document_type": document_type,
                "quality": quality,
                "pages": []
            }
            
            for page_info in pages:
                # Clean the data for this page
                cleaned_data = {}
                for key, value in page_info["data"].items():
                    if key not in ["Filename", "Formtype"]:
                        if pd.isna(value) or value is None or str(value).lower() == 'nan':
                            cleaned_data[key] = "NA"
                        else:
                            cleaned_data[key] = value
                
                page_entry = {
                    "page_number": page_info["page_number"],
                    "form_type": page_info["form_type"],
                    "subtype": page_info["subtype"],
                    "data_used": cleaned_data
                }
                document_entry["pages"].append(page_entry)
        
        self.ground_truth["documents"].append(document_entry)

    def _save_ground_truth(self):
        # save ground truth to JSON
        ground_truth_path = self.output_dir / 'complete_ground_truth.json'
        
        simplified_ground_truth = {
            "documents": self.ground_truth["documents"]
        }
        
        with open(ground_truth_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_ground_truth, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Ground truth saved to: {ground_truth_path}")

# Main execution
if __name__ == "__main__":
    generator = DirectDocumentGenerator()
    generator.run()