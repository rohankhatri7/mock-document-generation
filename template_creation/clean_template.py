import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from PIL import Image

class TemplateCleaner:
    def __init__(self):
        self.output_dir = Path(__file__).parent.parent / 'output' / 'clean_templates'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update forms directory reference to root level
        self.forms_dir = Path(__file__).parent.parent / 'forms'

    def load_template(self, template_path):
        self.template_path = Path(template_path)
        # Load image in BGR then convert to RGB
        bgr = cv2.imread(str(template_path))
        if bgr is None:
            raise ValueError(f"Could not load image: {template_path}")
        self.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        self.mask = np.zeros((self.height, self.width), dtype="uint8")
        self.fields = []
        
    def load_annotations(self, json_path):
        # Load annotations from LabelMe JSON file.
        import json
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.width = data['imageWidth']
        self.height = data['imageHeight']
        
        # Parse shapes
        self.annotations = []
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            
            # Convert polygon points to bounding box
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)
            
            # Convert to relative coordinates [x, y, w, h]
            x = xmin / self.width
            y = ymin / self.height
            w = (xmax - xmin) / self.width
            h = (ymax - ymin) / self.height
            
            self.annotations.append({
                'label': label,
                'bbox': [x, y, w, h],
                'shape_type': shape.get('shape_type', 'rectangle')
            })
            
            # Ensure coordinates are within image bounds
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(self.width, int(xmax))
            ymax = min(self.height, int(ymax))
            
            cv2.rectangle(self.mask, (xmin, ymin), (xmax, ymax), 255, -1)
            
            # Store field info
            self.fields.append({
                'name': label,
                'bbox': [
                    xmin / self.width,
                    ymin / self.height,
                    (xmax - xmin) / self.width,
                    (ymax - ymin) / self.height
                ]
            })
    
    def clean_template(self, subtype: str):
        # Find the form type by scanning forms directory
        form_type = None
        for form_dir in self.forms_dir.iterdir():
            if form_dir.is_dir():
                template_file = form_dir / f'{subtype}.png'
                if template_file.exists():
                    form_type = form_dir.name
                    break
        
        if not form_type:
            raise FileNotFoundError(f"Could not find template for subtype: {subtype}")
        
        # Use the found form_type directory
        template_path = self.forms_dir / form_type / f'{subtype}.png'
        annotation_path = self.forms_dir / form_type / f'{subtype}.json'
        
        output_template_path = self.output_dir / f'{subtype}_clean.png'
        output_spec_path = self.output_dir / f'{subtype}_spec.json'
        
        img_rgb = self.image.copy()
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if method == 'white':
            white = np.full_like(img_rgb, 255)
            img_rgb[mask > 0] = white[mask > 0]
            cleaned = img_rgb
            if blur_kernel and blur_kernel % 2 == 1 and blur_kernel > 1:
                mask_blur = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
                mask_blur_3 = np.stack([mask_blur]*3,-1)/255.0
                base = img_rgb.astype(float)
                cleaned = (mask_blur_3*base + (1-mask_blur_3)*cleaned).astype(np.uint8)
        else:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            border = 5
            mask_with_border = cv2.copyMakeBorder(mask, border, border, border, border, 
                                               cv2.BORDER_CONSTANT, value=0)
            
            mask_dilated = cv2.dilate(mask_with_border, kernel, iterations=2)
            
            img_with_border = cv2.copyMakeBorder(img_bgr, border, border, border, border, 
                                              cv2.BORDER_REPLICATE)
            
            inpaint_radius = 3
            cleaned_bgr = cv2.inpaint(
                img_with_border,
                mask_dilated,
                inpaint_radius,
                flags=cv2.INPAINT_TELEA
            )
            
            h, w = img_bgr.shape[:2]
            cleaned_bgr = cleaned_bgr[border:border+h, border:border+w]
            
            cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
            
            smoothed = cv2.edgePreservingFilter(cleaned_rgb, flags=1, sigma_s=30, sigma_r=0.15)
            if blur_kernel and blur_kernel % 2 == 1 and blur_kernel > 1:
                smoothed = cv2.GaussianBlur(smoothed, (blur_kernel, blur_kernel), 0)
            
            mask_3ch = np.stack([mask] * 3, axis=-1).astype(bool)
            img_rgb[mask_3ch] = smoothed[mask_3ch]
            cleaned = img_rgb
        
        Image.fromarray(cleaned).save(output_template_path, quality=95, dpi=(300, 300))
        return output_template_path
    
    def save_spec(self, output_path=None):
        # save template to json
        if output_path is None:
            output_path = self.template_path.parent / f"{self.template_path.stem}_spec.json"
            
        spec = {
            'template_path': str(self.template_path.name),
            'width': self.width,
            'height': self.height,
            'fields': self.fields
        }
        
        with open(output_path, 'w') as f:
            json.dump(spec, f, indent=2)
            
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Clean document template and generate spec file')
    parser.add_argument('doc_type', help='Document type (e.g., passport, adp_paystub)')
    parser.add_argument('--output-dir', default='output/clean_templates', help='Output directory (default: output/clean_templates)')
    parser.add_argument('--method', default='inpaint_enhanced', 
                       choices=['inpaint_enhanced', 'white'], 
                       help='Cleaning method (default: inpaint_enhanced)')
    parser.add_argument('--blur', type=int, default=5, 
                       help='Gaussian blur kernel size (odd number, default: 5)')
    
    args = parser.parse_args()
    
    templates_dir = Path(__file__).parent / 'forms'
    output_dir = Path(args.output_dir)
    
    base_type = args.doc_type.split('_')[-1]  # e.g., 'adp_paystub' -> 'paystub'
    template_path = templates_dir / base_type / f"{args.doc_type}.png"
    json_path = templates_dir / base_type / f"{args.doc_type}.json"
    
    # Check if files exist
    if not template_path.exists():
        print(f" Template not found: {template_path}")
        print(f"Please make sure '{args.doc_type}.png' exists in the forms/ directory")
        return
        
    if not json_path.exists():
        print(f" Annotation file not found: {json_path}")
        print(f"Please create annotations using LabelMe and save as '{args.doc_type}.json'")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        cleaner = TemplateCleaner()
        cleaner.load_template(template_path)
        cleaner.load_annotations(json_path)
        cleaned_path = cleaner.clean_template(output_dir / f"{args.doc_type}_clean.png", method=args.method, blur_kernel=args.blur)
        spec_path = cleaner.save_spec(output_dir / f"{args.doc_type}_spec.json")
        print(f"Cleaned template: {cleaned_path}")
        print(f"Template spec: {spec_path}")
        
    except Exception as e:
        print(f" Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()