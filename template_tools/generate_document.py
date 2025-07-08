import json
import csv
import os
import io
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import argparse
import random

FONTS_DIR = Path(__file__).parent.parent / 'fonts'
DEFAULT_FONT = FONTS_DIR / 'OpenSans_SemiCondensed-Regular.ttf'
SIGNATURE_FONT = FONTS_DIR / 'signature.ttf'
# Available handwriting style fonts
HANDWRITING_FONTS = [
    FONTS_DIR / 'handwriting.ttf',
    FONTS_DIR / 'handwriting2.ttf',  
    FONTS_DIR / 'handwriting3.ttf'
]

# Map base document types to their available subtypes
DOCUMENT_SUBTYPES = {
    'passport': ['us_passport', 'india_passport'],
    'paystub': ['adp_paystub', 'paychex_paystub'],
    'ssn': ['us_ssn'],
}

class DocumentGenerator:
    def __init__(self, template_path, spec_path, font_path=None, font_size=12, quality='unclear'):
        try:
            self.template = Image.open(template_path).convert('RGBA')
            self.spec = self._load_spec(spec_path)
            self.font_path = str(font_path) if font_path else None
            self.font_size = font_size
            self.quality = quality.lower()
            if self.quality not in ['clear', 'unclear']:
                self.quality = 'unclear'  # Default to unclear if invalid value provided
            self.font_cache = {}
            
            if not self.font_path and DEFAULT_FONT.exists():
                self.font_path = str(DEFAULT_FONT)
            
        except Exception as e:
            raise Exception(f"Failed to initialize DocumentGenerator: {str(e)}")
        
    def _load_spec(self, spec_path):
        with open(spec_path) as f:
            return json.load(f)
    
    def _get_font(self, field_name, default_size=None):
        size = default_size or self.font_size
        cache_key = f"{field_name}_{size}"
        
        if cache_key not in self.font_cache:
            try:
                if self.font_path and os.path.exists(self.font_path):
                    try:
                        font = ImageFont.truetype(self.font_path, size)
                        self.font_cache[cache_key] = font
                    except Exception as e:
                        print(f"Could not load specified font {self.font_path}: {e}")
                        raise
                else:
                    raise FileNotFoundError("No font path provided")
            except Exception as e:
                print(f"Could not load font: {e}. Using default font.")
                self.font_cache[cache_key] = ImageFont.load_default()
                    
        return self.font_cache[cache_key]
    
    def _add_noise_effects(self, img): #add realistic noise
        if self.quality != 'unclear':
            return img.convert('RGBA')
            
        img = img.convert('RGBA')
        
        original_size = img.size
        
        scale_down = random.uniform(0.3, 0.5)  # 30-50% of original size (less aggressive)
        
        #downscale image
        small_size = (
            max(20, int(original_size[0] * scale_down * random.uniform(0.95, 1.05))),  # ±5% aspect ratio
            max(20, int(original_size[1] * scale_down * random.uniform(0.95, 1.05)))
        )
        img = img.resize(small_size, Image.Resampling.LANCZOS)
        
        #upscale image
        scale_up = random.uniform(2.0, 3.0)
        large_size = (
            int(small_size[0] * scale_up * random.uniform(0.98, 1.02)),  # ±2% aspect ratio
            int(small_size[1] * scale_up * random.uniform(0.98, 1.02))
        )
        img = img.resize(large_size, Image.Resampling.NEAREST)
        
        # Add slight rotation
        rotation_angle = random.uniform(-2, 2)
        img = img.rotate(rotation_angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
        
        # Final resize to original dimensions with bilinear for better quality
        img = img.resize(original_size, Image.Resampling.BILINEAR)
        
        # Add moderate JPEG compression
        if random.random() > 0.3:  # 70% chance
            buffer = io.BytesIO()
            img.convert('RGB').save(buffer, format='JPEG', quality=random.randint(30, 60))
            img = Image.open(buffer).convert('RGBA')
        
        # Add subtle noise
        if random.random() > 0.3:  # 70% chance, less frequent
            noise = np.random.randint(0, 30, (img.size[1], img.size[0], 3), dtype=np.uint8)  # Less intense noise
            noise_img = Image.fromarray(noise).convert('RGBA')
            img = Image.blend(img, noise_img, alpha=0.05)  # More subtle blend
        
        # Random blur
        if random.random() > 0.3:  # 70% chance of blur
            blur_radius = random.uniform(1.2, 3.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Random brightness/contrast adjustment
        if random.random() > 0.6:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.95, 1.05))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.97, 1.03))
        
        # Add grain
        if random.random() > 0.2:
            np_img = np.array(img.convert('L'))
            noise = np.random.normal(0, random.uniform(0.5, 1.5), np_img.shape).astype(np.uint8)
            noise_img = Image.fromarray(np.clip(np_img.astype(np.int16) + noise, 0, 255).astype(np.uint8), 'L')
            img = Image.blend(img.convert('L'), noise_img, 0.2)
        
        return img.convert('RGBA')

    def _add_printer_artifacts(self, img, intensity=0.02):
        """Add realistic grayscale printer/fax artifacts like streaks and dots"""
        if img.mode != 'L':
            img = img.convert('L').convert('RGBA')
        
        width, height = img.size
        
        # create a transparent overlay for artifacts to appear on top
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, 'RGBA')
        
        # add random dots
        dot_count = int(width * height * intensity * 1.08 / 100)
        for _ in range(dot_count):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            size = random.randint(3, 6)
            darkness = random.randint(120, 200)
            gray_value = random.randint(0, 80)
            draw.ellipse([x, y, x + size, y + size], fill=(gray_value, gray_value, gray_value, darkness))
        
        # add vertical streaks
        for _ in range(random.randint(1, 5)):
            x = random.randint(0, width - 1)
            streak_width = random.randint(1, 3)
            darkness = min(150, int(random.randint(20, 80) * 1.10))
            gray_value = random.randint(0, 60)
            for dx in range(streak_width):
                if x + dx < width:
                    for y in range(0, height, 2):
                        if random.random() < 0.7:
                            draw.point((x + dx, y), fill=(gray_value, gray_value, gray_value, darkness))
        
        # add some random horizontal streaks
        if random.random() < 0.3:
            y = random.randint(0, height - 1)
            streak_height = random.randint(1, 2)
            darkness = min(150, int(random.randint(15, 60) * 1.10))
            gray_value = random.randint(0, 70)
            for dy in range(streak_height):
                if y + dy < height:
                    for x in range(0, width, 2):
                        if random.random() < 0.6:
                            draw.point((x, y + dy), fill=(gray_value, gray_value, gray_value, darkness))
        
        # add subtle noise in grayscale
        noise = Image.effect_noise((width, height), 10)
        noise = noise.convert('L')
        noise = Image.merge('RGBA', (noise, noise, noise, Image.new('L', (width, height), 10)))
        
        # composite everything together
        result = Image.alpha_composite(img, overlay)
        result = Image.alpha_composite(result, noise)
        
        return result

    def _composite_on_a4(self, img, data=None):
        """Composite the document onto an A4 page with grayscale artifacts"""
        try:
            # create a4 sheet
            a4_width, a4_height = 2480, 3508
            
            # create a blank grayscale a4 sheet with white background
            a4_img = Image.new('L', (a4_width, a4_height), 255).convert('RGBA')
            
            a4_img = self._add_printer_artifacts(a4_img)
            
            doc_width, doc_height = img.size
            x = (a4_width - doc_width) // 2
            y = (a4_height - doc_height) // 2
            
            # convert document to grayscale if needed
            if img.mode != 'RGBA':
                img = img.convert('L').convert('RGBA')
            
            #add a shadow
            shadow = Image.new('L', (doc_width + 20, doc_height + 20), 255)
            shadow_draw = ImageDraw.Draw(shadow)
            for i in range(10, 0, -1):
                shadow_draw.rectangle([i, i, doc_width + 20 - i, doc_height + 20 - i], 
                                   outline=200, width=1)
            shadow = shadow.convert('RGBA')
            shadow = Image.eval(shadow, lambda x: 0 if x < 255 else 0) 
            a4_img.alpha_composite(shadow, (x - 10, y - 10))
            
            # paste the document onto the A4 page
            a4_img.paste(img, (x, y), img.split()[3] if img.mode == 'RGBA' else None)
            
            # add noise
            noise = Image.effect_noise((a4_width, a4_height), 10)
            noise = noise.convert('L')
            noise = Image.merge('RGBA', (noise, noise, noise, Image.new('L', (a4_width, a4_height), 15)))
            a4_img = Image.alpha_composite(a4_img, noise)
            
            # add a border
            border = Image.new('L', (doc_width + 10, doc_height + 10), 255)
            border_draw = ImageDraw.Draw(border)
            border_draw.rectangle([0, 0, doc_width + 9, doc_height + 9], 
                               outline=200, width=1)
            border = border.convert('RGBA')
            border = Image.eval(border, lambda x: 0 if x < 255 else 0) 
            a4_img.alpha_composite(border, (x - 5, y - 5))
            
            return a4_img.convert('L').convert('RGB')
            
        except Exception as e:
            print(f"Error during A4 composition: {e}")
            
            return img.convert('RGB')

    def _get_optimal_font_size(self, draw, text, font_path, max_width, max_height, min_size=6, max_size=100):
        """Find the optimal font size that makes the text fit within max_width and max_height"""
        if not text or not font_path.exists():
            return min_size
            
        left, right = min_size, max_size
        best_size = min_size
        
        while left <= right:
            mid = (left + right) // 2
            try:
                font = ImageFont.truetype(str(font_path), mid)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                if text_width <= max_width and text_height <= max_height:
                    best_size = mid
                    left = mid + 1
                else:
                    right = mid - 1
            except Exception:
                right = mid - 1
                
        return best_size

    def _add_handwritten_annotations(self, img, data):
        """Add handwritten annotations for AccountID and HealthBenefitID using labeled areas"""
        if not data:
            return img
            
        draw = ImageDraw.Draw(img)
        
        # check for labeled areas in the spec
        combined_areas = [f for f in self.spec.get('fields', []) 
                         if set(f['name'].split(',')).issuperset({'AccountID', 'HealthBenefitID'})]
        account_areas = [f for f in self.spec.get('fields', []) if f['name'] == 'AccountID']
        health_areas = [f for f in self.spec.get('fields', []) if f['name'] == 'HealthBenefitID']
        
        # select a single random handwriting font for both IDs to ensure consistency
        hw_font_path = random.choice(HANDWRITING_FONTS)
        
        # handle combined AccountID,HealthBenefitID label
        if combined_areas and ('AccountID' in data or 'HealthBenefitID' in data):
            area = combined_areas[0]  # Use first matching area
            bbox = area['bbox']
            x = int(bbox[0] * img.width)
            y = int(bbox[1] * img.height)
            w = int(bbox[2] * img.width)
            h = int(bbox[3] * img.height)
            
            # Calculate available height for each line (split in half)
            line_height = h // 2
            
            # Draw AccountID if present
            if 'AccountID' in data and data['AccountID']:
                account_text = f"Account: {data['AccountID']}"
                # Calculate optimal font size for this text
                font_size = self._get_optimal_font_size(draw, account_text, hw_font_path, w, line_height * 0.9)
                try:
                    hw_font = ImageFont.truetype(str(hw_font_path), font_size)
                    # Get text size to center it vertically within its line
                    text_bbox = draw.textbbox((0, 0), account_text, font=hw_font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    # Center text in the line
                    text_y = y + (line_height - text_h) // 2
                    draw.text((x, text_y), account_text, font=hw_font, fill=(80, 80, 80, 220))
                except Exception as e:
                    print(f"Error rendering AccountID: {e}")
            
            # Draw HealthBenefitID below AccountID if present
            if 'HealthBenefitID' in data and data['HealthBenefitID']:
                health_text = f"Health: {data['HealthBenefitID']}"
                # Calculate optimal font size for this text
                font_size = self._get_optimal_font_size(draw, health_text, hw_font_path, w, line_height * 0.9)
                try:
                    hw_font = ImageFont.truetype(str(hw_font_path), font_size)
                    # Get text size to center it vertically within its line
                    text_bbox = draw.textbbox((0, 0), health_text, font=hw_font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    # Center text in the line (second line)
                    text_y = y + line_height + (line_height - text_h) // 2
                    draw.text((x, text_y), health_text, font=hw_font, fill=(80, 80, 80, 220))
                except Exception as e:
                    print(f"Error rendering HealthBenefitID: {e}")
                    
        # handle separate AccountID and HealthBenefitID labels (for documents on A4)
        else:
            if 'AccountID' in data and data['AccountID'] and account_areas:
                area = account_areas[0]
                bbox = area['bbox']
                x = int(bbox[0] * img.width)
                y = int(bbox[1] * img.height)
                w = int(bbox[2] * img.width)
                h = int(bbox[3] * img.height)
                
                account_text = f"Account: {data['AccountID']}"
                font_size = self._get_optimal_font_size(draw, account_text, hw_font_path, w, h)
                
                try:
                    hw_font = ImageFont.truetype(str(hw_font_path), font_size)
                    text_bbox = draw.textbbox((0, 0), account_text, font=hw_font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    text_x = x + (w - text_w) // 2
                    text_y = y + (h - text_h) // 2
                    draw.text((text_x, text_y), account_text, font=hw_font, fill=(80, 80, 80, 220))
                except Exception as e:
                    print(f"Error rendering AccountID: {e}")
            
            if 'HealthBenefitID' in data and data['HealthBenefitID'] and health_areas:
                area = health_areas[0]
                bbox = area['bbox']
                x = int(bbox[0] * img.width)
                y = int(bbox[1] * img.height)
                w = int(bbox[2] * img.width)
                h = int(bbox[3] * img.height)
                
                health_text = f"Health: {data['HealthBenefitID']}"
                font_size = self._get_optimal_font_size(draw, health_text, hw_font_path, w, h)
                
                try:
                    hw_font = ImageFont.truetype(str(hw_font_path), font_size)
                    text_bbox = draw.textbbox((0, 0), health_text, font=hw_font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    text_x = x + (w - text_w) // 2
                    text_y = y + (h - text_h) // 2
                    draw.text((text_x, text_y), health_text, font=hw_font, fill=(80, 80, 80, 220))
                except Exception as e:
                    print(f"Error rendering HealthBenefitID: {e}")
        
        return img

    def generate(self, data, output_path=None):
        try:
            img = self.template.copy()
            draw = ImageDraw.Draw(img)
            
            # check if full-page document by looking for AccountID/HealthBenefitID fields
            has_id_fields = any(
                set(f['name'].split(',')).issuperset({'AccountID', 'HealthBenefitID'}) or 
                f['name'] in ['AccountID', 'HealthBenefitID'] 
                for f in self.spec.get('fields', [])
            )
            
            # first, render all the fields from the spec
            for field in self.spec.get('fields', []):
                name = field['name']

                if name in ['AccountID', 'HealthBenefitID'] or \
                   set(name.split(',')).issuperset({'AccountID', 'HealthBenefitID'}):
                    continue
                x, y, w, h = field['bbox']
                
                if w <= 0 or h <= 0:
                    continue
                # convert labelme coords to absolute
                abs_x = int(x * self.spec['width'])
                abs_y = int(y * self.spec['height'])
                abs_w = int(w * self.spec['width'])
                abs_h = int(h * self.spec['height'])
                
                # handle multiple fields in one label (comma-separated)
                field_names = [n.strip() for n in field['name'].split(',')]
                
                # group city and state together if they appear consecutively
                processed_fields = []
                i = 0
                while i < len(field_names):
                    if field_names[i] == 'City' and i + 1 < len(field_names) and field_names[i+1] == 'State':
                        processed_fields.append(('City,State', i, i+1))
                        i += 2
                    else:
                        processed_fields.append((field_names[i], i, i))
                        i += 1
                
                field_values = []
                for field_info in processed_fields:
                    field_name, start_idx, end_idx = field_info
                    
                    if field_name == 'City,State':
                        # get both city and state values
                        city = str(data.get('City', '')).strip()
                        state = str(data.get('State', '')).strip()
                        if city and state:
                            value = f"{city}, {state}"
                        else:
                            value = city or state
                        if value:
                            field_values.append(value)
                    else:
                        value = str(data.get(field_name, '')).strip()
                        if value:
                            field_values.append(value)
                
                if not field_values:
                    continue
                
                max_font_size = 72
                best_size = 0
                
                for size in range(12, max_font_size + 1):
                    try:
                        font = self._get_font(name, size)
                        total_height = 0
                        max_width = 0
                        
                        for value in field_values:
                            bbox = draw.textbbox((0, 0), value, font=font)
                            total_height += (bbox[3] - bbox[1]) * 1.08  # 8% spacing
                            max_width = max(max_width, bbox[2] - bbox[0])
                        
                        if total_height <= abs_h * 0.9 and max_width <= abs_w * 0.9:
                            best_size = size
                        else:
                            break
                            
                    except Exception:
                        break
                        
                if best_size > 0:
                    try:
                        font = self._get_font(name, best_size)
                        y_offset = 0  
                        
                        for value in field_values:
                            if value:
                                text_bbox = draw.textbbox((0, 0), value, font=font)
                                text_width = text_bbox[2] - text_bbox[0]
                                text_height = text_bbox[3] - text_bbox[1]
                                
                                text_x = abs_x + 2  
                                text_y = abs_y + y_offset
                                
                                draw.text((text_x, text_y), value, font=font, fill=(0, 0, 0, 255))
                                
                                y_offset += text_height * 1.08  # 8% spacing between lines
                        
                    except Exception:
                        pass 
            
            signature_field = next((f for f in self.spec['fields'] if f['name'].lower() == 'signature'), None)
            if signature_field and 'FullName' in data:
                try:
                    x, y, w, h = signature_field['bbox']
                    abs_x = int(x * self.spec['width'])
                    abs_y = int(y * self.spec['height'])
                    abs_w = int(w * self.spec['width'])
                    abs_h = int(h * self.spec['height'])
                    
                    signature_text = data['FullName']
                    
                    max_font_size = min(72, abs_h)
                    min_font_size = 6
                    best_size = min_font_size
                    
                    try:
                        # binary search for best font size
                        low = min_font_size
                        high = max_font_size
                        
                        while low <= high:
                            mid = (low + high) // 2
                            try:
                                font = ImageFont.truetype(str(SIGNATURE_FONT), mid)
                                text_width = draw.textlength(signature_text, font=font)
                                
                                if text_width <= abs_w * 0.9:
                                    best_size = mid
                                    low = mid + 1
                                else:
                                    high = mid - 1
                            except Exception:
                                high = mid - 1
                        
                        sig_font = ImageFont.truetype(str(SIGNATURE_FONT), best_size)
                        
                        text_bbox = draw.textbbox((0, 0), signature_text, font=sig_font)
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        text_x = abs_x + 5
                        text_y = abs_y + 2
                        
                        draw.text((text_x, text_y), signature_text, font=sig_font, fill=(0, 0, 0, 255))
                        
                    except Exception as e:
                        print(f"Could not use signature font, falling back to italic: {e}")
                        font = ImageFont.truetype(str(DEFAULT_FONT), min(14, abs_h))
                        font = font.font_variant(style='italic')
                        text_bbox = draw.textbbox((0, 0), signature_text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        if text_width > abs_w * 0.9:
                            scale_factor = (abs_w * 0.9) / text_width
                            font_size = int(min(14, abs_h) * scale_factor)
                            font = ImageFont.truetype(str(DEFAULT_FONT), max(6, font_size))
                            font = font.font_variant(style='italic')
                            text_bbox = draw.textbbox((0, 0), signature_text, font=font)
                            text_height = text_bbox[3] - text_bbox[1]
                        
                        text_x = abs_x + 5
                        text_y = abs_y + 2
                        
                        draw.text((text_x, text_y), signature_text, font=font, fill=(0, 0, 0, 255))
                
                except Exception as e:
                    print(f"Failed to render signature: {e}")
            
            # save or return the image
            # composite onto A4 before saving
            if output_path is None:
                return img
            
            # for documents with ID fields, add handwritten annotations, apply artifacts, and save
            if has_id_fields:
                img = self._add_handwritten_annotations(img, data)
                # Always apply artifacts to full-page documents
                img = self._add_printer_artifacts(img)
                # Apply DPI distortion / noise effects (shrink + enlarge) for realistic pixelation
                if self.quality == 'unclear':
                    img = self._add_noise_effects(img)
                img = img.convert('RGB')  # Convert to RGB before saving
                img.save(output_path, 'PNG', quality=95, dpi=(300, 300))
                return img
                
            # for regular documents, composite on A4 (which includes artifacts) and save
            final_img = self._composite_on_a4(img, data)
            # Apply DPI distortion after compositing for consistent texture (only if quality is unclear)
            if self.quality == 'unclear':
                final_img = self._add_noise_effects(final_img)
            final_img.save(output_path, 'PNG', quality=95, dpi=(300, 300))
            return final_img

        except Exception as e:
            print(f"Failed to generate document: {e}")

def generate_batch(generator, data_list, output_dir, count=None):
    """Generate multiple documents from a list of data dictionaries"""
    if not data_list:
        print("No data provided for document generation")
        return []
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # limit to requested count or all available data
    count = min(count, len(data_list)) if count else len(data_list)
    
    generated_files = []
    for i in range(count):
        output_path = output_dir / f"document{i+1}.png"
        try:
            generator.generate(data_list[i], output_path=output_path)
            generated_files.append(str(output_path))
        except Exception as e:
            print(f"Error generating document {i+1}: {e}")
    
    return generated_files

def main():
    parser = argparse.ArgumentParser(description='Generate documents from template and data')
    parser.add_argument('doc_type', help='Document type base (passport, paystub, ssn) or specific subtype (e.g., us_passport)')
    parser.add_argument('--subtype', help='Explicit subtype to use (e.g., us_passport). Overrides random choice.')
    parser.add_argument('--no-random', action='store_true', help='Disable random subtype selection when base doc_type is given')
    parser.add_argument('--data', default='data/sample_data.csv', help='Path to CSV data file (default: data/sample_data.csv)')
    parser.add_argument('--output-dir', default='output/documents', help='Output directory for generated documents (default: output/documents)')
    parser.add_argument('--row', type=int, help='Specific row to use from CSV (0-based)')
    parser.add_argument('--count', type=int, help='Number of documents to generate (default: 1 or number of rows in CSV)')
    parser.add_argument('--pdf', choices=['single', 'multi'], help="Output PDFs: 'single' = separate PDF per doc, 'multi' = combined multi-page PDF")
    parser.add_argument('--quality', choices=['clear', 'unclear'], default='unclear', 
                      help="Output quality: 'clear' for clean output, 'unclear' for realistic/noisy output (default: unclear)")
    
    args = parser.parse_args()
    
    # set up paths
    base_dir = Path(__file__).parent
    templates_dir = base_dir / 'templates'
    output_dir = Path(args.output_dir)
    
    # create output directories
    documents_dir = base_dir / 'output' / 'documents'
    pdfs_dir = base_dir / 'output' / 'pdfs'
    documents_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    
    # set output directory for PNGs
    output_dir = Path(args.output_dir) if args.output_dir else documents_dir
    
    # resolve subtype(s)
    if '_' in args.doc_type:
        # user already passed a full subtype
        base_type = args.doc_type.split('_')[-1]
        subtype_list = [args.doc_type]
    else:
        base_type = args.doc_type.lower()
        subtype_list = DOCUMENT_SUBTYPES.get(base_type)
        if not subtype_list:
            print(f"ERROR: Unknown document type '{args.doc_type}'. Available: {list(DOCUMENT_SUBTYPES.keys())}")
            return
    
    # validate explicit subtype
    if args.subtype:
        if args.subtype not in subtype_list:
            print(f"ERROR: Subtype '{args.subtype}' not valid for base type '{base_type}'. Choose from: {subtype_list}")
            return
        subtype_list = [args.subtype]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_file = Path(args.data)
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        return
    with open(data_file) as f:
        reader = csv.DictReader(f)
        data_rows = list(reader)
    if not data_rows:
        print("ERROR: No data found in the CSV file")
        return
    
    # determine how many docs to generate
    max_docs = len(data_rows)
    count = min(args.count, max_docs) if args.count else max_docs
    if args.count and args.count > max_docs:
        print(f"Only {max_docs} rows available, generating {count} documents")

    # choose a single subtype for this batch
    if len(subtype_list) == 1:
        chosen_subtype = subtype_list[0]
    else:
        chosen_subtype = subtype_list[0] if args.no_random else random.choice(subtype_list)
        print(f"Randomly selected subtype: {chosen_subtype}")

    cleaned_template = base_dir / 'output' / 'clean_templates' / f"{chosen_subtype}_clean.png"
    spec_file       = base_dir / 'output' / 'clean_templates' / f"{chosen_subtype}_spec.json"
    if not cleaned_template.exists() or not spec_file.exists():
        print(f"ERROR: Cleaned assets for '{chosen_subtype}' not found. Run 'python clean_template.py {chosen_subtype}' first.")
        return

    # create generator once with specified quality
    generator = DocumentGenerator(cleaned_template, spec_file, None, 12, args.quality)  # using default font and size

    generated_files = []
    for i in range(count):
        if not cleaned_template.exists() or not spec_file.exists():
            print(f"Skipping {subtype}: cleaned assets missing. Run 'python clean_template.py {subtype}' first.")
            continue
        
        try:
            output_path = output_dir / f"{chosen_subtype}_{i+1}.png"
            generator.generate(data_rows[i], output_path=output_path)
            generated_files.append(str(output_path))
        except Exception as e:
            print(f"Error generating document {i+1} ({chosen_subtype}): {e}")

    # summary
    if generated_files:
        print("\nGenerated documents:")
        for idx, fp in enumerate(generated_files, 1):
            print(f"  {idx}. {fp}")
        # optional PDF conversion
        if args.pdf:
            if args.pdf == 'single':
                print("\nGenerating single-page PDFs…")
                for img_path in generated_files:
                    # Create PDF path in the pdfs directory with the same filename
                    pdf_filename = Path(img_path).name.replace('.png', '.pdf')
                    pdf_path = pdfs_dir / pdf_filename
                    try:
                        with Image.open(img_path) as im:
                            im.convert('RGB').save(pdf_path, 'PDF', resolution=300)
                        print(f"  → {pdf_path}")
                    except Exception as e:
                        print(f"  Failed to convert {img_path} to PDF: {e}")
            elif args.pdf == 'multi':
                print("\nGenerating combined multi-page PDF…")
                pdf_path = pdfs_dir / f"{chosen_subtype}_batch.pdf"
                try:
                    with Image.open(generated_files[0]).convert('RGB') as img1:
                        img1.save(pdf_path, 'PDF', resolution=300, save_all=True, 
                                append_images=[Image.open(f).convert('RGB') for f in generated_files[1:]])
                    print(f"  → {pdf_path}")
                except Exception as e:
                    print(f"  Failed to create multi-page PDF: {e}")
    else:
        print("No documents generated due to previous errors.")
    cleaned_template = base_dir / 'output' / 'clean_templates' / f"{args.doc_type}_clean.png"
    spec_file = base_dir / 'output' / 'clean_templates' / f"{args.doc_type}_spec.json"
    data_file = Path(args.data)
    
    # check if files exist
    if not cleaned_template.exists():

        print(f"Please run 'python clean_template.py {args.doc_type}' first")
        return
        
    if not spec_file.exists():
        print(f"ERROR: Spec file not found: {spec_file}")
        print(f"Please run 'python clean_template.py {args.doc_type}' first")
        return
        
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        print("Please provide a valid CSV file with --data")
        return
    
    # output directories are already created at the beginning of the function
    
    try:
        generator = DocumentGenerator(cleaned_template, spec_file, None, 12)  # using default font and size
        
        with open(data_file) as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        if not data:
            print("ERROR: No data found in the CSV file")
            return
            
        count = min(args.count, len(data)) if args.count else len(data)
        
        if args.count and args.count > len(data):
            print(f"Only {len(data)} rows available, generating {count} documents")
        
        if count > 1:
            generated_files = generate_batch(generator, data[:count], output_dir, count)
            print("\nGenerated documents:")
            for i, file_path in enumerate(generated_files, 1):
                print(f"  {i}. {file_path}")
        else:
            output_path = output_dir / f"{args.doc_type}1.png"
            generator.generate(data[0], output_path=output_path)
            print(str(output_path))
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
