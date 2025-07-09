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
                self.quality = 'unclear'
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

    def _add_noise_effects(self, img):
        if self.quality != 'unclear':
            return img.convert('RGBA')

        img = img.convert('RGBA')
        original_size = img.size
        scale_down = random.uniform(0.3, 0.5)
        small_size = (
            max(20, int(original_size[0] * scale_down * random.uniform(0.95, 1.05))),
            max(20, int(original_size[1] * scale_down * random.uniform(0.95, 1.05)))
        )
        img = img.resize(small_size, Image.Resampling.LANCZOS)
        scale_up = random.uniform(2.0, 3.0)
        large_size = (
            int(small_size[0] * scale_up * random.uniform(0.98, 1.02)),
            int(small_size[1] * scale_up * random.uniform(0.98, 1.02))
        )
        img = img.resize(large_size, Image.Resampling.NEAREST)
        rotation_angle = random.uniform(-2, 2)
        img = img.rotate(rotation_angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
        img = img.resize(original_size, Image.Resampling.BILINEAR)
        if random.random() > 0.3:
            buffer = io.BytesIO()
            img.convert('RGB').save(buffer, format='JPEG', quality=random.randint(30, 60))
            img = Image.open(buffer).convert('RGBA')
        if random.random() > 0.3:
            noise = np.random.randint(0, 30, (img.size[1], img.size[0], 3), dtype=np.uint8)
            noise_img = Image.fromarray(noise).convert('RGBA')
            img = Image.blend(img, noise_img, alpha=0.05)
        if random.random() > 0.3:
            blur_radius = random.uniform(1.2, 3.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        if random.random() > 0.6:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.95, 1.05))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.97, 1.03))
        if random.random() > 0.2:
            np_img = np.array(img.convert('L'))
            noise = np.random.normal(0, random.uniform(0.5, 1.5), np_img.shape).astype(np.uint8)
            noise_img = Image.fromarray(np.clip(np_img.astype(np.int16) + noise, 0, 255).astype(np.uint8), 'L')
            img = Image.blend(img.convert('L'), noise_img, 0.2)

        return img.convert('RGBA')

    def _add_printer_artifacts(self, img, intensity=0.02):
        if img.mode != 'L':
            img = img.convert('L').convert('RGBA')

        width, height = img.size
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, 'RGBA')
        dot_count = int(width * height * intensity * 1.08 / 100)
        for _ in range(dot_count):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            size = random.randint(3, 6)
            darkness = random.randint(120, 200)
            gray_value = random.randint(0, 80)
            draw.ellipse([x, y, x + size, y + size], fill=(gray_value, gray_value, gray_value, darkness))
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

        noise = Image.effect_noise((width, height), 10)
        noise = noise.convert('L')
        noise = Image.merge('RGBA', (noise, noise, noise, Image.new('L', (width, height), 10)))

        result = Image.alpha_composite(img, overlay)
        result = Image.alpha_composite(result, noise)
        return result

    def _composite_on_a4(self, img, data=None):
        try:
            a4_width, a4_height = 2480, 3508
            a4_img = Image.new('L', (a4_width, a4_height), 255).convert('RGBA')
            a4_img = self._add_printer_artifacts(a4_img)
            doc_width, doc_height = img.size
            x = (a4_width - doc_width) // 2
            y = (a4_height - doc_height) // 2
            if img.mode != 'RGBA':
                img = img.convert('L').convert('RGBA')
            shadow = Image.new('L', (doc_width + 20, doc_height + 20), 255)
            shadow_draw = ImageDraw.Draw(shadow)
            for i in range(10, 0, -1):
                shadow_draw.rectangle([i, i, doc_width + 20 - i, doc_height + 20 - i],
                                      outline=200, width=1)
            shadow = shadow.convert('RGBA')
            shadow = Image.eval(shadow, lambda x: 0 if x < 255 else 0)
            a4_img.alpha_composite(shadow, (x - 10, y - 10))
            a4_img.paste(img, (x, y), img.split()[3] if img.mode == 'RGBA' else None)
            noise = Image.effect_noise((a4_width, a4_height), 10)
            noise = noise.convert('L')
            noise = Image.merge('RGBA', (noise, noise, noise, Image.new('L', (a4_width, a4_height), 15)))
            a4_img = Image.alpha_composite(a4_img, noise)
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
        if not data:
            return img
        draw = ImageDraw.Draw(img)
        combined_areas = [f for f in self.spec.get('fields', [])
                          if set(f['name'].split(',')).issuperset({'AccountID', 'HealthBenefitID'})]
        account_areas = [f for f in self.spec.get('fields', []) if f['name'] == 'AccountID']
        health_areas = [f for f in self.spec.get('fields', []) if f['name'] == 'HealthBenefitID']
        hw_font_path = random.choice(HANDWRITING_FONTS)
        if combined_areas and ('AccountID' in data or 'HealthBenefitID' in data):
            area = combined_areas[0]
            bbox = area['bbox']
            x = int(bbox[0] * img.width)
            y = int(bbox[1] * img.height)
            w = int(bbox[2] * img.width)
            h = int(bbox[3] * img.height)
            line_height = h // 2
            if 'AccountID' in data and data['AccountID']:
                account_text = str(data['AccountID'])
                font_size = self._get_optimal_font_size(draw, account_text, hw_font_path, w, line_height * 0.9)
                try:
                    hw_font = ImageFont.truetype(str(hw_font_path), font_size)
                    text_bbox = draw.textbbox((0, 0), account_text, font=hw_font)
                    text_h = text_bbox[3] - text_bbox[1]
                    text_y = y + (line_height - text_h) // 2
                    draw.text((x, text_y), account_text, font=hw_font, fill=(80, 80, 80, 220))
                except Exception as e:
                    print(f"Error rendering AccountID: {e}")
            if 'HealthBenefitID' in data and data['HealthBenefitID']:
                health_text = str(data['HealthBenefitID'])
                font_size = self._get_optimal_font_size(draw, health_text, hw_font_path, w, line_height * 0.9)
                try:
                    hw_font = ImageFont.truetype(str(hw_font_path), font_size)
                    text_bbox = draw.textbbox((0, 0), health_text, font=hw_font)
                    text_h = text_bbox[3] - text_bbox[1]
                    text_y = y + line_height + (line_height - text_h) // 2
                    draw.text((x, text_y), health_text, font=hw_font, fill=(80, 80, 80, 220))
                except Exception as e:
                    print(f"Error rendering HealthBenefitID: {e}")
        else:
            if 'AccountID' in data and data['AccountID'] and account_areas:
                area = account_areas[0]
                bbox = area['bbox']
                x = int(bbox[0] * img.width)
                y = int(bbox[1] * img.height)
                w = int(bbox[2] * img.width)
                h = int(bbox[3] * img.height)
                account_text = str(data['AccountID'])
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
                health_text = str(data['HealthBenefitID'])
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

    # place ids on A4
    def _add_random_ids_on_a4(self, a4_img, doc_bbox, data):
        draw = ImageDraw.Draw(a4_img)
        a4_w, a4_h = a4_img.size
        doc_x1, doc_y1, doc_x2, doc_y2 = doc_bbox
        targets = [(k, str(v)) for k, v in (('AccountID', data.get('AccountID')), ('HealthBenefitID', data.get('HealthBenefitID'))) if v]
        if not targets:
            return a4_img
        placed = []
        hw_font_path = random.choice(HANDWRITING_FONTS)
        for _, text in targets:
            size = self._get_optimal_font_size(draw, text, hw_font_path, 250, 120, min_size=12, max_size=48)
            font = ImageFont.truetype(str(hw_font_path), size)
            text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]
            tries = 0
            while tries < 100:
                x = random.randint(50, a4_w - text_w - 50)
                y = random.randint(50, a4_h - text_h - 50)
                box = (x, y, x + text_w, y + text_h)
                if (box[2] < doc_x1 or box[0] > doc_x2 or box[3] < doc_y1 or box[1] > doc_y2) and all(
                    not (box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1]) for b in placed):
                    placed.append(box)
                    draw.text((x, y), text, font=font, fill=(80, 80, 80, 255))
                    break
        return a4_img

    def generate(self, data, output_path=None):
        try:
            img = self.template.copy()
            draw = ImageDraw.Draw(img)
            has_id_fields = any(
                set(f['name'].split(',')).issuperset({'AccountID', 'HealthBenefitID'}) or
                f['name'] in ['AccountID', 'HealthBenefitID']
                for f in self.spec.get('fields', [])
            )
            for field in self.spec.get('fields', []):
                name = field['name']
                if name in ['AccountID', 'HealthBenefitID'] or \
                   set(name.split(',')).issuperset({'AccountID', 'HealthBenefitID'}):
                    continue
                x, y, w, h = field['bbox']
                if w <= 0 or h <= 0:
                    continue
                abs_x = int(x * self.spec['width'])
                abs_y = int(y * self.spec['height'])
                abs_w = int(w * self.spec['width'])
                abs_h = int(h * self.spec['height'])
                field_names = [n.strip() for n in field['name'].split(',')]
                processed_fields = []
                i = 0
                while i < len(field_names):
                    if field_names[i] == 'City' and i + 1 < len(field_names) and field_names[i + 1] == 'State':
                        processed_fields.append(('City,State', i, i + 1))
                        i += 2
                    else:
                        processed_fields.append((field_names[i], i, i))
                        i += 1
                field_values = []
                for field_info in processed_fields:
                    field_name, _, _ = field_info
                    if field_name == 'City,State':
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
                            total_height += (bbox[3] - bbox[1]) * 1.08
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
                                text_height = text_bbox[3] - text_bbox[1]
                                text_x = abs_x + 2
                                text_y = abs_y + y_offset
                                draw.text((text_x, text_y), value, font=font, fill=(0, 0, 0, 255))
                                y_offset += text_height * 1.08
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
                    text_x = abs_x + 5
                    text_y = abs_y + 2
                    draw.text((text_x, text_y), signature_text, font=sig_font, fill=(0, 0, 0, 255))
                except Exception as e:
                    print(f"Failed to render signature: {e}")
            if output_path is None:
                return img
            if has_id_fields:
                img = self._add_printer_artifacts(img)
                img = self._add_handwritten_annotations(img, data)
                if self.quality == 'unclear':
                    img = self._add_noise_effects(img)
                img = img.convert('RGB')
                img.save(output_path, 'PNG', quality=95, dpi=(300, 300))
                return img
            final_img = self._composite_on_a4(img, data)
            a4_w, a4_h = 2480, 3508
            doc_w, doc_h = img.size
            doc_x = (a4_w - doc_w) // 2
            doc_y = (a4_h - doc_h) // 2
            doc_bbox = (doc_x, doc_y, doc_x + doc_w, doc_y + doc_h)
            final_img = self._add_random_ids_on_a4(final_img, doc_bbox, data)
            if self.quality == 'unclear':
                final_img = self._add_noise_effects(final_img)
            final_img.save(output_path, 'PNG', quality=95, dpi=(300, 300))
            return final_img
        except Exception as e:
            print(f"Failed to generate document: {e}")

def generate_batch(generator, data_list, output_dir, count=None):
    if not data_list:
        print("No data provided for document generation")
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    base_dir = Path(__file__).parent
    templates_dir = base_dir / 'templates'
    output_dir = Path(args.output_dir)
    documents_dir = base_dir / 'output' / 'documents'
    pdfs_dir = base_dir / 'output' / 'pdfs'
    documents_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir) if args.output_dir else documents_dir
    if '_' in args.doc_type:
        base_type = args.doc_type.split('_')[-1]
        subtype_list = [args.doc_type]
    else:
        base_type = args.doc_type.lower()
        subtype_list = DOCUMENT_SUBTYPES.get(base_type)
        if not subtype_list:
            print(f"ERROR: Unknown document type '{args.doc_type}'. Available: {list(DOCUMENT_SUBTYPES.keys())}")
            return
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
    max_docs = len(data_rows)
    count = min(args.count, max_docs) if args.count else max_docs
    if args.count and args.count > max_docs:
        print(f"Only {max_docs} rows available, generating {count} documents")
    if len(subtype_list) == 1:
        chosen_subtype = subtype_list[0]
    else:
        chosen_subtype = subtype_list[0] if args.no_random else random.choice(subtype_list)
        print(f"Randomly selected subtype: {chosen_subtype}")
    cleaned_template = base_dir / 'output' / 'clean_templates' / f"{chosen_subtype}_clean.png"
    spec_file = base_dir / 'output' / 'clean_templates' / f"{chosen_subtype}_spec.json"
    if not cleaned_template.exists() or not spec_file.exists():
        print(f"ERROR: Cleaned assets for '{chosen_subtype}' not found. Run 'python clean_template.py {chosen_subtype}' first.")
        return
    generator = DocumentGenerator(cleaned_template, spec_file, None, 12, args.quality)
    generated_files = []
    for i in range(count):
        if not cleaned_template.exists() or not spec_file.exists():
            print(f"Skipping {chosen_subtype}: cleaned assets missing. Run 'python clean_template.py {chosen_subtype}' first.")
            continue
        try:
            output_path = output_dir / f"{chosen_subtype}_{i+1}.png"
            generator.generate(data_rows[i], output_path=output_path)
            generated_files.append(str(output_path))
        except Exception as e:
            print(f"Error generating document {i+1} ({chosen_subtype}): {e}")
    if generated_files:
        print("\nGenerated documents:")
        for idx, fp in enumerate(generated_files, 1):
            print(f"  {idx}. {fp}")
        if args.pdf:
            if args.pdf == 'single':
                print("\nGenerating single-page PDFs…")
                for img_path in generated_files:
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
                        img1.save(pdf_path, 'PDF', resolution=300,
                                  save_all=True, append_images=[Image.open(f).convert('RGB') for f in generated_files[1:]])
                    print(f"  → {pdf_path}")
                except Exception as e:
                    print(f"  Failed to create multi-page PDF: {e}")
    else:
        print("No documents generated due to previous errors.")

if __name__ == "__main__":
    main()
