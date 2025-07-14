import json
import csv
import os
import io
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

FONTS_DIR      = Path(__file__).parent.parent / 'fonts'
DEFAULT_FONT   = FONTS_DIR / 'OpenSans_SemiCondensed-Regular.ttf'
SIGNATURE_FONT = FONTS_DIR / 'signature.ttf'

HANDWRITING_FONTS = [
    FONTS_DIR / 'handwriting.ttf',
    FONTS_DIR / 'handwriting2.ttf',
    FONTS_DIR / 'handwriting3.ttf'
]

DOCUMENT_SUBTYPES = {
    'passport': ['us_passport', 'india_passport'],
    'paystub' : ['adp_paystub', 'paychex_paystub'],
    'ssn'     : ['ssn1_ssn'],
    'empletter': ['empletter1_empletter'], 
    'taxreturn': ['w4_taxreturn'], 
    'i766': ['form_i766']
}

# random-placement tuning
A4_W, A4_H  = 2480, 3508
MARGIN      = 50
MIN_SCALE   = 0.55
MAX_SCALE   = 0.90


class DocumentGenerator:
    def __init__(self, template_path, spec_path, font_path=None,
                 font_size=12, quality='unclear'):
        self.template   = Image.open(template_path).convert('RGBA')
        self.spec       = self._load_spec(spec_path)
        self.font_path  = str(font_path) if font_path else (str(DEFAULT_FONT) if DEFAULT_FONT.exists() else None)
        self.font_size  = font_size
        self.template_stem = Path(template_path).stem
        self.quality    = quality.lower() if quality.lower() in ('clear', 'unclear') else 'unclear'
        self.font_cache = {}

    def _load_spec(self, p):
        with open(p) as f:
            return json.load(f)

    def _get_font(self, key, default_size=None):
        size = default_size or self.font_size
        ck   = f'{key}_{size}'
        if ck not in self.font_cache:
            try:
                self.font_cache[ck] = ImageFont.truetype(self.font_path, size) if self.font_path else ImageFont.load_default()
            except Exception:
                self.font_cache[ck] = ImageFont.load_default()
        return self.font_cache[ck]

    def _add_noise_effects(self, img):
        if self.quality != 'unclear':
            return img.convert('RGBA')

        img   = img.convert('RGBA')
        orig  = img.size
        small = (max(20, int(orig[0] * random.uniform(.3, .5))),
                 max(20, int(orig[1] * random.uniform(.3, .5))))
        img   = img.resize(small, Image.Resampling.LANCZOS)
        large = (int(small[0] * random.uniform(2, 3)),
                 int(small[1] * random.uniform(2, 3)))
        img   = img.resize(large, Image.Resampling.NEAREST)
        img   = img.rotate(random.uniform(-2, 2), Image.BICUBIC, expand=True, fillcolor='white')
        img   = img.resize(orig, Image.Resampling.BILINEAR)

        if random.random() > .3:
            buf = io.BytesIO()
            img.convert('RGB').save(buf, 'JPEG', quality=random.randint(30, 60))
            img = Image.open(buf).convert('RGBA')

        if random.random() > .3:
            noise = np.random.randint(0, 30, (img.size[1], img.size[0], 3), dtype=np.uint8)
            img   = Image.blend(img, Image.fromarray(noise).convert('RGBA'), .05)

        if random.random() > .3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.2, 3)))

        if random.random() > .6:
            img  = ImageEnhance.Brightness(img).enhance(random.uniform(.95, 1.05))
            img  = ImageEnhance.Contrast(img).enhance(random.uniform(.97, 1.03))

        if random.random() > .2:
            L = np.array(img.convert('L'))
            g = np.random.normal(0, random.uniform(.5, 1.5), L.shape)
            img = Image.blend(
                img.convert('L'),
                Image.fromarray(np.clip(L + g, 0, 255).astype('uint8'), 'L'),
                .2
            )

        return img.convert('RGBA')

    def _fast_printer_artifacts(self, size, intensity=.02):
        w, h = size
        m = np.zeros((h, w), np.uint8)

        # dots
        n = int(w * h * intensity * 1.08 / 100)
        m[np.random.randint(0, h, n), np.random.randint(0, w, n)] = np.random.randint(120, 200, n)

        # vertical streaks
        for _ in range(random.randint(1, 5)):
            sw  = np.random.randint(1, 3)           # 1 or 2 pixels wide
            x0  = np.random.randint(0, w - sw + 1)  # keep streak fully inside the image
            m[:, x0:x0 + sw] = np.where(
                np.random.rand(h, sw) < .7,
                np.random.randint(20, 80, (h, sw)),
                m[:, x0:x0 + sw]
            )

        # horizontal streak 
        if random.random() < .3:
            sh  = np.random.randint(1, 3)           # 1 or 2 rows high
            y0  = np.random.randint(0, h - sh + 1)
            m[y0:y0 + sh, :] = np.where(
                np.random.rand(sh, w) < .6,
                np.random.randint(15, 60, (sh, w)),
        m[y0:y0 + sh, :]
    )


        alpha = (m > 0).astype(np.uint8) * 255
        return Image.fromarray(np.dstack([m] * 3 + [alpha]), 'RGBA')

    def _add_printer_artifacts(self, img, intensity=.02):
        base  = img.convert('L').convert('RGBA')
        arte  = self._fast_printer_artifacts(base.size, intensity)
        noise = Image.effect_noise(base.size, 10).convert('L')
        noise = Image.merge('RGBA', (noise, noise, noise, Image.new('L', base.size, 10)))
        return Image.alpha_composite(Image.alpha_composite(base, arte), noise)

    def _get_optimal_font_size(self, draw, text, font_path,
                               max_w, max_h, min_s=6, max_s=100):
        if not text or not font_path.exists():
            return min_s
        lo, hi, best = min_s, max_s, min_s
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                f   = ImageFont.truetype(str(font_path), mid)
                bbox = draw.textbbox((0, 0), text, font=f)
                if bbox[2] - bbox[0] <= max_w and bbox[3] - bbox[1] <= max_h:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            except Exception:
                hi = mid - 1
        return best

    def _find_random_position(self, bx, by, bw, bh, tw, th, placed, margin=4, attempts=300):
        for _ in range(attempts):
            x = random.randint(bx, bx + bw - tw)
            y = random.randint(by, by + bh - th)
            nb = (x - margin, y - margin, x + tw + margin, y + th + margin)
            if all(not (nb[0] < p[2] and nb[2] > p[0] and
                        nb[1] < p[3] and nb[3] > p[1]) for p in placed):
                placed.append(nb)
                return x, y
        return None, None
        for _ in range(attempts):
            x = random.randint(bx, bx + bw - tw)
            y = random.randint(by, by + bh - th)
            nb = (x - margin, y - margin, x + tw + margin, y + th + margin)
            if all(not (nb[0] < p[2] and nb[2] > p[0] and
                        nb[1] < p[3] and nb[3] > p[1]) for p in placed):
                placed.append(nb)
                return x, y
        return None, None

    def _draw_id_in_box(self, draw, box_px, txt, hw_font_path, placed):
        bx, by, bw, bh = box_px
        max_fs = self._get_optimal_font_size(draw, txt, hw_font_path, bw, bh)
        max_fs = min(max_fs, int(bh * 0.15))  # <= 40 % box height
        if max_fs < 6:
            return 
        fs = int(max_fs * random.uniform(0.45, 0.65))
        f  = ImageFont.truetype(str(hw_font_path), fs)
        tw, th = draw.textbbox((0, 0), txt, font=f)[2:]
        while (tw > bw or th > bh) and fs > 6:
            fs = max(6, int(fs * 0.9))
            f  = ImageFont.truetype(str(hw_font_path), fs)
            tw, th = draw.textbbox((0, 0), txt, font=f)[2:]

        x, y = self._find_random_position(bx, by, bw, bh, tw, th, placed)
        if x is None:
            # could not find a non-overlapping spot – shrink & centre
            fs = max(6, int(fs * 0.8))
            f  = ImageFont.truetype(str(hw_font_path), fs)
            tw, th = draw.textbbox((0, 0), txt, font=f)[2:]
            x = bx + (bw - tw) // 2
            y = by + (bh - th) // 2
            placed.append((x, y, x + tw, y + th))
        draw.text((x, y), txt, font=f, fill=(80, 80, 80, 220))

    def _split_box(self, box_px, orientation='auto'):
        bx, by, bw, bh = box_px
        if orientation == 'auto':
            orientation = 'h' if bw >= bh else 'v'
        if orientation == 'h':  # horizontal split (left / right)
            mid = bx + bw // 2
            return (
                (bx,      by, mid - bx, bh),
                (mid,     by, bx + bw - mid, bh)
            )
        else:  # vertical split (top / bottom)
            mid = by + bh // 2
            return (
                (bx, by, bw, mid - by),
                (bx, mid, bw, by + bh - mid)
            )

    def _add_handwritten_annotations(self, img, data):
        draw = ImageDraw.Draw(img)
        combined_areas = [f for f in self.spec.get('fields', [])
                          if set(f['name'].split(',')).issuperset({'AccountID', 'HealthBenefitID'})]
        account_areas  = [f for f in self.spec.get('fields', []) if f['name'] == 'AccountID']
        health_areas   = [f for f in self.spec.get('fields', []) if f['name'] == 'HealthBenefitID']
        hw_font_path   = random.choice(HANDWRITING_FONTS)

        placed = []  # track all bounding boxes across boxes

        # combined label box
        if combined_areas and ('AccountID' in data or 'HealthBenefitID' in data):
            area = combined_areas[0]
            bx, by, bw, bh = area['bbox']
            box_px = (int(bx * img.width),
                      int(by * img.height),
                      int(bw * img.width),
                      int(bh * img.height))
            # split orientation auto
            boxA, boxB = self._split_box(box_px)
            id_pairs = [('AccountID', boxA), ('HealthBenefitID', boxB)]
            random.shuffle(id_pairs)  # randomise who gets top/left vs bottom/right
            for key, sub_box in id_pairs:
                if key in data and data[key]:
                    self._draw_id_in_box(
                        draw, sub_box, str(data[key]), hw_font_path, placed)
            return img

        # separate label boxes
        field_map = (
            ('AccountID', account_areas),
            ('HealthBenefitID', health_areas)
        )
        for key, areas in field_map:
            if key not in data or not data[key] or not areas:
                continue
            area = areas[0]
            bx, by, bw, bh = area['bbox']
            box_px = (int(bx * img.width),
                      int(by * img.height),
                      int(bw * img.width),
                      int(bh * img.height))
            self._draw_id_in_box(
                draw, box_px, str(data[key]), hw_font_path, placed)

        return img

    # add random IDs directly on A4

    def _add_random_ids_on_a4(self, a4_img, doc_bbox, data):
        draw = ImageDraw.Draw(a4_img)
        a4_w, a4_h = a4_img.size
        d1, d2, d3, d4 = doc_bbox
        pairs = [('AccountID', data.get('AccountID')),
                 ('HealthBenefitID', data.get('HealthBenefitID'))]
        pairs = [(k, str(v)) for k, v in pairs if v]
        if not pairs:
            return a4_img
        placed = []
        hw_font_path = random.choice(HANDWRITING_FONTS)
        for _, txt in pairs:
            fs = self._get_optimal_font_size(draw, txt, hw_font_path,250, 120, 12, 28)
            fs = int(fs * random.uniform(0.45, 0.70))
            f  = ImageFont.truetype(str(hw_font_path), fs)
            tw, th = draw.textbbox((0, 0), txt, font=f)[2:]
            for _ in range(100):
                x = random.randint(50, a4_w - tw - 50)
                y = random.randint(50, a4_h - th - 50)
                box = (x - 4, y - 4, x + tw + 4, y + th + 4)
                if (box[2] < d1 or box[0] > d3 or box[3] < d2 or box[1] > d4) and \
                   all(not (box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1]) for b in placed):
                    placed.append(box)
                    draw.text((x, y), txt, font=f, fill=(80, 80, 80, 255))
                    break
        return a4_img

    # random placement on A4

    def _place_doc_on_a4(self, doc_img):
        a4 = Image.new('L', (A4_W, A4_H), 255).convert('RGBA')
        a4 = self._add_printer_artifacts(a4)

        # pick a random target size and then enlarge by 30%
        max_w = A4_W - 2 * MARGIN
        max_h = A4_H - 2 * MARGIN
        scale_hi = min(max_w / doc_img.width, max_h / doc_img.height, MAX_SCALE)
        scale_lo = max(MIN_SCALE, min(scale_hi * 0.5, MIN_SCALE))
        scale    = random.uniform(scale_lo, scale_hi)

        # enlarge by 30 % but keep within limits
        scale = min(scale * 1.3, max_w / doc_img.width, max_h / doc_img.height, MAX_SCALE)

        new_w = int(doc_img.width * scale)
        new_h = int(doc_img.height * scale)
        doc   = doc_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # random top-left
        x0 = random.randint(MARGIN, A4_W - new_w - MARGIN)
        y0 = random.randint(MARGIN, A4_H - new_h - MARGIN)

        a4.paste(doc.convert('RGBA'), (x0, y0), doc)

        bbox = (x0, y0, x0 + new_w, y0 + new_h)
        return a4.convert('L').convert('RGB'), bbox

    # main generator

    def generate(self, data, output_path=None):
        img  = self.template.copy()
        draw = ImageDraw.Draw(img)

        # render template fields (unchanged)
        for field in self.spec.get('fields', []):
            name = field['name']
            if name in ('AccountID', 'HealthBenefitID') or \
               set(name.split(',')).issuperset({'AccountID', 'HealthBenefitID'}):
                continue
            x, y, w, h = field['bbox']
            if w <= 0 or h <= 0:
                continue
            abs_x = int(x * self.spec['width'])
            abs_y = int(y * self.spec['height'])
            abs_w = int(w * self.spec['width'])
            abs_h = int(h * self.spec['height'])

            field_names = [n.strip() for n in name.split(',')]
            processed = []
            i = 0
            while i < len(field_names):
                if (field_names[i] == 'City' and i + 1 < len(field_names)
                        and field_names[i + 1] == 'State'):
                    processed.append(('City,State', i, i + 1))
                    i += 2
                else:
                    processed.append((field_names[i], i, i))
                    i += 1
            values = []
            for fn, _, _ in processed:
                if fn == 'City,State':
                    city  = str(data.get('City', '')).strip()
                    state = str(data.get('State', '')).strip()
                    if city and state:
                        values.append(f"{city}, {state}")
                    elif city or state:
                        values.append(city or state)
                values = []
            for fn, _, _ in processed:
                if fn == 'City,State':
                    city  = str(data.get('City',  '')).strip()
                    state = str(data.get('State', '')).strip()
                    if city and state:
                        values.append(f"{city}, {state}")
                    elif city or state:
                        values.append(city or state)

                # address
                elif fn == 'Address':                       
                    parts = [
                        data.get('Street1', '').strip(),
                        data.get('Street2', '').strip(),
                        data.get('City',    '').strip(),
                        data.get('State',   '').strip(),
                        data.get('Zip',     '').strip()
                    ]
                    addr_line = " ".join([p for p in parts if p])  # drop blanks, join with spaces
                    if addr_line:
                        values.append(addr_line)

                else:
                    v = str(data.get(fn, '')).strip()
                    if v:
                        values.append(v)

            if not values:
                continue

            best = 0
            for sz in range(12, 73):
                f   = self._get_font(name, sz)
                t_h = 0
                t_w = 0
                for v in values:
                    bb  = draw.textbbox((0, 0), v, font=f)
                    t_h += (bb[3] - bb[1]) * 1.08
                    t_w  = max(t_w, bb[2] - bb[0])
                if t_h <= abs_h * .9 and t_w <= abs_w * .9:
                    best = sz
                else:
                    break
            f = self._get_font(name, best)
            y_off = 0
            for v in values:
                bb = draw.textbbox((0, 0), v, font=f)
                draw.text((abs_x + 2, abs_y + y_off), v, font=f,
                          fill=(0, 0, 0, 255))
                y_off += (bb[3] - bb[1]) * 1.08

        # handwritten signature
        sig_field = next((f for f in self.spec['fields']
                          if f['name'].lower() == 'signature'), None)
        if sig_field and 'FullName' in data:
            x, y, w, h = sig_field['bbox']
            ax, ay, aw, ah = (int(x * self.spec['width']),
                              int(y * self.spec['height']),
                              int(w * self.spec['width']),
                              int(h * self.spec['height']))
            text = data['FullName']
            low, high, best = 6, min(72, ah), 6
            while low <= high:
                mid = (low + high) // 2
                try:
                    f = ImageFont.truetype(str(SIGNATURE_FONT), mid)
                    if draw.textlength(text, font=f) <= aw * .9:
                        best, low = mid, mid + 1
                    else:
                        high = mid - 1
                except Exception:
                    high = mid - 1
            f = ImageFont.truetype(str(SIGNATURE_FONT), best)
            draw.text((ax + 5, ay + 2), text, font=f,
                      fill=(0, 0, 0, 255))

        # IDs & final placement
        has_ids = any(set(f['name'].split(',')).issuperset({'AccountID', 'HealthBenefitID'})
                      or f['name'] in ('AccountID', 'HealthBenefitID')
                      for f in self.spec.get('fields', []))

        if has_ids:
            img = self._add_printer_artifacts(img)
            img = self._add_handwritten_annotations(img, data)
            if self.quality == 'unclear':
                img = self._add_noise_effects(img)
            if output_path:
                img.convert('RGB').save(output_path, 'PNG', quality=95, dpi=(300, 300))
            return img

        final, bbox = self._place_doc_on_a4(img)
        final       = self._add_random_ids_on_a4(final, bbox, data)
        if self.quality == 'unclear':
            final = self._add_noise_effects(final)
        if output_path:
            final.save(output_path, 'PNG', quality=95, dpi=(300, 300))
        return final

# batch / parallel helpers
def _make_output_stem(stem: str, quality: str) -> str:
    """Return the base filename stem adjusted for quality.

    If quality == 'unclear' we replace any trailing '_clean' with
    '_unclean'. Otherwise we leave the stem untouched.
    """
    if quality == 'unclear':
        return stem[:-6] + '_unclean' if stem.endswith('_clean') else f"{stem}_unclean"
    return stem

def _render_worker(task):
    tpl, spec, qual, row, out = task
    gen = DocumentGenerator(tpl, spec, None, 12, qual)
    gen.generate(row, output_path=out)
    return str(out)

def generate_batch_parallel(generator, tpl, spec, qual,
                            rows, out_dir, count, workers):
    if not workers or workers < 2:
        return generate_batch(generator, rows, out_dir, count)
    tasks = []
    for i in range(count):
        out = Path(out_dir) / f"{_make_output_stem(tpl.stem, qual)}_{i + 1}.png"
        tasks.append((tpl, spec, qual, rows[i], out))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_render_worker, tasks))

def generate_batch(generator, rows, out_dir, count=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    count = min(count, len(rows)) if count else len(rows)
    files = []
    for i in range(count):
        out = Path(out_dir) / f"{_make_output_stem(generator.template_stem, generator.quality)}{i + 1 if count>1 else ''}.png"
        generator.generate(rows[i], output_path=out)
        files.append(str(out))
    return files

# CLI

def main():
    p = argparse.ArgumentParser(description='Generate documents')
    p.add_argument('doc_type')
    p.add_argument('--subtype')
    p.add_argument('--no-random', action='store_true')
    p.add_argument('--data', default='data/sample_data.csv')
    p.add_argument('--output-dir', default='output/documents')
    p.add_argument('--row', type=int)
    p.add_argument('--count', type=int)
    p.add_argument('--pdf', choices=['single', 'multi'])
    p.add_argument('--quality', choices=['clear', 'unclear'], default='unclear')
    p.add_argument('--workers', type=int,
                   help='Parallel workers (default serial)')
    args = p.parse_args()

    base    = Path(__file__).parent
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # subtype resolution
    if '_' in args.doc_type:
        subtype = args.doc_type
    else:
        cand = DOCUMENT_SUBTYPES.get(args.doc_type.lower())
        if not cand:
            print('Unknown doc_type')
            return
        subtype = args.subtype if args.subtype else (
            cand[0] if args.no_random else random.choice(cand))

    tpl  = base / 'output/clean_templates' / f'{subtype}_clean.png'
    spec = base / 'output/clean_templates' / f'{subtype}_spec.json'
    if not tpl.exists() or not spec.exists():
        print('Run clean_template first')
        return

    rows = list(csv.DictReader(open(args.data)))
    if not rows:
        print('No CSV rows')
        return
    if args.row is not None:
        rows = [rows[args.row]]
    count = min(args.count, len(rows)) if args.count else len(rows)

    gen   = DocumentGenerator(tpl, spec, None, 12, args.quality)
    paths = generate_batch_parallel(gen, tpl, spec, args.quality,
                                    rows, out_dir, count, args.workers)

    print('\nGenerated documents:')
    for i, pth in enumerate(paths, 1):
        print(f'  {i}. {pth}')

    if args.pdf:
        pdf_dir = base / 'output/pdfs'
        pdf_dir.mkdir(parents=True, exist_ok=True)
        if args.pdf == 'single':
            for pth in paths:
                pdf = pdf_dir / (Path(pth).stem + '.pdf')
                Image.open(pth).convert('RGB').save(pdf, 'PDF', resolution=300)
                print('→', pdf)
        else:
            pdf = pdf_dir / f'{subtype}_batch.pdf'
            imgs = [Image.open(p).convert('RGB') for p in paths]
            imgs[0].save(pdf, 'PDF', resolution=300,
                         save_all=True, append_images=imgs[1:])
            print('→', pdf)

if __name__ == '__main__':
    main()
