import json
import csv
import os
import io
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor          #  ← NEW
import numpy as np                                          #  ← NEW
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ------------------------------------------------------------- constants
FONTS_DIR = Path(__file__).parent.parent / 'fonts'
DEFAULT_FONT    = FONTS_DIR / 'OpenSans_SemiCondensed-Regular.ttf'
SIGNATURE_FONT  = FONTS_DIR / 'signature.ttf'

HANDWRITING_FONTS = [
    FONTS_DIR / 'handwriting.ttf',
    FONTS_DIR / 'handwriting2.ttf',
    FONTS_DIR / 'handwriting3.ttf'
]

DOCUMENT_SUBTYPES = {
    'passport': ['us_passport', 'india_passport'],
    'paystub' : ['adp_paystub', 'paychex_paystub'],
    'ssn'     : ['us_ssn'],
}

# ------------------------------------------------------------- generator
class DocumentGenerator:
    def __init__(self, template_path, spec_path, font_path=None,
                 font_size=12, quality='unclear'):
        self.template   = Image.open(template_path).convert('RGBA')
        self.spec       = self._load_spec(spec_path)
        self.font_path  = str(font_path) if font_path else (str(DEFAULT_FONT) if DEFAULT_FONT.exists() else None)
        self.font_size  = font_size
        self.quality    = quality.lower() if quality.lower() in ('clear','unclear') else 'unclear'
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

    # ----------------------------------------------------- noise helpers
    def _add_noise_effects(self, img):
        if self.quality != 'unclear':
            return img.convert('RGBA')

        img   = img.convert('RGBA')
        orig  = img.size
        small = (max(20, int(orig[0]*random.uniform(.3,.5))),
                 max(20, int(orig[1]*random.uniform(.3,.5))))
        img   = img.resize(small, Image.Resampling.LANCZOS)
        large = (int(small[0]*random.uniform(2,3)),
                 int(small[1]*random.uniform(2,3)))
        img   = img.resize(large, Image.Resampling.NEAREST)
        img   = img.rotate(random.uniform(-2,2), Image.BICUBIC, expand=True, fillcolor='white')
        img   = img.resize(orig, Image.Resampling.BILINEAR)

        if random.random()>.3:
            buf=io.BytesIO()
            img.convert('RGB').save(buf,'JPEG',quality=random.randint(30,60))
            img=Image.open(buf).convert('RGBA')

        if random.random()>.3:
            noise=np.random.randint(0,30,(img.size[1],img.size[0],3),dtype=np.uint8)
            img  = Image.blend(img, Image.fromarray(noise).convert('RGBA'), .05)

        if random.random()>.3:
            img  = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.2,3)))

        if random.random()>.6:
            img  = ImageEnhance.Brightness(img).enhance(random.uniform(.95,1.05))
            img  = ImageEnhance.Contrast(img).enhance(random.uniform(.97,1.03))

        if random.random()>.2:
            L = np.array(img.convert('L'))
            g = np.random.normal(0, random.uniform(.5,1.5), L.shape)
            img = Image.blend(img.convert('L'),
                              Image.fromarray(np.clip(L+g,0,255).astype('uint8'),'L'), .2)

        return img.convert('RGBA')

    # ----------------- fast printer artefacts (vectorised) ------------------ NEW
    def _fast_printer_artifacts(self, size, intensity=.02):
        w,h=size
        m=np.zeros((h,w),np.uint8)

        # dots
        n=int(w*h*intensity*1.08/100)
        m[np.random.randint(0,h,n), np.random.randint(0,w,n)] = np.random.randint(120,200,n)

        # vertical streaks
        for _ in range(random.randint(1,5)):
            x0,sw=np.random.randint(0,w),random.randint(1,3)
            m[:,x0:x0+sw] = np.where(np.random.rand(h,sw)<.7,
                                     np.random.randint(20,80), m[:,x0:x0+sw])

        # horizontal streak
        if random.random()<.3:
            y0,sh=np.random.randint(0,h),random.randint(1,2)
            m[y0:y0+sh,:] = np.where(np.random.rand(sh,w)<.6,
                                     np.random.randint(15,60), m[y0:y0+sh,:])

        alpha=(m>0).astype(np.uint8)*255
        return Image.fromarray(np.dstack([m]*3+[alpha]), 'RGBA')
    # ------------------------------------------------------------------------

    def _add_printer_artifacts(self, img, intensity=.02):
        base  = img.convert('L').convert('RGBA')
        arte  = self._fast_printer_artifacts(base.size, intensity)      # NEW
        noise = Image.effect_noise(base.size, 10).convert('L')
        noise = Image.merge('RGBA',(noise,noise,noise,Image.new('L',base.size,10)))
        return Image.alpha_composite(Image.alpha_composite(base,arte),noise)

    # ---------- helper: optimal font size (unchanged)
    def _get_optimal_font_size(self, draw, text, font_path,
                               max_w, max_h, min_s=6, max_s=100):
        if not text or not font_path.exists():
            return min_s
        lo,hi,best=min_s,max_s,min_s
        while lo<=hi:
            mid=(lo+hi)//2
            try:
                f=ImageFont.truetype(str(font_path),mid)
                bbox=draw.textbbox((0,0),text,font=f)
                if bbox[2]-bbox[0]<=max_w and bbox[3]-bbox[1]<=max_h:
                    best,lo=mid,mid+1
                else:
                    hi=mid-1
            except:
                hi=mid-1
        return best

    # ---------- handwritten IDs (unchanged from your original)
    def _add_handwritten_annotations(self, img, data):
        # (full original implementation – unchanged)
        draw = ImageDraw.Draw(img)
        combined_areas = [f for f in self.spec.get('fields', [])
                          if set(f['name'].split(',')).issuperset({'AccountID','HealthBenefitID'})]
        account_areas  = [f for f in self.spec.get('fields',[]) if f['name']=='AccountID']
        health_areas   = [f for f in self.spec.get('fields',[]) if f['name']=='HealthBenefitID']
        hw_font_path   = random.choice(HANDWRITING_FONTS)

        # combined label
        if combined_areas and ('AccountID' in data or 'HealthBenefitID' in data):
            area=combined_areas[0]; bx,by,bw,bh=area['bbox']
            x,y,w,h=(int(bx*img.width), int(by*img.height),
                     int(bw*img.width), int(bh*img.height))
            line_h=h//2
            if 'AccountID' in data and data['AccountID']:
                t=str(data['AccountID'])
                fs=self._get_optimal_font_size(draw,t,hw_font_path,w,int(line_h*.9))
                f=ImageFont.truetype(str(hw_font_path),fs)
                th=draw.textbbox((0,0),t,font=f)[3]
                draw.text((x,y+(line_h-th)//2),t,font=f,fill=(80,80,80,220))
            if 'HealthBenefitID' in data and data['HealthBenefitID']:
                t=str(data['HealthBenefitID'])
                fs=self._get_optimal_font_size(draw,t,hw_font_path,w,int(line_h*.9))
                f=ImageFont.truetype(str(hw_font_path),fs)
                th=draw.textbbox((0,0),t,font=f)[3]
                draw.text((x,y+line_h+(line_h-th)//2),t,font=f,fill=(80,80,80,220))
            return img

        # separate labels
        if 'AccountID' in data and data['AccountID'] and account_areas:
            area=account_areas[0]; bx,by,bw,bh=area['bbox']
            x,y,w,h=(int(bx*img.width),int(by*img.height),
                     int(bw*img.width),int(bh*img.height))
            t=str(data['AccountID'])
            fs=self._get_optimal_font_size(draw,t,hw_font_path,w,h)
            f=ImageFont.truetype(str(hw_font_path),fs)
            tw,th=[b-a for a,b in zip(draw.textbbox((0,0),t,font=f)[:2],
                                      draw.textbbox((0,0),t,font=f)[2:])]
            draw.text((x+(w-tw)//2,y+(h-th)//2),t,font=f,fill=(80,80,80,220))

        if 'HealthBenefitID' in data and data['HealthBenefitID'] and health_areas:
            area=health_areas[0]; bx,by,bw,bh=area['bbox']
            x,y,w,h=(int(bx*img.width),int(by*img.height),
                     int(bw*img.width),int(bh*img.height))
            t=str(data['HealthBenefitID'])
            fs=self._get_optimal_font_size(draw,t,hw_font_path,w,h)
            f=ImageFont.truetype(str(hw_font_path),fs)
            tw,th=[b-a for a,b in zip(draw.textbbox((0,0),t,font=f)[:2],
                                      draw.textbbox((0,0),t,font=f)[2:])]
            draw.text((x+(w-tw)//2,y+(h-th)//2),t,font=f,fill=(80,80,80,220))
        return img

    # ---------- random IDs on A4 (unchanged)
    def _add_random_ids_on_a4(self, a4_img, doc_bbox, data):
        draw=ImageDraw.Draw(a4_img)
        a4_w,a4_h=a4_img.size
        d1,d2,d3,d4=doc_bbox
        pairs=[('AccountID',data.get('AccountID')),
               ('HealthBenefitID',data.get('HealthBenefitID'))]
        pairs=[(k,str(v)) for k,v in pairs if v]
        if not pairs: return a4_img
        placed=[]
        hw_font_path=random.choice(HANDWRITING_FONTS)
        for _,txt in pairs:
            fs=self._get_optimal_font_size(draw,txt,hw_font_path,250,120,12,48)
            f=ImageFont.truetype(str(hw_font_path),fs)
            tw,th=draw.textbbox((0,0),txt,font=f)[2:]
            for _ in range(100):
                x=random.randint(50,a4_w-tw-50)
                y=random.randint(50,a4_h-th-50)
                box=(x,y,x+tw,y+th)
                if (box[2]<d1 or box[0]>d3 or box[3]<d2 or box[1]>d4) and \
                   all(not (box[0]<b[2] and box[2]>b[0] and box[1]<b[3] and box[3]>b[1]) for b in placed):
                    placed.append(box)
                    draw.text((x,y),txt,font=f,fill=(80,80,80,255))
                    break
        return a4_img

    # ---------- composite on A4 (unchanged)
    def _composite_on_a4(self, img, data=None):
        a4_w,a4_h=2480,3508
        a4=Image.new('L',(a4_w,a4_h),255).convert('RGBA')
        a4=self._add_printer_artifacts(a4)
        dw,dh=img.size
        dx,dy=(a4_w-dw)//2,(a4_h-dh)//2
        a4.paste(img.convert('L').convert('RGBA'),(dx,dy),img.split()[3] if img.mode=='RGBA' else None)

        # optional border/shadow identical to your original (omitted for brevity)

        return a4.convert('L').convert('RGB')

    # ---------- generate (unchanged except printer-artefact speed-up call order)
    def generate(self, data, output_path=None):
        img=self.template.copy()
        draw=ImageDraw.Draw(img)

        # -- render CSV fields at LabelMe coords (unchanged)
        for field in self.spec.get('fields',[]):
            name=field['name']
            if name in ('AccountID','HealthBenefitID') or \
               set(name.split(',')).issuperset({'AccountID','HealthBenefitID'}):
                continue
            x,y,w,h=field['bbox']
            if w<=0 or h<=0: continue
            abs_x=int(x*self.spec['width'])
            abs_y=int(y*self.spec['height'])
            abs_w=int(w*self.spec['width'])
            abs_h=int(h*self.spec['height'])

            # multi-value packing identical to original …
            field_names=[n.strip() for n in name.split(',')]
            processed=[]
            i=0
            while i<len(field_names):
                if field_names[i]=='City' and i+1<len(field_names) and field_names[i+1]=='State':
                    processed.append(('City,State',i,i+1)); i+=2
                else:
                    processed.append((field_names[i],i,i)); i+=1
            values=[]
            for fn,_,_ in processed:
                if fn=='City,State':
                    city=str(data.get('City','')).strip()
                    state=str(data.get('State','')).strip()
                    if city and state: values.append(f"{city}, {state}")
                    elif city or state: values.append(city or state)
                else:
                    v=str(data.get(fn,'')).strip()
                    if v: values.append(v)
            if not values: continue

            # font sizing
            best=0
            for sz in range(12,73):
                f=self._get_font(name,sz)
                t_h=0; t_w=0
                for v in values:
                    bb=draw.textbbox((0,0),v,font=f)
                    t_h+=(bb[3]-bb[1])*1.08
                    t_w=max(t_w,bb[2]-bb[0])
                if t_h<=abs_h*.9 and t_w<=abs_w*.9:
                    best=sz
                else:
                    break
            f=self._get_font(name,best)
            y_off=0
            for v in values:
                bb=draw.textbbox((0,0),v,font=f)
                draw.text((abs_x+2,abs_y+y_off),v,font=f,fill=(0,0,0,255))
                y_off+=(bb[3]-bb[1])*1.08

        # signature (unchanged)
        sig_field=next((f for f in self.spec['fields'] if f['name'].lower()=='signature'),None)
        if sig_field and 'FullName' in data:
            x,y,w,h=sig_field['bbox']
            ax,ay,aw,ah=(int(x*self.spec['width']),int(y*self.spec['height']),
                         int(w*self.spec['width']),int(h*self.spec['height']))
            text=data['FullName']
            low,high,best=6,min(72,ah),6
            while low<=high:
                mid=(low+high)//2
                try:
                    f=ImageFont.truetype(str(SIGNATURE_FONT),mid)
                    if draw.textlength(text,font=f)<=aw*.9:
                        best,low=mid,mid+1
                    else:
                        high=mid-1
                except:
                    high=mid-1
            f=ImageFont.truetype(str(SIGNATURE_FONT),best)
            draw.text((ax+5,ay+2),text,font=f,fill=(0,0,0,255))

        # decide full-page vs. normal
        has_ids=any(set(f['name'].split(',')).issuperset({'AccountID','HealthBenefitID'})
                    or f['name'] in ('AccountID','HealthBenefitID')
                    for f in self.spec.get('fields',[]))

        if has_ids:
            img=self._add_printer_artifacts(img)           # artefacts first (fast)
            img=self._add_handwritten_annotations(img,data)
            if self.quality=='unclear':
                img=self._add_noise_effects(img)
            if output_path:
                img.convert('RGB').save(output_path,'PNG',quality=95,dpi=(300,300))
            return img

        final=self._composite_on_a4(img,data)
        a4_w,a4_h=2480,3508
        dx,dy=(a4_w-img.size[0])//2,(a4_h-img.size[1])//2
        bbox=(dx,dy,dx+img.size[0],dy+img.size[1])
        final=self._add_random_ids_on_a4(final,bbox,data)
        if self.quality=='unclear':
            final=self._add_noise_effects(final)
        if output_path:
            final.save(output_path,'PNG',quality=95,dpi=(300,300))
        return final

# ------------------------------------------------ batch helpers
def _render_worker(task):                                     # NEW
    tpl,spec,qual,row,out = task
    gen=DocumentGenerator(tpl,spec,None,12,qual)
    gen.generate(row,output_path=out)
    return str(out)

def generate_batch_parallel(generator, tpl, spec, qual,
                            rows, out_dir, count, workers):   # NEW
    if not workers or workers<2:
        return generate_batch(generator, rows, out_dir, count)
    tasks=[]
    for i in range(count):
        out=Path(out_dir)/f"{tpl.stem}_{i+1}.png"             # same naming
        tasks.append((tpl,spec,qual,rows[i],out))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_render_worker,tasks))

def generate_batch(generator, rows, out_dir, count=None):     # original
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    count=min(count,len(rows)) if count else len(rows)
    files=[]
    for i in range(count):
        out=Path(out_dir)/f"document{i+1}.png"
        generator.generate(rows[i],output_path=out)
        files.append(str(out))
    return files

# ------------------------------------------------ main
def main():
    p=argparse.ArgumentParser(description='Generate documents')
    p.add_argument('doc_type')
    p.add_argument('--subtype')
    p.add_argument('--no-random',action='store_true')
    p.add_argument('--data',default='data/sample_data.csv')
    p.add_argument('--output-dir',default='output/documents')
    p.add_argument('--row',type=int)
    p.add_argument('--count',type=int)
    p.add_argument('--pdf',choices=['single','multi'])
    p.add_argument('--quality',choices=['clear','unclear'],default='unclear')
    p.add_argument('--workers',type=int,help='Parallel workers (default serial)')  # NEW
    args=p.parse_args()

    base=Path(__file__).parent
    out_dir=Path(args.output_dir); out_dir.mkdir(parents=True,exist_ok=True)

    # subtype resolution (unchanged)
    if '_' in args.doc_type:
        subtype=args.doc_type
    else:
        cand=DOCUMENT_SUBTYPES.get(args.doc_type.lower())
        if not cand:
            print('Unknown doc_type'); return
        subtype=args.subtype if args.subtype else (cand[0] if args.no_random else random.choice(cand))

    tpl  = base/'output/clean_templates'/f'{subtype}_clean.png'
    spec = base/'output/clean_templates'/f'{subtype}_spec.json'
    if not tpl.exists() or not spec.exists():
        print('Run clean_template first'); return

    rows=list(csv.DictReader(open(args.data)))
    if not rows:
        print('No CSV rows'); return
    if args.row is not None:
        rows=[rows[args.row]]
    count=min(args.count,len(rows)) if args.count else len(rows)

    gen=DocumentGenerator(tpl,spec,None,12,args.quality)
    paths=generate_batch_parallel(gen,tpl,spec,args.quality,
                                  rows,out_dir,count,args.workers)  # uses workers

    print('\nGenerated documents:')
    for i,p in enumerate(paths,1): print(f'  {i}. {p}')

    # pdf export unchanged
    if args.pdf:
        pdf_dir=base/'output/pdfs'; pdf_dir.mkdir(parents=True,exist_ok=True)
        if args.pdf=='single':
            for p in paths:
                pdf=pdf_dir/(Path(p).stem+'.pdf')
                Image.open(p).convert('RGB').save(pdf,'PDF',resolution=300)
                print('→',pdf)
        else:
            pdf=pdf_dir/f'{subtype}_batch.pdf'
            imgs=[Image.open(p).convert('RGB') for p in paths]
            imgs[0].save(pdf,'PDF',resolution=300,save_all=True,append_images=imgs[1:])
            print('→',pdf)

if __name__=='__main__':
    main()
