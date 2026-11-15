"""Extract receipt items using a trained OCR-line classifier and train-only catalog."""
import argparse, json, re, statistics
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
import joblib
from rapidfuzz import process, fuzz
from analyze_paddle_ocr import alphabetic
from ocr_grouping import line_groups
from receipt_common import ML, normalize_name, normalize_sum
MONEY = re.compile(r'(?<!\d)(\d{1,6}\s*[.,]\s*\d{2})(?!\d|\s*(?:мг|кг|мл|mg)\b)', re.I)
from train_ocr_lines import rows

MODEL=ML/'models/ocr-lines.joblib'

def predict(row,height,bundle,threshold=.5,catalog_threshold=60):
    groups=line_groups(row,height)
    if not groups:return {'items':[]}
    probs=bundle['model'].predict_proba([g['text'] for g in groups])[:,1]
    catalog=bundle['catalog']; items=[]
    gaps=[groups[i+1]['median_y']-groups[i]['median_y'] for i in range(len(groups)-1)]
    typical_gap=statistics.median(gaps) if gaps else 0
    for index,(group,prob) in enumerate(zip(groups,probs)):
        if prob < threshold: continue
        text=group['text']
        if not MONEY.search(text) and index+1<len(groups):
            nxt=groups[index+1]; numeric=nxt['text']
            if nxt['median_y']-group['median_y'] <= typical_gap*1.6 and MONEY.search(numeric) and not re.search(r'[а-яА-ЯёЁa-wyzA-WYZ]',numeric):
                text += ' ' + numeric
        text=re.sub(r'^(?:ТОВАР[. :]*|НДС\s*\d+%\s*)+', '', text, flags=re.I)
        text=re.sub(r'(\d+[.,]\d{2})1(?!\d)', r'\1*1', text); matches=list(MONEY.finditer(text))
        name=text[:matches[0].start()].strip(' :*-') if matches else text.strip()
        query=alphabetic(name)
        if len(query)<3: continue
        match=process.extractOne(query,catalog,processor=alphabetic,scorer=fuzz.ratio,score_cutoff=catalog_threshold)
        raw_name=name
        if match: name=match[0]
        values=[normalize_sum(re.sub(r'\s+', '', m.group())) for m in matches]
        q=re.search(r'[xх*×]\s*(\d+(?:[.,]\d+)?)(?![\d.,а-яёa-z])',text[matches[0].start():],re.I) if matches else None
        if not q and matches:
            q=re.search(r'(?<![\w.,])(\d+(?:[.,]\d+)?)\s*[xх*×]\s*$',text[:matches[0].start()],re.I)
        quantity=q.group(1).replace(',','.') if q else None
        quantity_inferred=False
        price=values[0] if values else None
        total=values[-1] if len(values)>1 else None
        sum_inferred=False
        if quantity is None and price is not None and (total is None or total == price):
            quantity = '1'
            quantity_inferred=True
            if total is None:
                total = price
                sum_inferred=True
        if price and quantity and total is None:
            try:
                total=str((Decimal(price)*Decimal(quantity)).quantize(Decimal('.01')))
                sum_inferred=True
            except InvalidOperation: pass
        items.append({'name':name,'quantity':quantity,'price':price,'sum':total,'raw_name':raw_name,'name_source':'catalog' if match else 'ocr','quantity_inferred':quantity_inferred,'sum_inferred':sum_inferred,'line_score':round(float(prob),3),'catalog_similarity':round(match[1]/100,3) if match else None})
    return {'items':items}

def evaluate(bundle,threshold,catalog_threshold,output,ocr_path):
    data={r['image_id']:r for r in rows(ML/'data/processed/receipt-ocr-ru-quantity-price/validation.jsonl')}
    ocr=list(rows(ocr_path)); c=Counter(); predictions=[]
    for row in ocr:
        gold=data[row['image_id']]; pred=predict(row,gold['height'],bundle,threshold,catalog_threshold); predictions.append({'image_id':row['image_id'],**pred}); actual=pred['items']; expected=gold['target']['items']; c['gold']+=len(expected); c['predicted']+=len(actual)
        def norm(v,field):
            if v is None:return None
            if field=='name':return normalize_name(v)
            try:return str(Decimal(str(v).replace(',','.')).normalize())
            except InvalidOperation:return None
        for field in ('name','quantity','price','sum'):
            a=Counter(filter(None,(norm(x.get(field),field) for x in actual))); b=Counter(filter(None,(norm(x.get(field),field) for x in expected))); c[field]+=sum((a&b).values())
        def key(x):
            v=tuple(norm(x.get(f),f) for f in ('name','quantity','price','sum'))
            return v if all(v) else None
        actual_keys=list(map(key,actual)); gold_keys=list(map(key,expected))
        c['full_row']+=sum((Counter(filter(None,actual_keys))&Counter(filter(None,gold_keys))).values())
        if all(actual_keys) and all(gold_keys) and Counter(actual_keys)==Counter(gold_keys): c['exact_receipts']+=1
    metrics={'receipts':len(ocr),'train_ocr_receipts':bundle['receipts'],'gold_items':c['gold'],'predicted_items':c['predicted'],'exact_receipts':c['exact_receipts'],'threshold':threshold,'catalog_threshold':catalog_threshold,'fields':{}}
    for f in ('name','quantity','price','sum','full_row'):
        n=c[f]; p=n/max(1,c['predicted']); r=n/max(1,c['gold']); metrics['fields'][f]={'matches':n,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0}
    output.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n'); output.with_suffix('.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in predictions));print(json.dumps(metrics,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('image',nargs='?',type=Path);p.add_argument('--evaluate',action='store_true');p.add_argument('--threshold',type=float,default=.5);p.add_argument('--catalog-threshold',type=float,default=60);p.add_argument('--output',type=Path,default=ML/'reports/ocr_learned_validation.json');p.add_argument('--ocr',type=Path,default=ML/'reports/paddleocr_validation.jsonl');p.add_argument('--recognizer',choices=('v5','v6'),default='v5');p.add_argument('--model',type=Path,default=MODEL);a=p.parse_args();bundle=joblib.load(a.model)
    if a.evaluate:return evaluate(bundle,a.threshold,a.catalog_threshold,a.output,a.ocr)
    if not a.image:p.error('image required')
    import os
    os.environ.setdefault('PADDLE_PDX_CACHE_HOME',str(ML/'models/paddleocr-cache'))
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True')
    from PIL import Image
    if a.recognizer == 'v6':
        import cv2
        from paddleocr import TextDetection
        from rerecognize_v6 import load_recognizer, rerecognize
        detector=TextDetection(model_name='PP-OCRv5_mobile_det',limit_side_len=736,limit_type='min',device='cpu',cpu_threads=4,enable_mkldnn=False)
        detected=detector.predict(str(a.image))[0].json['res']
        polys=detected['dt_polys']
        boxes=[[float(min(pt[0] for pt in p)),float(min(pt[1] for pt in p)),float(max(pt[0] for pt in p)),float(max(pt[1] for pt in p))] for p in polys]
        row={'rec_polys':polys,'rec_boxes':boxes}
        row=rerecognize(row,cv2.imread(str(a.image)),load_recognizer())
    else:
        from paddleocr import PaddleOCR
        ocr=PaddleOCR(text_detection_model_name='PP-OCRv5_mobile_det',text_recognition_model_name='eslav_PP-OCRv5_mobile_rec',text_det_limit_side_len=736,text_det_limit_type='min',use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,device='cpu',cpu_threads=4,enable_mkldnn=False)
        row=ocr.predict(str(a.image))[0].json['res']
    with Image.open(a.image) as image: height=image.height
    print(json.dumps(predict(row,height,bundle,a.threshold,a.catalog_threshold),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
