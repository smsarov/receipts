"""Experimental receipt extraction from PaddleOCR lines; missing fields are null."""
import argparse
import json
import re
from pathlib import Path
from collections import Counter
from decimal import Decimal, InvalidOperation
from analyze_paddle_ocr import line_groups
from receipt_common import ML, normalize_name, normalize_sum

MONEY = re.compile(r'(?<!\d)(\d{1,6}[.,]\d{2})(?!\d)')
SKIP = re.compile(r'итог|оплат|сдача|кассир|ндс|налог|сумма|безнал|карт[аы]|чек|скидк|товар|приход', re.I)

def extract(row, height):
    items = []
    for group in line_groups(row, height):
        text = group['text']
        matches = list(MONEY.finditer(text))
        if not matches or SKIP.search(text):
            continue
        name = text[:matches[0].start()].strip(' -:*№')
        name = re.sub(r'^\d+[.)]?\s+', '', name)
        if len(re.findall(r'[а-яА-ЯёЁa-zA-Z]', name)) < 3:
            continue
        values = [normalize_sum(m.group()) for m in matches]
        quantity = None
        q = re.search(r'[xх*×]\s*(\d+(?:[.,]\d+)?)', text, re.I)
        if q:
            quantity = q.group(1).replace(',', '.')
        price = values[0]
        total = values[-1] if len(values) > 1 else None
        if total is None and quantity:
            try:
                total = str((Decimal(price) * Decimal(quantity)).quantize(Decimal('.01')))
            except InvalidOperation:
                pass
        items.append({'name': name, 'quantity': quantity, 'price': price, 'sum': total})
    return {'items': items}

def evaluate():
    data = {r['image_id']: r for r in map(json.loads, (ML / 'data/processed/receipt-ocr-ru-quantity-price/validation.jsonl').read_text().splitlines())}
    rows = list(map(json.loads, (ML / 'reports/paddleocr_validation.jsonl').read_text().splitlines()))
    counts = Counter(); predictions = []
    for row in rows:
        gold = data[row['image_id']]; pred = extract(row, gold['height']); predictions.append({'image_id': row['image_id'], **pred})
        expected = gold['target']['items']; actual = pred['items']
        counts['gold'] += len(expected); counts['predicted'] += len(actual)
        for field in ('name', 'quantity', 'price', 'sum'):
            def norm(v):
                if v is None: return None
                if field == 'name': return normalize_name(v)
                try: return str(Decimal(str(v).replace(',', '.')).normalize())
                except InvalidOperation: return None
            a = Counter(filter(None, (norm(x.get(field)) for x in actual)))
            b = Counter(filter(None, (norm(x.get(field)) for x in expected)))
            counts[field] += sum((a & b).values())
        def key(item):
            try:
                return (normalize_name(item['name']), *(str(Decimal(str(item[f]).replace(',', '.')).normalize()) for f in ('quantity','price','sum')))
            except (InvalidOperation, TypeError): return None
        counts['full_row'] += sum((Counter(filter(None, map(key, actual))) & Counter(filter(None, map(key, expected)))).values())
    metrics = {'receipts': len(rows), 'gold_items': counts['gold'], 'predicted_items': counts['predicted'], 'fields': {}}
    for field in ('name','quantity','price','sum','full_row'):
        n = counts[field]; p = n / max(1, counts['predicted']); r = n / max(1, counts['gold'])
        metrics['fields'][field] = {'matches': n, 'precision': p, 'recall': r, 'f1': 2*p*r/(p+r) if p+r else 0}
    out = ML / 'reports/ocr_parser_validation_2026-11-01.json'
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n')
    (out.parent / 'ocr_parser_validation_2026-11-01.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in predictions))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?', type=Path)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.evaluate:
        evaluate(); return
    if not args.image: parser.error('image is required')
    from paddleocr import PaddleOCR
    from PIL import Image
    ocr = PaddleOCR(text_detection_model_name='PP-OCRv5_mobile_det', text_recognition_model_name='eslav_PP-OCRv5_mobile_rec', text_det_limit_side_len=736, text_det_limit_type='min', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device='cpu', cpu_threads=4, enable_mkldnn=False)
    result = ocr.predict(str(args.image))[0].json['res']
    with Image.open(args.image) as image: height = image.height
    print(json.dumps(extract(result, height), ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
