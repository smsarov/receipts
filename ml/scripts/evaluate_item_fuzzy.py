"""Score one-to-one item matches by name similarity and linked numeric fields."""
import argparse,json,re,unicodedata
from collections import Counter
from decimal import Decimal,InvalidOperation
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from rapidfuzz.fuzz import ratio
ML=Path(__file__).resolve().parents[1]
def name(s):return ' '.join(re.findall(r'[a-zа-яё0-9]+',unicodedata.normalize('NFKC',s).casefold().replace('ё','е')))
def numeric(value):
    try:return Decimal(str(value).replace(',','.')).normalize() if value is not None else None
    except InvalidOperation:return None
def main():
    p=argparse.ArgumentParser();p.add_argument('predictions',type=Path);p.add_argument('--threshold',type=float,default=.8);a=p.parse_args()
    data={r['image_id']:r for r in map(json.loads,(ML/'data/processed/receipt-ocr-ru-quantity-price/validation.jsonl').read_text().splitlines())};pred=list(map(json.loads,a.predictions.read_text().splitlines()));counts=Counter()
    for row in pred:
        gold=data[row['image_id']]['target']['items'];items=row['items'];counts['gold']+=len(gold);counts['predicted']+=len(items)
        if not items or not gold:continue
        scores=np.array([[ratio(name(x['name']),name(y['name']))/100 for y in gold] for x in items]);ri,ci=linear_sum_assignment(scores,maximize=True)
        for i,j in zip(ri,ci):
            if scores[i,j]<a.threshold:continue
            counts['name']+=1; hits=[]
            for field in ('quantity','price','sum'):
                actual=numeric(items[i].get(field));expected=numeric(gold[j].get(field));hit=actual is not None and expected is not None and actual==expected;counts[field]+=int(hit);hits.append(hit)
            counts['full_row']+=int(all(hits))
    result={'receipts':len(pred),'name_similarity_threshold':a.threshold,'gold_items':counts['gold'],'predicted_items':counts['predicted'],'fields':{}}
    for field in ('name','quantity','price','sum','full_row'):
        n=counts[field];p=n/max(1,counts['predicted']);r=n/max(1,counts['gold']);result['fields'][field]={'matches':n,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0}
    path=a.predictions.with_suffix('.fuzzy-metrics.json');path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
