"""Train a weakly supervised item-line classifier from train OCR and labels."""
import argparse, json, re, random
from pathlib import Path
import joblib
from rapidfuzz import fuzz, process
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from analyze_paddle_ocr import alphabetic
from ocr_grouping import line_groups
from receipt_common import ML, normalize_name

DATA = ML / 'data/processed/receipt-ocr-ru-quantity-price'
MODEL = ML / 'models/ocr-lines.joblib'

def rows(path):
    with path.open() as f:
        for line in f:
            try: yield json.loads(line)
            except json.JSONDecodeError: continue

def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, default=ML/'reports/paddleocr_train.jsonl'); p.add_argument('--output', type=Path, default=MODEL); p.add_argument('--synthetic-copies',type=int,default=3); a=p.parse_args()
    train={r['image_id']:r for r in rows(DATA/'train.jsonl')}
    texts=[]; labels=[]; receipts=0
    for row in rows(a.input):
        if row['image_id'] not in train: continue
        receipt=train[row['image_id']]; groups=line_groups(row,receipt['height']); targets=receipt['target']['items']
        pairs=[]
        for i,g in enumerate(groups):
            text=alphabetic(g['text'])
            if len(text)<4: continue
            for j,t in enumerate(targets):
                name=alphabetic(t['name'])
                if len(name)<4: continue
                score=fuzz.ratio(text,name)
                if len(text)>len(name): score=max(score, fuzz.partial_ratio(name,text)*0.9)
                pairs.append((score,i,j))
        assigned=set(); used=set()
        for score,i,j in sorted(pairs,reverse=True):
            if score<60: break
            if i not in assigned and j not in used: assigned.add(i); used.add(j)
        for i,g in enumerate(groups):
            texts.append(g['text']); labels.append(int(i in assigned))
        receipts+=1
    if sum(labels)<10: raise ValueError('Too few matched item lines')
    rng=random.Random(42)
    synthetic=0
    for receipt in train.values():
        for item in receipt['target']['items']:
            name=item['name']
            for _ in range(a.synthetic_copies):
                chars=list(name.lower())
                for i in range(len(chars)):
                    if rng.random()<0.05: chars[i]=rng.choice('аоеи')
                text=''.join(chars)+' '+item['price']+'*'+item['quantity']+' '+item['sum']
                texts.append(text); labels.append(1); synthetic+=1
    model=make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,sublinear_tf=True),LogisticRegression(max_iter=500,class_weight='balanced',C=5))
    model.fit(texts,labels)
    catalog=sorted({normalize_name(item['name']) for row in train.values() for item in row['target']['items']})
    a.output.parent.mkdir(parents=True,exist_ok=True); joblib.dump({'model':model,'catalog':catalog,'receipts':receipts},a.output)
    print(json.dumps({'receipts':receipts,'lines':len(texts),'positive':sum(labels),'synthetic':synthetic,'catalog':len(catalog),'output':str(a.output)}))
if __name__=='__main__': main()
