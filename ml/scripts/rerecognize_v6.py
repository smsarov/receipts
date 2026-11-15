"""Re-recognize saved PaddleOCR detections with Cyrillic PP-OCRv6 ONNX."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from rapidocr import RapidOCR
ML=Path(__file__).resolve().parents[1]

def crop(image,poly):
    pts=np.asarray(poly,dtype=np.float32);w=max(1,int(max(np.linalg.norm(pts[0]-pts[1]),np.linalg.norm(pts[2]-pts[3]))));h=max(1,int(max(np.linalg.norm(pts[0]-pts[3]),np.linalg.norm(pts[1]-pts[2]))));target=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype=np.float32)
    return cv2.warpPerspective(image,cv2.getPerspectiveTransform(pts,target),(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

def load_recognizer():
    recognizer=RapidOCR(params={'Global.use_det':False,'Global.use_cls':False,'Global.use_rec':True,'Rec.model_path':str(ML/'models/ppocr-v6-cyrillic/inference.onnx'),'Rec.rec_img_shape':[3,48,640],'Rec.rec_batch_num':6,'EngineConfig.onnxruntime.intra_op_num_threads':4})
    rec=recognizer._load_rec_model();old=rec.resize_norm_img;rec.rec_image_shape=[3,48,48];rec.resize_norm_img=lambda img,r:old(img,max(640/48,1.5*r))
    return recognizer

def rerecognize(row,image,recognizer):
    polys=row['rec_polys']
    if len(polys)==0:
        row['rec_texts']=[];row['rec_scores']=[];return row
    result=recognizer.recognize_txt([crop(image,p) for p in polys])
    row['rec_texts']=list(result.txts);row['rec_scores']=[float(x) for x in result.scores]
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ML/'reports/paddleocr_validation.jsonl');p.add_argument('--output',type=Path,default=ML/'reports/ppocr_v6_validation.jsonl');p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
    recognizer=load_recognizer()
    seen=set()
    if a.resume and a.output.exists():seen={json.loads(line)['image_id'] for line in a.output.read_text().splitlines() if line.strip()}
    data=[json.loads(line) for line in a.input.read_text().splitlines()];data=data[:a.limit] if a.limit else data
    with a.output.open('a' if a.resume else 'w') as out:
        for index,row in enumerate(data):
            if row['image_id'] in seen:continue
            start=time.monotonic();image=cv2.imread(str(ML/'data/raw/receipt-ocr-ru'/row['image']));row=rerecognize(row,image,recognizer)
            row['model']='PP-OCRv5_mobile_det+PP-OCRv6_medium_cyrillic_rec';out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();print(f'{index+1}/{len(data)} {row["image_id"]} {time.monotonic()-start:.1f}s',flush=True)
if __name__=='__main__':main()
