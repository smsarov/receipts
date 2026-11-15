"""Reusable local receipt predictor: PP-OCR detection, Cyrillic OCR, item classifier."""
import os
from pathlib import Path
import joblib
import cv2
from PIL import Image
from ocr_learned import predict
from rerecognize_v6 import load_recognizer, rerecognize
ML=Path(__file__).resolve().parents[1]

class ReceiptPredictor:
    def __init__(self, threshold=0.5, catalog_threshold=60):
        os.environ.setdefault('PADDLE_PDX_CACHE_HOME',str(ML/'models/paddleocr-cache'))
        os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True')
        from paddleocr import TextDetection
        self.detector=TextDetection(model_name='PP-OCRv5_mobile_det',limit_side_len=736,limit_type='min',device='cpu',cpu_threads=4,enable_mkldnn=False)
        self.recognizer=load_recognizer()
        self.bundle=joblib.load(ML/'models/ocr-lines.joblib')
        self.threshold=threshold
        self.catalog_threshold=catalog_threshold

    def __call__(self,image_path):
        path=Path(image_path)
        image=cv2.imread(str(path))
        if image is None: raise ValueError(f'Cannot read image: {path}')
        detected=self.detector.predict(str(path))[0].json['res']
        polys=detected['dt_polys']
        boxes=[[float(min(pt[0] for pt in p)),float(min(pt[1] for pt in p)),float(max(pt[0] for pt in p)),float(max(pt[1] for pt in p))] for p in polys]
        row=rerecognize({'rec_polys':polys,'rec_boxes':boxes},image,self.recognizer)
        return predict(row,image.shape[0],self.bundle,self.threshold,self.catalog_threshold)

if __name__=='__main__':
    import argparse,contextlib,json,sys
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    args=parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        output=ReceiptPredictor()(args.image)
    print(json.dumps(output,ensure_ascii=False,indent=2))
