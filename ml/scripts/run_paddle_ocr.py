"""Run Russian PaddleOCR over one prepared split and save raw boxes/text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from paddleocr import PaddleOCR

ML = Path(__file__).resolve().parents[1]
RAW = ML / "data/raw/receipt-ocr-ru"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--det-limit-side-len", type=int, default=736)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_jsonl(ML / "data/processed/receipt-ocr-ru-quantity-price" / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    output = args.output or ML / f"reports/paddleocr_{args.split}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
        text_det_limit_side_len=args.det_limit_side_len,
        text_det_limit_type="min",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        cpu_threads=4,
        enable_mkldnn=False,
    )
    results = []
    started = time.perf_counter()
    with output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            image = RAW / row["image"]
            prediction = ocr.predict(str(image))[0].json["res"]
            result = {
                "image_id": row["image_id"], "image": row["image"], "target": row["target"],
                "rec_boxes": prediction.get("rec_boxes", []).tolist()
                    if hasattr(prediction.get("rec_boxes"), "tolist") else prediction.get("rec_boxes", []),
                "rec_polys": prediction.get("rec_polys", []).tolist()
                    if hasattr(prediction.get("rec_polys"), "tolist") else prediction.get("rec_polys", []),
                "rec_texts": prediction.get("rec_texts", []),
                "rec_scores": prediction.get("rec_scores", []).tolist()
                    if hasattr(prediction.get("rec_scores"), "tolist") else prediction.get("rec_scores", []),
                "model": "PP-OCRv5_mobile_det+eslav_PP-OCRv5_mobile_rec",
            }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(result)
            print(f"{args.split} {index}/{len(rows)} texts={len(result['rec_texts'])} "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
