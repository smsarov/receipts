"""Run the original Donut checkpoint with its pretraining task prompts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel


ML_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ML_DIR / "data/raw/receipt-ocr-ru"
MODEL_DIR = ML_DIR / "models/donut-base"
OUTPUT_PATH = ML_DIR / "reports/donut_base_results.jsonl"
SAMPLES = ("IMG_1399_bin", "IMG_0817_bin", "IMG_0965_bin")
PROMPTS = ("<s_synthdog>", "<s_iitcdip>")


def main() -> None:
    torch.set_num_threads(4)
    processor = DonutProcessor.from_pretrained(str(MODEL_DIR), use_fast=False)
    model, loading_info = VisionEncoderDecoderModel.from_pretrained(
        str(MODEL_DIR), use_safetensors=False, output_loading_info=True
    )
    model.eval()
    unexpected = {
        name: values
        for name, values in loading_info.items()
        if values and name in {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    }
    if unexpected:
        raise RuntimeError(f"Checkpoint did not load cleanly: {unexpected}")

    annotations: dict[str, tuple[str, dict]] = {}
    for split in ("train", "validation", "test"):
        with (DATA_DIR / f"annotations/{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["image_id"] in SAMPLES:
                    annotations[row["image_id"]] = (split, row)

    missing_ids = set(SAMPLES) - annotations.keys()
    if missing_ids:
        raise ValueError(f"Sample annotations were not found: {sorted(missing_ids)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for image_id in SAMPLES:
            split, row = annotations[image_id]
            image_path = DATA_DIR / row["image"]
            if not image_path.is_file():
                print(f"Skip {image_id}: image is not downloaded ({image_path})")
                continue

            with Image.open(image_path) as source:
                pixel_values = processor(source.convert("RGB"), return_tensors="pt").pixel_values

            for prompt in PROMPTS:
                prompt_ids = processor.tokenizer(
                    prompt, add_special_tokens=False, return_tensors="pt"
                ).input_ids
                started = time.perf_counter()
                with torch.inference_mode():
                    generation = model.generate(
                        pixel_values,
                        decoder_input_ids=prompt_ids,
                        max_new_tokens=80,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        bad_words_ids=[[processor.tokenizer.unk_token_id]],
                        return_dict_in_generate=True,
                    )

                sequence = generation.sequences[0].tolist()
                generated_ids = sequence[prompt_ids.shape[-1] :]
                result = {
                    "model": "naver-clova-ix/donut-base",
                    "fine_tuned": False,
                    "split": split,
                    "image_id": image_id,
                    "image": row["image"],
                    "prompt": prompt,
                    "prompt_token_ids": prompt_ids[0].tolist(),
                    "generated_token_ids": generated_ids,
                    "generated_text": processor.tokenizer.decode(
                        generated_ids, skip_special_tokens=False
                    ),
                    "expected_items": [
                        {"name": item.get("text"), "sum": item.get("sum")}
                        for item in row.get("items", [])
                    ],
                    "inference_seconds": round(time.perf_counter() - started, 2),
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                print(json.dumps(result, ensure_ascii=False), flush=True)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
