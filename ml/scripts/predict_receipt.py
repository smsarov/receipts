"""Run the locally fine-tuned Donut checkpoint on a receipt image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps
import torch
from transformers import VisionEncoderDecoderModel
from transformers.modeling_outputs import BaseModelOutput

from receipt_common import PROMPT, RUN, device, load_receipt_processor, parse_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=RUN / "best")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    target_device = device(args.device)
    processor = load_receipt_processor(args.checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(str(args.checkpoint)).eval().to(target_device)
    with Image.open(args.image) as source:
        pixels = processor(ImageOps.exif_transpose(source).convert("RGB"), return_tensors="pt").pixel_values.to(target_device)
    prompt = processor.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids.to(target_device)
    with torch.inference_mode():
        hidden = model.encoder(pixels).last_hidden_state.to(dtype=torch.float16).float()
        output = model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            decoder_input_ids=prompt, max_new_tokens=args.max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=processor.tokenizer.eos_token_id,
            forced_eos_token_id=None, bad_words_ids=[[processor.tokenizer.unk_token_id]],
            num_beams=1, do_sample=False, use_cache=True,
        )
    text = processor.tokenizer.decode(output[0, prompt.shape[-1]:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
    items, valid = parse_prediction(text)
    print(json.dumps({"items": items} if valid else {"valid_json": False, "generated_text": text},
                     ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
