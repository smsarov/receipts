"""Check that training changed the decoder and preserved original encoder and token IDs."""

from __future__ import annotations

import hashlib
import json

from safetensors import safe_open
import torch

from receipt_common import BASE, FEATURES, IMAGE_SIZE, REPORTS, RUN, base_processor, load_receipt_processor, write_json


def main() -> None:
    original_path = BASE / "pytorch_model.bin"
    with original_path.open("rb") as stream:
        original_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    feature_manifest = json.loads((FEATURES / "manifest.json").read_text())
    assert original_sha == feature_manifest["model_weight_sha256"], "Original checkpoint file changed"
    original = torch.load(original_path, map_location="cpu", weights_only=True, mmap=True)
    checked = 0
    differences = {}
    with safe_open(RUN / "best/model.safetensors", framework="pt", device="cpu") as trained:
        keys = set(trained.keys())
        for name, value in original.items():
            if name.startswith("encoder."):
                assert name in keys and torch.equal(value, trained.get_tensor(name)), f"Encoder changed: {name}"
                checked += 1
        for name in ("decoder.model.decoder.layers.0.self_attn.q_proj.weight",
                     "decoder.model.decoder.embed_tokens.weight"):
            assert name in keys
            value = trained.get_tensor(name)[:original[name].shape[0]]
            difference = (value - original[name]).abs().mean().item()
            assert difference > 0, f"Decoder did not change: {name}"
            differences[name] = difference
    assert checked > 0
    source_vocab = base_processor().tokenizer.get_vocab()
    processor = load_receipt_processor(RUN / "best")
    trained_vocab = processor.tokenizer.get_vocab()
    assert all(trained_vocab.get(token) == token_id for token, token_id in source_vocab.items())
    assert processor.image_processor.size == IMAGE_SIZE
    assert processor.image_processor.do_align_long_axis is False
    report = {"original_weight_sha256": original_sha, "original_weights_unchanged": True,
              "encoder_tensors_compared": checked, "encoder_identical": True,
              "decoder_mean_absolute_weight_changes": differences,
              "original_token_ids_preserved": True, "processor_geometry_verified": True}
    write_json(REPORTS / "checkpoint_integrity.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
