"""Cache original Donut vision features, benchmark, fine-tune its decoder and evaluate."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
import platform
import random
import time

from PIL import Image
import torch
from torch.nn import functional as F
import transformers
from transformers.modeling_outputs import BaseModelOutput

from receipt_common import (
    BASE, FEATURES, IMAGE_SIZE, ML, PREPARED, PROMPT, RAW, REPORTS, RUN,
    base_processor, device, load_base, load_receipt_processor, parse_prediction, read_jsonl,
    receipt_processor, score_predictions, write_json,
)


def log(message: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def feature_path(row: dict):
    return FEATURES / f"{row['pixel_sha256']}.pt"


def load_features(row: dict, target_device: torch.device) -> torch.Tensor:
    return torch.load(feature_path(row), map_location="cpu", weights_only=True).to(
        target_device, dtype=torch.float32)


def save_features(row: dict, hidden: torch.Tensor) -> None:
    destination = feature_path(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".{os.getpid()}.partial")
    torch.save(hidden.to("cpu", dtype=torch.float16), temporary)
    temporary.replace(destination)


def cache(args) -> None:
    target_device = device(args.device)
    FEATURES.mkdir(parents=True, exist_ok=True)
    processor = base_processor()
    fingerprint = {"model_weight_sha256": hashlib.file_digest((BASE / "pytorch_model.bin").open("rb"), "sha256").hexdigest(),
                   "image_processor": processor.image_processor.to_dict(), "feature_dtype": "float16"}
    manifest = FEATURES / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
        raise ValueError("Feature cache configuration changed; use a separate cache directory")
    write_json(manifest, fingerprint)
    rows = [row for split in ("train", "validation", "test") for row in read_jsonl(PREPARED / f"{split}.jsonl")]
    model = load_base()
    encoder = model.encoder.eval().to(target_device)
    del model
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        destination = feature_path(row)
        if not destination.exists():
            with Image.open(RAW / row["image"]) as source:
                pixels = processor(source.convert("RGB"), return_tensors="pt").pixel_values.to(target_device)
            with torch.inference_mode():
                hidden = encoder(pixels).last_hidden_state.to("cpu", dtype=torch.float16)
            save_features(row, hidden)
            del pixels, hidden
        if index % 25 == 0 or index == len(rows):
            log(f"Vision features {index}/{len(rows)}, elapsed {time.perf_counter()-started:.1f}s")
            if target_device.type == "mps":
                torch.mps.empty_cache()
    log("Feature cache complete")


def targets(rows: list[dict], tokenizer, max_tokens: int) -> tuple[list[dict], list[str]]:
    usable, excluded = [], []
    for row in rows:
        ids = tokenizer(PROMPT + row["target_text"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        if len(ids) > max_tokens:
            excluded.append(row["image_id"])
        else:
            usable.append(dict(row, token_ids=ids))
    return usable, excluded


def loss_for(model, row: dict, target_device: torch.device):
    sequence = torch.tensor(row["token_ids"], dtype=torch.long, device=target_device)
    count = sequence.numel() - 1
    padded_length = math.ceil(count / 64) * 64
    inputs = torch.full((1, padded_length), model.config.pad_token_id, dtype=torch.long, device=target_device)
    labels = torch.full_like(inputs, -100)
    attention = torch.zeros_like(inputs)
    inputs[0, :count] = sequence[:-1]
    labels[0, :count] = sequence[1:]
    attention[0, :count] = 1
    hidden = load_features(row, target_device)
    logits = model(input_ids=inputs, attention_mask=attention, encoder_hidden_states=hidden, use_cache=False).logits
    # Explicit one-token shift: predict the first JSON token from the task prefix.
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)), count


def validation_loss(model, rows: list[dict], target_device: torch.device) -> float:
    model.eval()
    total_loss = total_tokens = 0
    with torch.inference_mode():
        for row in rows:
            loss, count = loss_for(model, row, target_device)
            total_loss += loss.item() * count
            total_tokens += count
    model.train()
    return total_loss / total_tokens


def train(args) -> None:
    target_device = device(args.device)
    torch.manual_seed(42)
    random.seed(42)
    RUN.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    processor, tokenizer_metadata = receipt_processor()
    write_json(REPORTS / "tokenizer.json", tokenizer_metadata)
    model = load_base()
    original_size = model.decoder.get_input_embeddings().weight.shape[0]
    model.decoder.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    # Initialise added alphabet pieces from related original pieces when possible.
    with torch.no_grad():
        original_processor = base_processor()
        embedding = model.decoder.get_input_embeddings().weight
        for index in range(original_size, len(processor.tokenizer)):
            token = processor.tokenizer.convert_ids_to_tokens(index)
            source_ids = original_processor.tokenizer(token, add_special_tokens=False).input_ids
            if processor.tokenizer.unk_token_id in source_ids:
                source_ids = []
            source_ids = [item for item in source_ids if item < original_size]
            if token == PROMPT:
                source_ids = original_processor.tokenizer("<s_synthdog>", add_special_tokens=False).input_ids
            if source_ids:
                embedding[index].copy_(embedding[source_ids].mean(dim=0))
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(PROMPT)
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.eos_token_id
    model.generation_config.forced_eos_token_id = None
    model.encoder.requires_grad_(False)
    decoder = model.decoder.to(target_device)
    decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_rows, excluded_train = targets(read_jsonl(PREPARED / "train.jsonl"), processor.tokenizer, args.max_tokens)
    val_rows, excluded_val = targets(read_jsonl(PREPARED / "validation.jsonl"), processor.tokenizer, args.max_tokens)
    if not train_rows or not val_rows:
        raise ValueError("Training and validation sets must both be nonempty")
    config = {"base_model": "naver-clova-ix/donut-base", "seed": 42,
              "epochs": args.epochs, "learning_rate": args.lr, "gradient_accumulation": args.accumulation,
              "max_sequence_tokens": args.max_tokens, "image_size": IMAGE_SIZE, "encoder_frozen": True,
              "gradient_checkpointing": True,
              "loss_padding_multiple": 64, "do_align_long_axis": False,
              "trainable_parameters": sum(parameter.numel() for parameter in decoder.parameters()),
              "train_examples": len(train_rows), "validation_examples": len(val_rows),
              "excluded_overlength_train_ids": excluded_train, "excluded_overlength_validation_ids": excluded_val,
              "device": str(target_device), "torch": torch.__version__, "transformers": transformers.__version__,
              "python": platform.python_version(), "created_utc": datetime.now(timezone.utc).isoformat(),
              "selection": "lowest token-weighted validation cross-entropy; test never used for selection",
              "preparation_sha256": hashlib.sha256((ML / "reports/data_preparation.json").read_bytes()).hexdigest()}
    write_json(REPORTS / "config.json", config)
    processor.save_pretrained(RUN / "processor")
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.01, foreach=False)
    total_steps = math.ceil(len(train_rows) / args.accumulation) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup_steps, max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))))
    best_loss = float("inf")
    start_epoch = step = 0
    last_checkpoint = RUN / "last.pt"
    if args.resume and last_checkpoint.exists():
        state = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
        decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        for values in optimizer.state.values():
            for name, value in values.items():
                if isinstance(value, torch.Tensor) and name != "step":
                    values[name] = value.to(target_device)
        start_epoch, step, best_loss = state["epoch"], state["step"], state["best_loss"]
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if target_device.type == "mps" and state.get("mps_rng_state") is not None:
            torch.mps.set_rng_state(state["mps_rng_state"])
        log(f"Resumed after epoch {start_epoch}")
    history_path = REPORTS / "training.jsonl"
    baseline_val = validation_loss(decoder, val_rows, target_device)
    log(f"Validation loss before this run: {baseline_val:.4f}; train={len(train_rows)}, validation={len(val_rows)}")
    if not args.resume:
        write_json(REPORTS / "initial_validation.json", {"loss": baseline_val})
    started = time.perf_counter()
    with history_path.open("a" if args.resume else "w", encoding="utf-8") as history:
        for epoch in range(start_epoch, args.epochs):
            decoder.train()
            order = list(range(len(train_rows)))
            random.Random(42 + epoch).shuffle(order)
            total_loss = total_tokens = 0
            optimizer.zero_grad(set_to_none=True)
            for position, index in enumerate(order):
                group_start = (position // args.accumulation) * args.accumulation
                group_size = min(args.accumulation, len(order) - group_start)
                loss, tokens = loss_for(decoder, train_rows[index], target_device)
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch+1}, example {index}")
                (loss / group_size).backward()
                total_loss += value * tokens
                total_tokens += tokens
                if (position + 1) % args.accumulation == 0 or position + 1 == len(order):
                    norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Non-finite gradients")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                if (position + 1) % 25 == 0 or position + 1 == len(order):
                    event = {"epoch": epoch + 1, "examples": position + 1, "optimizer_step": step,
                             "train_loss": total_loss / total_tokens, "learning_rate": scheduler.get_last_lr()[0],
                             "elapsed_seconds": round(time.perf_counter() - started, 1)}
                    history.write(json.dumps(event) + "\n")
                    history.flush()
                    log(json.dumps(event))
                    if target_device.type == "mps":
                        torch.mps.empty_cache()
            val_loss = validation_loss(decoder, val_rows, target_device)
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                best_path = RUN / "best"
                # Save a standard HF checkpoint that loads without custom model code.
                model.save_pretrained(best_path, safe_serialization=True)
                processor.save_pretrained(best_path)
                write_json(REPORTS / "best_checkpoint.json", {"epoch": epoch + 1, "validation_loss": val_loss,
                                                            "path": str(best_path.relative_to(ML))})
            event = {"epoch": epoch + 1, "train_loss": total_loss / total_tokens,
                     "validation_loss": val_loss, "best": is_best, "elapsed_seconds": round(time.perf_counter() - started, 1)}
            history.write(json.dumps(event) + "\n")
            history.flush()
            log(json.dumps(event))
            state = {"decoder": {name: tensor.detach().cpu() for name, tensor in decoder.state_dict().items()},
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "epoch": epoch + 1, "step": step, "best_loss": best_loss,
                     "torch_rng_state": torch.get_rng_state(),
                     "mps_rng_state": torch.mps.get_rng_state() if target_device.type == "mps" else None}
            temporary = RUN / "last.partial.pt"
            torch.save(state, temporary)
            temporary.replace(last_checkpoint)
            del state
            gc.collect()
    write_json(REPORTS / "training_complete.json", {"epochs_completed": args.epochs, "optimizer_steps": step,
                                                  "best_validation_loss": best_loss,
                                                  "training_seconds": round(time.perf_counter() - started, 1)})
    log("Training complete")


def evaluate(args) -> None:
    target_device = device(args.device)
    REPORTS.mkdir(parents=True, exist_ok=True)
    original = args.command == "baseline"
    if original:
        processor = base_processor()
        model = load_base()
        prompt = "<s_synthdog>"
        name = "baseline_fixed_test"
    else:
        from transformers import VisionEncoderDecoderModel
        processor = load_receipt_processor(RUN / "best")
        model = VisionEncoderDecoderModel.from_pretrained(str(RUN / "best"))
        prompt = PROMPT
        name = f"finetuned_{args.split}"
    model.eval().to(target_device)
    rows = read_jsonl(PREPARED / f"{args.split}.jsonl")
    preparation = json.loads((ML / "reports/data_preparation.json").read_text())
    fixed_ids = set(preparation["fixed_test_ids"])
    if original:
        rows = [row for row in rows if row["image_id"] in fixed_ids]
    if args.limit:
        rows = rows[:args.limit]
    prompt_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(target_device)
    output = REPORTS / f"{name}.jsonl"
    results = read_jsonl(output) if args.resume and output.exists() else []
    completed_ids = {row["image_id"] for row in results}
    started = time.perf_counter()
    with output.open("a" if args.resume else "w", encoding="utf-8") as stream:
        for row in rows:
            if row["image_id"] in completed_ids:
                continue
            if not feature_path(row).exists():
                with Image.open(RAW / row["image"]) as source:
                    pixels = processor(source.convert("RGB"), return_tensors="pt").pixel_values.to(target_device)
                with torch.inference_mode():
                    hidden = model.encoder(pixels).last_hidden_state
                save_features(row, hidden)
                del pixels, hidden
            with torch.inference_mode():
                generated = model.generate(
                    decoder_input_ids=prompt_ids,
                    encoder_outputs=BaseModelOutput(last_hidden_state=load_features(row, target_device)),
                    max_new_tokens=args.generation_tokens, pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id, forced_eos_token_id=None,
                    use_cache=True, num_beams=1, do_sample=False,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                )
            generated_ids = generated[0, prompt_ids.shape[-1]:].tolist()
            text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
            predicted, valid = parse_prediction(text)
            result = {"image_id": row["image_id"], "image": row["image"], "target": row["target"],
                      "prompt": prompt, "generated_text": text, "predicted_items": predicted, "valid_json": valid,
                      "generated_tokens": len(generated_ids), "ended_with_eos": bool(generated_ids and generated_ids[-1] == processor.tokenizer.eos_token_id),
                      "fixed_comparison_subset": row["image_id"] in fixed_ids}
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            log(f"{name} {len(results)}/{len(rows)}, valid={valid}, tokens={len(generated_ids)}, elapsed={time.perf_counter()-started:.1f}s")
            if target_device.type == "mps":
                torch.mps.empty_cache()
    metrics = score_predictions(results)
    metrics["generation_max_new_tokens"] = args.generation_tokens
    metrics["split"] = args.split
    metrics["original_checkpoint"] = original
    metrics["device"] = str(target_device)
    write_json(REPORTS / f"{name}_metrics.json", metrics)
    if not original and args.split == "test":
        write_json(REPORTS / "finetuned_fixed_test_metrics.json",
                   score_predictions([row for row in results if row["fixed_comparison_subset"]]))
    log(json.dumps(metrics, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cache", "baseline", "train", "evaluate"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--generation-tokens", type=int, default=768)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    log(f"Starting {args.command} on {device(args.device)}")
    if args.command == "cache":
        cache(args)
    elif args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
