"""Shared paths, tokenizer setup, strict JSON parsing and receipt metrics."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import unicodedata

import torch
from transformers import AutoTokenizer, DonutImageProcessor, DonutProcessor, VisionEncoderDecoderModel
from tokenizers.models import Unigram

ML = Path(__file__).resolve().parents[1]
RAW = ML / "data/raw/receipt-ocr-ru"
PREPARED = ML / "data/processed/receipt-ocr-ru"
BASE = ML / "models/donut-base"
RUN = ML / "models/donut-receipts"
REPORTS = ML / "reports/finetune_2026-10-18"
FEATURES = ML / "data/features/donut-base-1280x960-upright"
PROMPT = "<s_receipt>"
IMAGE_SIZE = {"height": 1280, "width": 960}
RUSSIAN = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def device(preferred: str = "auto") -> torch.device:
    torch.set_num_threads(4)
    return torch.device(preferred if preferred != "auto" else "mps" if torch.backends.mps.is_available() else "cpu")


def base_processor() -> DonutProcessor:
    processor = DonutProcessor.from_pretrained(str(BASE), use_fast=False)
    processor.tokenizer = AutoTokenizer.from_pretrained(str(BASE), use_fast=True)
    processor.image_processor.size = IMAGE_SIZE.copy()
    processor.image_processor.do_align_long_axis = False
    return processor


def load_receipt_processor(path: Path) -> DonutProcessor:
    return DonutProcessor(image_processor=DonutImageProcessor.from_pretrained(str(path)),
                          tokenizer=AutoTokenizer.from_pretrained(str(path), use_fast=True))


def receipt_processor() -> tuple[DonutProcessor, dict]:
    processor = base_processor()
    tokenizer = processor.tokenizer
    train = read_jsonl(PREPARED / "train.jsonl")
    texts = [row["target_text"] for row in train]
    before_unknown = sum(tokenizer(text, add_special_tokens=False).input_ids.count(tokenizer.unk_token_id)
                         for text in texts)
    # The original four-language vocabulary cannot encode all Russian letters.
    # Add the Russian alphabet plus unsupported characters observed in train only.
    characters = set(RUSSIAN) | set("".join(texts))
    # Extend SentencePiece's underlying Unigram model rather than AddedToken:
    # individual AddedTokens can displace whitespace around Cyrillic words.
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    vocab = [tuple(entry) for entry in backend["model"]["vocab"]]
    original_vocab = tokenizer.get_vocab()
    for piece, index in sorted(original_vocab.items(), key=lambda pair: pair[1]):
        if index >= len(vocab):
            if index != len(vocab):
                raise ValueError("Non-contiguous original vocabulary")
            vocab.append((piece, 0.0))
    added_characters = [char for char in sorted(characters) if char not in original_vocab]
    vocab.extend((char, -10.0) for char in added_characters)
    tokenizer.backend_tokenizer.model = Unigram(vocab, unk_id=backend["model"]["unk_id"])
    added = len(added_characters)
    tokenizer.add_special_tokens({"additional_special_tokens": [PROMPT]}, replace_additional_special_tokens=False)
    lengths = {}
    after_unknown = 0
    for split in ("train", "validation", "test"):
        rows = read_jsonl(PREPARED / f"{split}.jsonl")
        split_lengths = []
        split_unknown = 0
        for row in rows:
            ids = tokenizer(row["target_text"], add_special_tokens=False).input_ids
            decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
            split_unknown += ids.count(tokenizer.unk_token_id)
            if split == "train" and decoded != row["target_text"]:
                raise ValueError(f"Tokenizer round trip failed: {row['image_id']}: {decoded!r}")
            split_lengths.append(len(ids) + 2)
        lengths[split] = {"max": max(split_lengths), "unknown_tokens": split_unknown}
        if split == "train":
            after_unknown = split_unknown
    if after_unknown:
        raise ValueError(f"Training targets still contain {after_unknown} unknown tokens")
    metadata = {"base_vocab_size": 57525, "vocab_size": len(tokenizer),
                "added_character_tokens": added, "added_characters": added_characters,
                "train_unknown_before": before_unknown, "train_unknown_after": after_unknown,
                "target_lengths": lengths, "task_prompt": PROMPT}
    return processor, metadata


def load_base() -> VisionEncoderDecoderModel:
    model, info = VisionEncoderDecoderModel.from_pretrained(
        str(BASE), use_safetensors=False, output_loading_info=True)
    problems = {key: info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if info.get(key)}
    if problems:
        raise RuntimeError(f"Checkpoint load failed: {problems}")
    return model


def normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def normalize_sum(value: str) -> str:
    try:
        value = str(value).replace(" ", "").replace("\u00a0", "").replace(",", ".")
        number = Decimal(value)
        if not number.is_finite():
            return "!invalid"
        return format(number.quantize(Decimal("0.01")), ".2f")
    except InvalidOperation:
        return "!invalid"


def parse_prediction(text: str) -> tuple[list[dict], bool]:
    try:
        data = json.loads(text.strip())
        if not isinstance(data, dict) or set(data) != {"items"} or not isinstance(data["items"], list):
            return [], False
        for item in data["items"]:
            if not isinstance(item, dict) or set(item) != {"name", "sum"}:
                return [], False
            if not isinstance(item["name"], str) or not item["name"].strip():
                return [], False
            if not isinstance(item["sum"], str) or normalize_sum(item["sum"]) == "!invalid":
                return [], False
        return data["items"], True
    except (ValueError, TypeError):
        return [], False


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, lchar in enumerate(left, 1):
        current = [i]
        for j, rchar in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (lchar != rchar)))
        previous = current
    return previous[-1]


def score_predictions(rows: list[dict]) -> dict:
    totals = {kind: Counter() for kind in ("name", "sum", "pair")}
    valid = exact = edits = characters = 0
    for row in rows:
        predicted = row["predicted_items"]
        target = row["target"]["items"]
        valid += int(row["valid_json"])
        def key(item: dict, kind: str):
            name, amount = normalize_name(item["name"]), normalize_sum(item["sum"])
            return name if kind == "name" else amount if kind == "sum" else (name, amount)
        for kind in totals:
            expected = Counter(key(item, kind) for item in target)
            actual = Counter(key(item, kind) for item in predicted)
            totals[kind].update(tp=sum((expected & actual).values()), predicted=sum(actual.values()),
                                target=sum(expected.values()))
        exact += int(row["valid_json"] and [key(item, "pair") for item in target] ==
                     [key(item, "pair") for item in predicted])
        expected_text = "\n".join(normalize_name(item["name"]) for item in target)
        actual_text = "\n".join(normalize_name(item["name"]) for item in predicted)
        edits += edit_distance(expected_text, actual_text)
        characters += len(expected_text)
    result = {"receipts": len(rows), "valid_json_rate": valid / max(1, len(rows)),
              "receipt_exact_match": exact / max(1, len(rows)),
              "product_names_cer": edits / max(1, characters)}
    for kind, counts in totals.items():
        precision = counts["tp"] / max(1, counts["predicted"])
        recall = counts["tp"] / max(1, counts["target"])
        result[kind] = dict(counts, precision=precision, recall=recall,
                            f1=2 * precision * recall / max(1e-12, precision + recall))
    return result
