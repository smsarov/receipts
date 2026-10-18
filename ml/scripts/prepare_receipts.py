"""Download missing receipt images and create deduplicated, auditable MVP targets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.parse
import urllib.request
import unicodedata

from PIL import Image

ML = Path(__file__).resolve().parents[1]
RAW = ML / "data/raw/receipt-ocr-ru"
PREPARED = ML / "data/processed/receipt-ocr-ru"
SPLITS = ("train", "validation", "test")
REPO = "cdek-ocr/receipt-ocr-ru"


def read_rows(split: str) -> list[dict]:
    with (RAW / f"annotations/{split}.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def download_missing() -> dict:
    missing = sorted({row["image"] for split in SPLITS for row in read_rows(split)
                      if not (RAW / row["image"]).is_file()})
    if not missing:
        return {"downloaded": 0}
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=60) as response:
        revision = json.load(response)["sha"]

    def fetch(relative: str) -> str:
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{urllib.parse.quote(relative)}"
        destination = RAW / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    temporary.write_bytes(response.read())
                with Image.open(temporary) as source:
                    source.verify()
                temporary.replace(destination)
                return relative
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 15))
        raise AssertionError("unreachable")

    print(f"Downloading {len(missing)} missing images at {revision}", flush=True)
    failures = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, name): name for name in missing}
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:
                failures[futures[future]] = str(error)
            if completed % 10 == 0 or completed == len(missing):
                print(f"Images downloaded: {completed - len(failures)}/{len(missing)}", flush=True)
    if failures:
        raise RuntimeError(f"Could not download images: {failures}")
    return {"downloaded": len(missing), "image_download_revision": revision}


def money(value: str) -> str:
    value = str(value).strip().replace(" ", "").replace("\u00a0", "").replace(",", ".")
    number = Decimal(value)
    if not number.is_finite() or number != number.quantize(Decimal("0.01")):
        raise ValueError(f"Invalid amount: {value}")
    return format(number, ".2f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    previous_report = ML / "reports/data_preparation.json"
    previous_download = json.loads(previous_report.read_text()).get("download", {}) if previous_report.exists() else {}
    download = download_missing() if args.download else previous_download
    PREPARED.mkdir(parents=True, exist_ok=True)
    report = {"dataset": REPO, "seed": 42, "download": download,
              "target_format": {"items": [{"name": "товар", "sum": "89.99"}]},
              "splits": {}, "excluded": [], "annotation_sha256": {}}
    seen = {split: set() for split in SPLITS}
    candidates = {split: [] for split in SPLITS}
    digest_splits = defaultdict(set)
    for split in SPLITS:
        source = RAW / f"annotations/{split}.jsonl"
        report["annotation_sha256"][split] = hashlib.sha256(source.read_bytes()).hexdigest()
        rows = read_rows(split)
        counts = Counter(raw_rows=len(rows))
        for row in rows:
            reason = None
            image_path = RAW / row["image"]
            if row["image"] in seen[split]:
                reason = "duplicate_reference"
            seen[split].add(row["image"])
            if not reason and not image_path.is_file():
                reason = "missing_image"
            if not reason and not row.get("items"):
                reason = "empty_items"
            if not reason:
                try:
                    items = [{"name": " ".join(unicodedata.normalize("NFKC", str(item["text"])).split()), "sum": money(item["sum"])}
                             for item in row["items"]]
                    if any(not item["name"] for item in items):
                        raise ValueError("Empty product name")
                except (KeyError, ValueError, InvalidOperation):
                    reason = "invalid_item"
            if reason:
                counts[reason] += 1
                report["excluded"].append({"split": split, "image": row["image"], "reason": reason})
                continue
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                width, height = image.size
            target = {"items": items}
            candidates[split].append({"image_id": row["image_id"], "image": row["image"],
                                      "pixel_sha256": digest, "width": width, "height": height,
                                      "target": target,
                                      "target_text": json.dumps(target, ensure_ascii=False, separators=(",", ":"))})
            digest_splits[digest].add(split)
        report["splits"][split] = dict(counts)

    # Keep test and validation independent even when identical pixels have different names.
    retained_hashes = set()
    cleaned = {}
    for split in reversed(SPLITS):
        cleaned[split] = []
        for row in candidates[split]:
            digest = row["pixel_sha256"]
            if digest in retained_hashes:
                report["excluded"].append({"split": split, "image": row["image"], "reason": "duplicate_pixels"})
                continue
            retained_hashes.add(digest)
            cleaned[split].append(row)
        report["splits"][split]["prepared"] = len(cleaned[split])
        path = PREPARED / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in cleaned[split]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["splits"][split]["prepared_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    report["cross_split_pixel_duplicates"] = sum(len(splits) > 1 for splits in digest_splits.values())
    test_ids = sorted(row["image_id"] for row in cleaned["test"])
    fixed = random.Random(42).sample(test_ids, min(24, len(test_ids)))
    report["fixed_test_ids"] = fixed
    output = ML / "reports/data_preparation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"splits": report["splits"], "cross_split_pixel_duplicates": report["cross_split_pixel_duplicates"]},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
