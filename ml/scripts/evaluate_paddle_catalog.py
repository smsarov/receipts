"""Evaluate OCR rows matched to a train-only item-name catalog on validation."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import statistics
from rapidfuzz import process, fuzz

from analyze_paddle_ocr import alphabetic, line_groups, number
from receipt_common import ML, normalize_name, normalize_sum, write_json

TRAIN = ML / "data/processed/receipt-ocr-ru-quantity-price/train.jsonl"
VALIDATION = ML / "data/processed/receipt-ocr-ru-quantity-price/validation.jsonl"
OCR = ML / "reports/paddleocr_validation.jsonl"
OUTPUT = ML / "reports/paddleocr_catalog_validation_analysis.json"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OCR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    train = [json.loads(line) for line in TRAIN.read_text().splitlines()]
    validation = {row["image_id"]: row for row in
                  (json.loads(line) for line in VALIDATION.read_text().splitlines())}
    catalog = sorted({normalize_name(item["name"]) for row in train for item in row["target"]["items"]})
    ocr_rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    counts = Counter()
    thresholds = (0.45, 0.55, 0.65, 0.75)
    for output in ocr_rows:
        receipt = validation[output["image_id"]]
        groups = line_groups(output, receipt["height"])
        targets = receipt["target"]["items"]
        assigned = []
        for group in groups:
            query = group["name_text"]
            if len(query) < 3:
                continue
            result = process.extractOne(query, catalog, scorer=fuzz.ratio)
            name, raw_similarity = (result[0], result[1]) if result else ("", 0.0)
            similarity = raw_similarity / 100
            assigned.append({"name": name, "similarity": similarity, "numbers": group["numbers"],
                             "text": group["text"]})
        remaining = Counter(normalize_name(item["name"]) for item in targets)
        for threshold in thresholds:
            matched = Counter(item["name"] for item in assigned if item["similarity"] >= threshold)
            exact = sum(min(count, remaining[name]) for name, count in matched.items())
            counts[f"catalog_exact_{threshold}"] += exact
            counts[f"catalog_predicted_{threshold}"] += sum(matched.values())
        # Numeric field checks only count OCR rows whose train-catalog name exactly matches gold.
        for item in targets:
            gold_name = normalize_name(item["name"])
            candidates = [row for row in assigned if row["name"] == gold_name and row["similarity"] >= 0.45]
            if not candidates:
                continue
            counts["matched_by_catalog"] += 1
            # For repeated identical names, consume rows in order to avoid reusing an OCR row.
            selected = candidates[0]
            assigned.remove(selected)
            for field in ("quantity", "price", "sum"):
                expected = item[field]
                expected = (number(re.match(r"\d+(?:[.,]\d+)?", expected).group())
                            if field == "quantity" and re.match(r"\d+(?:[.,]\d+)?", expected)
                            else normalize_sum(expected) if field != "quantity" else None)
                if expected:
                    counts[field + "_catalog_target"] += 1
                    counts[field + "_catalog_hit"] += int(expected in selected["numbers"])
    target_count = sum(len(row["target"]["items"]) for row in validation.values())
    result = {
        "receipts": len(ocr_rows), "target_items": target_count,
        "train_catalog_unique_names": len(catalog),
        "catalog_exact_name_matches": {
            str(t): {"true_positives": counts[f"catalog_exact_{t}"],
                     "predicted": counts[f"catalog_predicted_{t}"],
                     "precision": counts[f"catalog_exact_{t}"] / max(1, counts[f"catalog_predicted_{t}"]),
                     "recall": counts[f"catalog_exact_{t}"] / target_count}
            for t in thresholds
        },
        "exact_catalog_name_matched_rows": counts["matched_by_catalog"],
        "numeric_recall_given_exact_catalog_name": {
            field: {"hits": counts[field + "_catalog_hit"], "targets": counts[field + "_catalog_target"],
                   "recall": counts[field + "_catalog_hit"] / max(1, counts[field + "_catalog_target"])}
            for field in ("quantity", "price", "sum")
        },
        "note": "OCR groups are classified using only OCR text and train catalog. Exact-match scoring uses validation labels; no validation labels are used to choose a catalog name.",
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
