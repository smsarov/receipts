"""Inspect the downloaded cdek-ocr/receipt-ocr-ru JSONL files."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ML_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ML_DIR / "data/raw/receipt-ocr-ru"
IMAGE_DIR = DATA_DIR / "images"
ANNOTATION_DIR = DATA_DIR / "annotations"
REPORT_DIR = ML_DIR / "reports"
SPLITS = ("train", "validation", "test")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {"dataset": "cdek-ocr/receipt-ocr-ru", "splits": {}, "issues": {}}
    all_rows: list[tuple[str, dict]] = []
    all_image_refs: list[str] = []

    for split in SPLITS:
        rows = read_jsonl(ANNOTATION_DIR / f"{split}.jsonl")
        by_image: dict[str, dict] = {}
        for row in rows:
            all_rows.append((split, row))
            all_image_refs.append(row["image"])
            by_image.setdefault(row["image"], row)

        unique_rows = list(by_image.values())
        items = [item for row in unique_rows for item in row.get("items", [])]
        available_images = sum((DATA_DIR / row["image"]).is_file() for row in unique_rows)
        summary["splits"][split] = {
            "jsonl_rows": len(rows),
            "unique_images": len(unique_rows),
            "duplicate_rows": len(rows) - len(unique_rows),
            "records_with_items": sum(bool(row.get("items")) for row in unique_rows),
            "records_without_items": sum(not row.get("items") for row in unique_rows),
            "item_count": len(items),
            "available_referenced_images": available_images,
            "missing_referenced_images": len(unique_rows) - available_images,
        }

    counts = Counter(all_image_refs)
    duplicates = [
        {"image": image, "rows": count}
        for image, count in sorted(counts.items())
        if count > 1
    ]
    unique_rows_by_image: dict[str, tuple[str, dict]] = {}
    for split, row in all_rows:
        unique_rows_by_image.setdefault(row["image"], (split, row))
    unique_rows = list(unique_rows_by_image.values())
    items = [item for _, row in unique_rows for item in row.get("items", [])]
    summary["unique_total"] = len(unique_rows)
    summary["records_with_items"] = sum(bool(row.get("items")) for _, row in unique_rows)
    summary["records_without_items"] = sum(not row.get("items") for _, row in unique_rows)
    summary["item_count"] = len(items)
    summary["items_missing_text"] = sum(not item.get("text") for item in items)
    summary["items_missing_sum"] = sum(not item.get("sum") for item in items)
    summary["items_sum_not_indicated"] = sum(
        item.get("sum") == "не указана" for item in items
    )
    summary["records_with_latin_item_text"] = sum(
        any(re.search(r"[A-Za-z]", item.get("text", "")) for item in row.get("items", []))
        for _, row in unique_rows
    )
    summary["records_without_cyrillic_item_text"] = sum(
        bool(row.get("items"))
        and not any(re.search(r"[А-Яа-яЁё]", item.get("text", "")) for item in row["items"])
        for _, row in unique_rows
    )
    summary["duplicates"] = duplicates

    split_image_sets = {
        split: {row["image"] for row_split, row in all_rows if row_split == split}
        for split in SPLITS
    }
    summary["cross_split_image_overlaps"] = {
        f"{left}/{right}": sorted(split_image_sets[left] & split_image_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }

    output_path = REPORT_DIR / "dataset_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
