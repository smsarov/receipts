"""Create isolated full-field Donut targets from receipt OCR annotations."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from receipt_common import ML, PREPARED

RAW = ML / "data/raw/receipt-ocr-ru"
OUTPUT = ML / "data/processed/receipt-ocr-ru-quantity-price"


def quantity(value: object) -> str:
    value = " ".join(unicodedata.normalize("NFKC", str(value)).split())
    compact = value.replace(" ", "").replace(",", ".")
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(.*)", compact)
    if not match:
        raise ValueError(f"Non-numeric quantity: {value}")
    number = Decimal(match.group(1))
    if not number.is_finite():
        raise ValueError(f"Non-finite quantity: {value}")
    return format(number, "f") + match.group(2).casefold()


def money(value: object) -> str:
    value = str(value).replace(" ", "").replace("\u00a0", "").replace(",", ".")
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError(f"Non-finite amount: {value}")
    return format(number.quantize(Decimal("0.01")), ".2f")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"source": "cdek-ocr/receipt-ocr-ru", "fields": ["name", "quantity", "price", "sum"],
                    "splits": {}, "excluded": [], "annotation_sha256": {}}
    for split in ("train", "validation", "test"):
        annotations_path = RAW / "annotations" / f"{split}.jsonl"
        report["annotation_sha256"][split] = hashlib.sha256(annotations_path.read_bytes()).hexdigest()
        by_id = {}
        with annotations_path.open(encoding="utf-8") as stream:
            for line in stream:
                raw = json.loads(line)
                by_id.setdefault(raw["image_id"], raw)
        source_rows = [json.loads(line) for line in (PREPARED / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        kept, counts = [], Counter()
        for row in source_rows:
            annotation = by_id[row["image_id"]]
            items = []
            try:
                if len(annotation["items"]) != len(row["target"]["items"]):
                    raise ValueError("Prepared and source item counts differ")
                for original, prepared in zip(annotation["items"], row["target"]["items"]):
                    items.append({"name": prepared["name"], "quantity": quantity(original["quantity"]),
                                  "price": money(original["price"]), "sum": money(original["sum"])})
            except (KeyError, ValueError, InvalidOperation) as error:
                report["excluded"].append({"split": split, "image_id": row["image_id"], "reason": str(error)})
                counts["excluded"] += 1
                continue
            target = {"items": items}
            kept.append(dict(row, target=target,
                             target_text=json.dumps(target, ensure_ascii=False, separators=(",", ":"))))
            counts["prepared"] += 1
            counts["items"] += len(items)
        destination = OUTPUT / f"{split}.jsonl"
        destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
        report["splits"][split] = {**counts, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    report_path = OUTPUT / "preparation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=int) + "\n", encoding="utf-8")
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2, default=int))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
