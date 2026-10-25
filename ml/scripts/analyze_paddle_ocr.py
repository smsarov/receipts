"""Score PaddleOCR line groups against held-out receipt annotations."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import difflib
import json
from pathlib import Path
import re
import statistics
import unicodedata

from receipt_common import ML, normalize_name, normalize_sum, write_json


DATA = ML / "data/processed/receipt-ocr-ru-quantity-price"


def number(value: str) -> str | None:
    try:
        n = Decimal(value.replace(",", "."))
        if n.is_finite():
            return format(n.quantize(Decimal("0.01")), ".2f")
    except InvalidOperation:
        pass
    return None


def alphabetic(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return "".join(re.findall(r"[a-zа-я]+", value))


def line_groups(row: dict, height: int) -> list[dict]:
    polys = row["rec_polys"]
    boxes = row["rec_boxes"]
    texts = row["rec_texts"]
    slopes = []
    for poly in polys:
        if len(poly) >= 4:
            for a, b in ((poly[0], poly[1]), (poly[3], poly[2])):
                dx = float(b[0]) - float(a[0])
                if abs(dx) >= 20:
                    slopes.append((float(b[1]) - float(a[1])) / dx)
    slope = statistics.median(slopes) if slopes else 0.0
    boxes_with_lines = []
    for box, text in zip(boxes, texts):
        if len(box) < 4 or not text.strip():
            continue
        x1, y1, x2, y2 = map(float, box[:4])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        boxes_with_lines.append({"x": cx, "y": cy - slope * cx, "text": text})
    boxes_with_lines.sort(key=lambda item: (item["y"], item["x"]))
    threshold = max(16.0, height * 0.018)
    groups = []
    for item in boxes_with_lines:
        if not groups or item["y"] - groups[-1]["median_y"] > threshold:
            groups.append({"median_y": item["y"], "lines": [item]})
        else:
            group = groups[-1]
            group["lines"].append(item)
            group["median_y"] = statistics.median(line["y"] for line in group["lines"])
    for group in groups:
        group["text"] = " ".join(line["text"] for line in sorted(group["lines"], key=lambda x: x["x"]))
        group["name_text"] = alphabetic(group["text"])
        group["numbers"] = [number(match) for match in re.findall(r"(?<![A-Za-zА-Яа-я])\d{1,7}(?:[.,]\d{1,4})?", group["text"])]
        group["numbers"] = [value for value in group["numbers"] if value]
    return groups


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ML / "reports/paddleocr_validation.jsonl")
    parser.add_argument("--output", type=Path, default=ML / "reports/paddleocr_validation_analysis.json")
    args = parser.parse_args()
    prepared = {row["image_id"]: row for row in
                (json.loads(line) for line in (DATA / "validation.jsonl").read_text().splitlines())}
    output_rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    sums = Counter()
    exact_similarities, assigned_exact, assigned_high = [], 0, 0
    receipts = 0
    examples = []
    for output in output_rows:
        receipt = prepared[output["image_id"]]
        groups = line_groups(output, receipt["height"])
        targets = receipt["target"]["items"]
        pairs = []
        for ti, target in enumerate(targets):
            target_name = alphabetic(target["name"])
            if not target_name:
                continue
            for gi, group in enumerate(groups):
                predicted_name = group["name_text"]
                if not predicted_name:
                    continue
                similarity = difflib.SequenceMatcher(None, target_name, predicted_name, autojunk=False).ratio()
                pairs.append((similarity, ti, gi))
        used_targets, used_groups = set(), set()
        matched = {}
        for similarity, ti, gi in sorted(pairs, reverse=True):
            if ti in used_targets or gi in used_groups or similarity < 0.25:
                continue
            used_targets.add(ti); used_groups.add(gi); matched[ti] = (similarity, groups[gi])
            exact_similarities.append(similarity)
            assigned_exact += int(similarity >= 0.999)
            assigned_high += int(similarity >= 0.80)
        for kind in ("name", "quantity", "price", "sum"):
            sums[kind + "_target"] += len(targets)
        for ti, target in enumerate(targets):
            if ti not in matched:
                continue
            similarity, group = matched[ti]
            if similarity >= 0.80:
                sums["name_fuzzy_80"] += 1
            if similarity >= 0.90:
                sums["name_fuzzy_90"] += 1
            if similarity >= 0.999:
                sums["name_exact"] += 1
            sums["matched_groups"] += 1
            for field, expected in (("quantity", target["quantity"]), ("price", target["price"]), ("sum", target["sum"])):
                norm = normalize_sum(expected) if field != "quantity" else number(re.match(r"\d+(?:[.,]\d+)?", expected).group()) if re.match(r"\d+(?:[.,]\d+)?", expected) else None
                if norm and norm in group["numbers"]:
                    sums[field + "_line_hit"] += 1
                if norm:
                    sums[field + "_valid_target"] += 1
        receipts += 1
        if len(examples) < 10:
            examples.append({"image_id": output["image_id"], "target_count": len(targets),
                             "ocr_group_count": len(groups),
                             "rows": [{"target": targets[ti]["name"], "text": group["text"],
                                       "similarity": round(sim, 3), "numbers": group["numbers"]}
                                      for ti, (sim, group) in matched.items()]})
    result = {"receipts": receipts, "targets": sums["name_target"],
              "ocr_name_matches": sums["matched_groups"],
              "mean_best_name_similarity": statistics.mean(exact_similarities) if exact_similarities else 0.0,
              "name_exact_rate_among_targets": assigned_exact / max(1, sums["name_target"]),
              "name_similarity_at_least_0_80_rate": assigned_high / max(1, sums["name_target"]),
              "name_fuzzy_0_80_rate": sums["name_fuzzy_80"] / max(1, sums["name_target"]),
              "name_fuzzy_0_90_rate": sums["name_fuzzy_90"] / max(1, sums["name_target"]),
              "line_numeric_hits": {field: {"hits": sums[field + "_line_hit"],
                                             "targets": sums[field + "_valid_target"],
                                             "recall": sums[field + "_line_hit"] / max(1, sums[field + "_valid_target"])}
                                    for field in ("quantity", "price", "sum")},
              "examples": examples}
    write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
