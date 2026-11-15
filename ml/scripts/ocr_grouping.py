"""Group OCR boxes by baseline using the observed text height."""
import re
import statistics
from analyze_paddle_ocr import alphabetic, number

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
    heights = [float(b[3]) - float(b[1]) for b in boxes if len(b) >= 4 and float(b[3]) > float(b[1])]
    threshold = max(3.0, statistics.median(heights) * 0.6) if heights else 10.0
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

